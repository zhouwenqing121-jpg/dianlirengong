import argparse
import csv
import json
import os
import time
import numpy as np
import matplotlib

# Headless-safe backend (CI/servers)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_BASELINE_DENOMINATOR = 1e-12
MIN_BASELINE_TIME = 1.0
BAR_HEIGHT_SCALE = 0.75


class RealisticDispatchSimulator:
    """
    UC + DED + Storage (1h step) with:
      - u[i,t] in {0,1}
      - Pg[i,t] bounded by u*Pmin..u*Pmax
      - ramp constraints (when online; plus startup/shutdown relaxed ramps via big-M style repair)
      - fixed reachable spinning reserve R (MW): gen reachable + storage reachable >= R
      - storage with SOC dynamics, charge/discharge bounds, SOC_end = SOC0
      - power balance with optional transmission losses
      - valve-point cost optional
    Solved by hybrid PSO:
      - continuous variables: Pg, Pch, Pdis (real)
      - binary variables: u via sigmoid -> threshold + repair
    """

    def __init__(
        self,
        load_profile_mw,
        R_fixed=80.0,
        use_losses=True,
        use_valve_point=True,
        seed=42,
    ):
        self.rng = np.random.default_rng(seed)
        self.load = np.array(load_profile_mw, dtype=float)
        self.T = self.load.size
        self.R = float(R_fixed)
        self.use_losses = bool(use_losses)
        self.use_valve_point = bool(use_valve_point)

        # [a,b,c,d,e,Pmin,Pmax,RU,RD,start_cost,stop_cost,min_up,min_down]
        self.gen = np.array(
            [
                [561.0, 7.92, 0.001562, 300.0, 0.0315, 100.0, 600.0, 80.0, 80.0, 2500.0, 500.0, 3, 3],
                [310.0, 7.85, 0.001940, 200.0, 0.0420, 100.0, 400.0, 60.0, 60.0, 1800.0, 400.0, 2, 2],
                [78.0, 7.97, 0.004820, 150.0, 0.0630, 50.0, 200.0, 40.0, 40.0, 900.0, 250.0, 1, 1],
            ],
            dtype=float,
        )

        self.G = self.gen.shape[0]
        self.a, self.b, self.c = self.gen[:, 0], self.gen[:, 1], self.gen[:, 2]
        self.d, self.e = self.gen[:, 3], self.gen[:, 4]
        self.Pmin, self.Pmax = self.gen[:, 5], self.gen[:, 6]
        self.RU, self.RD = self.gen[:, 7], self.gen[:, 8]
        self.start_cost, self.stop_cost = self.gen[:, 9], self.gen[:, 10]
        self.min_up = self.gen[:, 11].astype(int)
        self.min_down = self.gen[:, 12].astype(int)

        self.B = np.array(
            [
                [1.0e-4, 2.0e-5, 1.0e-5],
                [2.0e-5, 1.2e-4, 1.5e-5],
                [1.0e-5, 1.5e-5, 1.5e-4],
            ],
            dtype=float,
        )
        self.B0 = np.zeros(self.G, dtype=float)
        self.B00 = 0.0

        self.Pch_max = 100.0
        self.Pdis_max = 100.0
        self.E_max = 300.0
        self.soc0 = 150.0
        self.eta_ch = 0.95
        self.eta_dis = 0.95
        self.soc_min = 0.0
        self.soc_max = self.E_max
        self.cycle_cost = 1.0

        self.P0 = np.zeros(self.G, dtype=float)

    def loss(self, P):
        if not self.use_losses:
            return np.zeros(P.shape[0], dtype=float)
        l = np.einsum("ij,jk,ik->i", P, self.B, P)
        l += P @ self.B0 + self.B00
        return l

    def gen_cost(self, P, u=None):
        if u is None:
            u = np.ones_like(P, dtype=float)
        u = u.astype(float)
        base = u * (self.a + self.b * P + self.c * (P ** 2))
        if not self.use_valve_point:
            return np.sum(base, axis=1)
        vpe = u * np.abs(self.d * np.sin(self.e * (self.Pmin - P)))
        return np.sum(base + vpe, axis=1)

    def reserve_gen(self, Pg):
        headroom = self.Pmax - Pg
        return np.sum(np.minimum(headroom, self.RU), axis=1)

    def reserve_storage(self, Pdis_t, soc_t):
        power_cap = np.maximum(0.0, self.Pdis_max - Pdis_t)
        energy_cap = np.maximum(0.0, soc_t * self.eta_dis)
        return np.minimum(power_cap, energy_cap)

    def _gen_priority(self):
        ref_p = 0.5 * (self.Pmin + self.Pmax)
        return np.argsort(self.b + 2.0 * self.c * ref_p)

    def _build_seed_schedule(self, online_units):
        u_seed = np.zeros((self.T, self.G), dtype=int)
        u_seed[:, online_units] = 1

        Pg_seed = np.zeros((self.T, self.G), dtype=float)
        order = [g for g in self._gen_priority() if g in online_units]

        for t in range(self.T):
            cur = np.zeros(self.G, dtype=float)
            if len(order) > 0:
                cur[order] = self.Pmin[order]
            residual = self.load[t] - np.sum(cur)

            if residual > 0.0:
                for g in order:
                    room = self.Pmax[g] - cur[g]
                    take = min(residual, room)
                    cur[g] += take
                    residual -= take
                    if residual <= 1e-9:
                        break

            Pg_seed[t, :] = cur

        Pch_seed = np.zeros((self.T,), dtype=float)
        Pdis_seed = np.zeros((self.T,), dtype=float)
        z_seed = np.full((self.T, self.G), -8.0, dtype=float)
        z_seed[:, online_units] = 8.0
        return z_seed, Pg_seed, Pch_seed, Pdis_seed

    def inject_warm_starts(self, z, Pg, Pch, Pdis):
        seed_sets = [
            [1],
            [0, 1],
            [0, 1, 2],
        ]
        k = min(z.shape[0], len(seed_sets))
        for i in range(k):
            z_i, Pg_i, Pch_i, Pdis_i = self._build_seed_schedule(seed_sets[i])
            z[i] = z_i
            Pg[i] = Pg_i
            Pch[i] = Pch_i
            Pdis[i] = Pdis_i

    def decode_u(self, z):
        z = np.clip(z, -60, 60)
        prob = 1.0 / (1.0 + np.exp(-z))
        u = (prob > 0.5).astype(int)
        return self.repair_min_up_down(u)

    def repair_min_up_down(self, u):
        N = u.shape[0]
        for n in range(N):
            for g in range(self.G):
                t = 0
                while t < self.T:
                    if u[n, t, g] == 1:
                        t2 = t
                        while t2 < self.T and u[n, t2, g] == 1:
                            t2 += 1
                        run = t2 - t
                        if run < self.min_up[g]:
                            end = min(self.T, t + self.min_up[g])
                            u[n, t:end, g] = 1
                            t = end
                        else:
                            t = t2
                    else:
                        t2 = t
                        while t2 < self.T and u[n, t2, g] == 0:
                            t2 += 1
                        run = t2 - t
                        if run < self.min_down[g]:
                            end = min(self.T, t + self.min_down[g])
                            u[n, t:end, g] = 0
                            t = end
                        else:
                            t = t2
        return u

    def repair_storage(self, Pch, Pdis):
        Pch = np.clip(Pch, 0.0, self.Pch_max)
        Pdis = np.clip(Pdis, 0.0, self.Pdis_max)

        net = Pdis - Pch
        Pdis = np.maximum(net, 0.0)
        Pch = np.maximum(-net, 0.0)

        N, T = Pch.shape
        soc = np.full((N,), self.soc0, dtype=float)

        for t in range(T):
            max_ch = np.maximum(0.0, (self.soc_max - soc) / self.eta_ch)
            Pch[:, t] = np.minimum(Pch[:, t], max_ch)

            max_dis = np.maximum(0.0, soc * self.eta_dis)
            Pdis[:, t] = np.minimum(Pdis[:, t], max_dis)

            soc = soc + self.eta_ch * Pch[:, t] - (1.0 / self.eta_dis) * Pdis[:, t]
            soc = np.clip(soc, self.soc_min, self.soc_max)

        gap = soc - self.soc0
        t = T - 1
        need_dis = np.maximum(0.0, gap * self.eta_dis)
        need_ch = np.maximum(0.0, (-gap) / self.eta_ch)

        Pdis[:, t] = np.clip(Pdis[:, t] + need_dis, 0.0, self.Pdis_max)
        Pch[:, t] = np.clip(Pch[:, t] + need_ch, 0.0, self.Pch_max)

        soc2 = np.full((N,), self.soc0, dtype=float)
        for tt in range(T):
            max_ch = np.maximum(0.0, (self.soc_max - soc2) / self.eta_ch)
            Pch[:, tt] = np.minimum(Pch[:, tt], max_ch)
            max_dis = np.maximum(0.0, soc2 * self.eta_dis)
            Pdis[:, tt] = np.minimum(Pdis[:, tt], max_dis)
            soc2 = soc2 + self.eta_ch * Pch[:, tt] - (1.0 / self.eta_dis) * Pdis[:, tt]
            soc2 = np.clip(soc2, self.soc_min, self.soc_max)

        return Pch, Pdis, soc2

    def repair_generators(self, Pg, u, Pch, Pdis):
        N = Pg.shape[0]
        Pg = Pg.astype(float)

        Pmin_row = self.Pmin.reshape(1, -1)
        Pmax_row = self.Pmax.reshape(1, -1)

        for t in range(self.T):
            lo = u[:, t, :] * Pmin_row
            hi = u[:, t, :] * Pmax_row
            Pg[:, t, :] = np.clip(Pg[:, t, :], lo, hi)

        for t in range(self.T):
            prevP = self.P0.reshape(1, -1).repeat(N, axis=0) if t == 0 else Pg[:, t - 1, :]
            prevU = np.zeros((N, self.G), dtype=int) if t == 0 else u[:, t - 1, :]
            curU = u[:, t, :]

            both_on = (prevU == 1) & (curU == 1)
            upper = prevP + self.RU
            lower = prevP - self.RD

            Pg[:, t, :] = np.where(both_on, np.minimum(Pg[:, t, :], upper), Pg[:, t, :])
            Pg[:, t, :] = np.where(both_on, np.maximum(Pg[:, t, :], lower), Pg[:, t, :])
            Pg[:, t, :] = np.where(curU == 0, 0.0, Pg[:, t, :])

            start_mask = (prevU == 0) & (curU == 1)
            Pg[:, t, :] = np.where(start_mask, np.maximum(Pg[:, t, :], Pmin_row), Pg[:, t, :])

            lo = curU * Pmin_row
            hi = curU * Pmax_row
            Pg[:, t, :] = np.clip(Pg[:, t, :], lo, hi)

        for t in range(self.T):
            P_t = Pg[:, t, :].copy()

            offline = np.where(np.sum(u[:, t, :], axis=1) == 0)[0]
            if offline.size > 0:
                u[offline, t, 2] = 1
                P_t[offline, 2] = self.Pmin[2]

            for _ in range(20):
                loss = self.loss(P_t)
                rhs = self.load[t] + loss
                lhs = np.sum(P_t, axis=1) + Pdis[:, t] - Pch[:, t]
                mismatch = rhs - lhs

                idx = np.where(np.abs(mismatch) > 1e-6)[0]
                if idx.size == 0:
                    break

                idx_pos = idx[mismatch[idx] > 0]
                if idx_pos.size > 0:
                    on_pos = u[idx_pos, t, :] == 1
                    headroom = (self.Pmax - P_t[idx_pos]) * on_pos
                    row_sum = np.sum(headroom, axis=1)
                    valid = row_sum > 1e-12
                    if np.any(valid):
                        k = idx_pos[valid]
                        hr = headroom[valid]
                        denom = row_sum[valid].reshape(-1, 1)
                        share = np.divide(hr, denom, out=np.zeros_like(hr), where=denom > 1e-12)
                        P_t[k] += mismatch[k].reshape(-1, 1) * share

                idx_neg = idx[mismatch[idx] < 0]
                if idx_neg.size > 0:
                    on_neg = u[idx_neg, t, :] == 1
                    footroom = (P_t[idx_neg] - self.Pmin) * on_neg
                    row_sum = np.sum(footroom, axis=1)
                    valid = row_sum > 1e-12
                    if np.any(valid):
                        k = idx_neg[valid]
                        fr = footroom[valid]
                        denom = row_sum[valid].reshape(-1, 1)
                        share = np.divide(fr, denom, out=np.zeros_like(fr), where=denom > 1e-12)
                        P_t[k] += mismatch[k].reshape(-1, 1) * share

                P_t = np.nan_to_num(P_t, nan=0.0, posinf=float(np.max(self.Pmax)), neginf=0.0)
                lo = u[:, t, :] * self.Pmin.reshape(1, -1)
                hi = u[:, t, :] * self.Pmax.reshape(1, -1)
                P_t = np.clip(P_t, lo, hi)

            Pg[:, t, :] = P_t

        return Pg, u

    def startup_shutdown_cost(self, u):
        N = u.shape[0]
        cost = np.zeros(N, dtype=float)
        prev = np.zeros((N, self.G), dtype=int)
        for t in range(self.T):
            cur = u[:, t, :]
            start = (prev == 0) & (cur == 1)
            stop = (prev == 1) & (cur == 0)
            cost += np.sum(start * self.start_cost + stop * self.stop_cost, axis=1)
            prev = cur
        return cost

    def evaluate(self, Pg, u, Pch, Pdis):
        N = Pg.shape[0]

        gen_cost_total = np.zeros(N, dtype=float)
        balance_mm = np.zeros(N, dtype=float)
        reserve_vio = np.zeros(N, dtype=float)
        ramp_vio = np.zeros(N, dtype=float)

        soc = np.zeros((N, self.T + 1), dtype=float)
        soc[:, 0] = self.soc0
        for t in range(self.T):
            soc[:, t + 1] = soc[:, t] + self.eta_ch * Pch[:, t] - (1.0 / self.eta_dis) * Pdis[:, t]
            soc[:, t + 1] = np.clip(soc[:, t + 1], self.soc_min, self.soc_max)

        prevP = self.P0.reshape(1, -1).repeat(N, axis=0)
        prevU = np.zeros((N, self.G), dtype=int)

        for t in range(self.T):
            P_t = Pg[:, t, :]
            curU = u[:, t, :]
            gen_cost_total += self.gen_cost(P_t, curU)

            loss = self.loss(P_t)
            lhs = np.sum(P_t, axis=1) + Pdis[:, t] - Pch[:, t]
            rhs = self.load[t] + loss
            balance_mm += np.abs(lhs - rhs)

            rgen = self.reserve_gen(P_t)
            rst = self.reserve_storage(Pdis[:, t], soc[:, t])
            rtot = rgen + rst
            reserve_vio += np.maximum(0.0, self.R - rtot)

            both_on = (prevU == 1) & (curU == 1)
            ramp_up = np.maximum(0.0, (P_t - prevP) - self.RU) * both_on
            ramp_dn = np.maximum(0.0, (prevP - P_t) - self.RD) * both_on
            ramp_vio += np.sum(ramp_up + ramp_dn, axis=1)
            prevP, prevU = P_t, curU

        throughput = np.sum(Pch + Pdis, axis=1)
        storage_cost = self.cycle_cost * throughput

        su_sd_cost = self.startup_shutdown_cost(u)

        pen_balance = 1e9
        pen_reserve = 1e7
        pen_ramp = 1e7
        pen_soc_end = 1e7

        soc_end_gap = np.abs(soc[:, -1] - self.soc0)

        fitness = (
            gen_cost_total
            + storage_cost
            + su_sd_cost
            + pen_balance * np.maximum(0.0, balance_mm - 1e-4)
            + pen_reserve * reserve_vio
            + pen_ramp * ramp_vio
            + pen_soc_end * soc_end_gap
        )

        details = dict(
            gen_cost=gen_cost_total,
            storage_cost=storage_cost,
            su_sd_cost=su_sd_cost,
            balance_mm=balance_mm,
            reserve_vio=reserve_vio,
            ramp_vio=ramp_vio,
            soc_end=soc[:, -1],
            soc=soc,
        )
        return fitness, details

    def run(self, num_particles=120, max_iter=600):
        N, T, G = num_particles, self.T, self.G

        z = self.rng.normal(0.0, 1.0, size=(N, T, G))
        Pg = self.rng.uniform(0.0, self.Pmax, size=(N, T, G))
        Pch = self.rng.uniform(0.0, self.Pch_max, size=(N, T))
        Pdis = self.rng.uniform(0.0, self.Pdis_max, size=(N, T))
        self.inject_warm_starts(z, Pg, Pch, Pdis)

        Vz = np.zeros_like(z)
        Vg = np.zeros_like(Pg)
        Vch = np.zeros_like(Pch)
        Vdis = np.zeros_like(Pdis)

        w_max, w_min = 0.9, 0.4
        c1, c2 = 1.6, 1.6

        vmax_z = 1.0
        vmax_g = (0.25 * self.Pmax).reshape(1, 1, G)
        vmax_ch = 0.25 * self.Pch_max
        vmax_dis = 0.25 * self.Pdis_max

        u = self.decode_u(z)
        Pch, Pdis, _ = self.repair_storage(Pch, Pdis)
        Pg, u = self.repair_generators(Pg, u, Pch, Pdis)
        fit, _ = self.evaluate(Pg, u, Pch, Pdis)

        pbest = (z.copy(), Pg.copy(), Pch.copy(), Pdis.copy())
        pbest_fit = fit.copy()

        gi = int(np.argmin(pbest_fit))
        gbest = (
            pbest[0][gi].copy(),
            pbest[1][gi].copy(),
            pbest[2][gi].copy(),
            pbest[3][gi].copy(),
        )
        gbest_fit = float(pbest_fit[gi])

        hist = {k: [] for k in ["fitness", "gen_cost", "storage_cost", "su_sd_cost", "balance_mm", "reserve_vio", "ramp_vio", "soc_end"]}

        for it in range(max_iter):
            w = w_max - (w_max - w_min) * (it / max_iter)

            r1z = self.rng.random(z.shape)
            r2z = self.rng.random(z.shape)
            r1g = self.rng.random(Pg.shape)
            r2g = self.rng.random(Pg.shape)
            r1c = self.rng.random(Pch.shape)
            r2c = self.rng.random(Pch.shape)
            r1d = self.rng.random(Pdis.shape)
            r2d = self.rng.random(Pdis.shape)

            Vz = w * Vz + c1 * r1z * (pbest[0] - z) + c2 * r2z * (gbest[0] - z)
            Vz = np.clip(Vz, -vmax_z, vmax_z)
            Vg = w * Vg + c1 * r1g * (pbest[1] - Pg) + c2 * r2g * (gbest[1] - Pg)
            Vg = np.clip(Vg, -vmax_g, vmax_g)
            Vch = w * Vch + c1 * r1c * (pbest[2] - Pch) + c2 * r2c * (gbest[2] - Pch)
            Vch = np.clip(Vch, -vmax_ch, vmax_ch)
            Vdis = w * Vdis + c1 * r1d * (pbest[3] - Pdis) + c2 * r2d * (gbest[3] - Pdis)
            Vdis = np.clip(Vdis, -vmax_dis, vmax_dis)

            z = z + Vz
            Pg = Pg + Vg
            Pch = Pch + Vch
            Pdis = Pdis + Vdis

            u = self.decode_u(z)
            Pch, Pdis, _ = self.repair_storage(Pch, Pdis)
            Pg, u = self.repair_generators(Pg, u, Pch, Pdis)

            fit, _ = self.evaluate(Pg, u, Pch, Pdis)

            improve = fit < pbest_fit
            pbest_fit[improve] = fit[improve]
            pbest = (
                np.where(improve.reshape(-1, 1, 1), z, pbest[0]),
                np.where(improve.reshape(-1, 1, 1), Pg, pbest[1]),
                np.where(improve.reshape(-1, 1), Pch, pbest[2]),
                np.where(improve.reshape(-1, 1), Pdis, pbest[3]),
            )

            gi = int(np.argmin(pbest_fit))
            if float(pbest_fit[gi]) < gbest_fit:
                gbest_fit = float(pbest_fit[gi])
                gbest = (
                    pbest[0][gi].copy(),
                    pbest[1][gi].copy(),
                    pbest[2][gi].copy(),
                    pbest[3][gi].copy(),
                )

            gu = self.decode_u(gbest[0].reshape(1, T, G))
            gPch, gPdis, _ = self.repair_storage(gbest[2].reshape(1, T), gbest[3].reshape(1, T))
            gPg, gu = self.repair_generators(gbest[1].reshape(1, T, G), gu, gPch, gPdis)
            _, gdet = self.evaluate(gPg, gu, gPch, gPdis)

            hist["fitness"].append(gbest_fit)
            hist["gen_cost"].append(float(gdet["gen_cost"][0]))
            hist["storage_cost"].append(float(gdet["storage_cost"][0]))
            hist["su_sd_cost"].append(float(gdet["su_sd_cost"][0]))
            hist["balance_mm"].append(float(gdet["balance_mm"][0]))
            hist["reserve_vio"].append(float(gdet["reserve_vio"][0]))
            hist["ramp_vio"].append(float(gdet["ramp_vio"][0]))
            hist["soc_end"].append(float(gdet["soc_end"][0]))

        z_best, Pg_best, Pch_best, Pdis_best = gbest
        u_best = self.decode_u(z_best.reshape(1, T, G))[0]
        Pch_best, Pdis_best, _ = self.repair_storage(Pch_best.reshape(1, T), Pdis_best.reshape(1, T))
        Pg_best, u_best2 = self.repair_generators(
            Pg_best.reshape(1, T, G), u_best.reshape(1, T, G), Pch_best, Pdis_best
        )
        Pg_best = Pg_best[0]
        u_best2 = u_best2[0]
        Pch_best = Pch_best[0]
        Pdis_best = Pdis_best[0]

        return (u_best2, Pg_best, Pch_best, Pdis_best), hist


def _set_plot_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 2.0,
        }
    )


def _save_dual_resolution(fig, out_dir, base_name):
    debug_path = os.path.join(out_dir, f"{base_name}_debug.png")
    hd_path = os.path.join(out_dir, f"{base_name}_hd.png")
    fig.savefig(debug_path, dpi=120, bbox_inches="tight")
    fig.savefig(hd_path, dpi=300, bbox_inches="tight")


def _compute_series(sim, load_profile, u, Pg, Pch, Pdis, soc):
    T = sim.T
    losses = sim.loss(Pg)
    lhs = Pg.sum(axis=1) + Pdis - Pch
    rhs = load_profile + losses
    balance_residual = lhs - rhs

    reserve_total = sim.reserve_gen(Pg) + sim.reserve_storage(Pdis, soc[:T])
    reserve_margin = reserve_total - sim.R

    ramp_vio_by_hour = np.zeros(T, dtype=float)
    for t in range(1, T):
        both_on = (u[t - 1, :] == 1) & (u[t, :] == 1)
        ramp_up = np.maximum(0.0, (Pg[t, :] - Pg[t - 1, :]) - sim.RU) * both_on
        ramp_dn = np.maximum(0.0, (Pg[t - 1, :] - Pg[t, :]) - sim.RD) * both_on
        ramp_vio_by_hour[t] = np.sum(ramp_up + ramp_dn)
    return losses, balance_residual, reserve_total, reserve_margin, ramp_vio_by_hour


def _extract_soc_array(soc_value):
    if isinstance(soc_value, np.ndarray) and soc_value.ndim > 1:
        return soc_value[0]
    return np.array(soc_value, dtype=float)


def build_unified_result(method_name, sim, load_profile, u, Pg, Pch, Pdis, det, solve_time_sec):
    soc = _extract_soc_array(det["soc"])
    _, balance_residual, reserve_total, reserve_margin, ramp_vio_by_hour = _compute_series(
        sim, load_profile, u, Pg, Pch, Pdis, soc
    )

    if "total_cost" in det:
        total_cost = float(det["total_cost"])
    elif isinstance(det.get("gen_cost"), np.ndarray):
        total_cost = float(det["gen_cost"][0] + det["storage_cost"][0] + det["su_sd_cost"][0])
    else:
        total_cost = float(det.get("gen_cost", 0.0)) + float(det.get("storage_cost", 0.0)) + float(det.get("su_sd_cost", 0.0))

    balance_mm = float(np.sum(np.abs(balance_residual)))
    reserve_vio = float(np.sum(np.maximum(0.0, -reserve_margin)))
    ramp_vio = float(np.sum(ramp_vio_by_hour))

    return {
        "method": method_name,
        "T": int(sim.T),
        "G": int(sim.G),
        "load_mw": np.array(load_profile, dtype=float),
        "u": np.array(u, dtype=int),
        "Pg_mw": np.array(Pg, dtype=float),
        "Pch_mw": np.array(Pch, dtype=float),
        "Pdis_mw": np.array(Pdis, dtype=float),
        "soc_mwh": np.array(soc, dtype=float),
        "total_cost_usd": total_cost,
        "balance_residual_mw": balance_residual,
        "balance_mm_mw": balance_mm,
        "reserve_total_mw": reserve_total,
        "reserve_margin_mw": reserve_margin,
        "reserve_violation_mw": reserve_vio,
        "ramp_violation_by_hour_mw": ramp_vio_by_hour,
        "ramp_violation_mw": ramp_vio,
        "soc_end_mwh": float(soc[-1]),
        "solve_time_sec": float(solve_time_sec),
        "constraints_ok": {
            "balance": bool(np.max(np.abs(balance_residual)) <= 1e-3),
            "reserve": bool(np.min(reserve_margin) >= -1e-6),
            "ramp": bool(ramp_vio <= 1e-6),
            "soc_end": bool(abs(soc[-1] - sim.soc0) <= 1e-6),
        },
    }


def load_unified_result_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    required_keys = ["method", "load_mw", "Pg_mw", "Pch_mw", "Pdis_mw", "soc_mwh", "total_cost_usd", "solve_time_sec"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"Missing keys in {path}: {', '.join(missing)}")
    out = dict(data)
    for k in ["load_mw", "Pg_mw", "Pch_mw", "Pdis_mw", "soc_mwh", "balance_residual_mw", "reserve_margin_mw"]:
        if k in out:
            out[k] = np.array(out[k], dtype=float)
    out["total_cost_usd"] = float(out["total_cost_usd"])
    out["solve_time_sec"] = float(out["solve_time_sec"])
    if "T" not in out:
        out["T"] = int(len(out["load_mw"]))
    if "G" not in out:
        out["G"] = int(out["Pg_mw"].shape[1]) if out["Pg_mw"].ndim == 2 else 1
    if "balance_mm_mw" not in out and "balance_residual_mw" in out:
        out["balance_mm_mw"] = float(np.sum(np.abs(out["balance_residual_mw"])))
    if "reserve_violation_mw" not in out and "reserve_margin_mw" in out:
        out["reserve_violation_mw"] = float(np.sum(np.maximum(0.0, -out["reserve_margin_mw"])))
    if "ramp_violation_mw" not in out:
        out["ramp_violation_mw"] = float(np.sum(out.get("ramp_violation_by_hour_mw", 0.0)))
    return out


def plot_dispatch_figures(sim, unified_result, hist, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    _set_plot_style()
    T, G = sim.T, sim.G
    hours = np.arange(T)
    Pg = unified_result["Pg_mw"]
    Pch = unified_result["Pch_mw"]
    Pdis = unified_result["Pdis_mw"]
    soc = unified_result["soc_mwh"]
    load_profile = unified_result["load_mw"]
    balance_residual = unified_result["balance_residual_mw"]
    reserve_margin = unified_result["reserve_margin_mw"]
    reserve_total = unified_result["reserve_total_mw"]
    method = unified_result["method"]
    ok = unified_result["constraints_ok"]

    fig = plt.figure(figsize=(10, 5.2))
    labels = [f"G{i + 1}" for i in range(G)]
    plt.stackplot(hours, Pg.T, labels=labels, alpha=0.85)
    plt.plot(hours, Pdis - Pch, color="#111827", linestyle="--", label="Storage Net (Pdis-Pch)")
    plt.plot(hours, load_profile, color="#dc2626", marker=".", label="Load")
    txt = f"Balance:{'OK' if ok['balance'] else 'FAIL'} | Reserve:{'OK' if ok['reserve'] else 'FAIL'} | Ramp:{'OK' if ok['ramp'] else 'FAIL'}"
    plt.text(0.01, 0.99, txt, transform=plt.gca().transAxes, va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.9))
    plt.title(f"01 Dispatch Main ({method})")
    plt.xlabel("Hour")
    plt.ylabel("Power (MW)")
    plt.legend(loc="upper left", ncol=2)
    _save_dual_resolution(fig, out_dir, "01_dispatch_main")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 4.5))
    plt.bar(hours - 0.18, Pch, width=0.36, label="Charge Pch", color="#2563eb")
    plt.bar(hours + 0.18, Pdis, width=0.36, label="Discharge Pdis", color="#f97316")
    plt.title("02 Storage Charge/Discharge Power")
    plt.xlabel("Hour")
    plt.ylabel("Power (MW)")
    plt.legend(loc="upper right")
    _save_dual_resolution(fig, out_dir, "02_storage_power")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 4.5))
    plt.plot(np.arange(T + 1), soc, marker="o", color="#0f766e")
    plt.axhline(sim.soc0, color="k", linestyle="--", alpha=0.7, label=f"SOC target={sim.soc0:.1f}")
    plt.text(0.01, 0.99, f"SOC End:{'OK' if ok['soc_end'] else 'FAIL'}", transform=plt.gca().transAxes, va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.9))
    plt.title("03 Storage SOC")
    plt.xlabel("Hour")
    plt.ylabel("SOC (MWh)")
    plt.legend(loc="best")
    _save_dual_resolution(fig, out_dir, "03_soc")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 4.5))
    plt.axhline(0.0, color="k", linewidth=1.2)
    plt.plot(hours, balance_residual, marker="o", color="#7c3aed")
    plt.text(
        0.01,
        0.99,
        f"max|res|={np.max(np.abs(balance_residual)):.3e} MW ({'OK' if ok['balance'] else 'FAIL'})",
        transform=plt.gca().transAxes,
        va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.9),
    )
    plt.title("04 Power Balance Residual")
    plt.xlabel("Hour")
    plt.ylabel("Residual (MW)")
    _save_dual_resolution(fig, out_dir, "04_balance_residual")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 4.5))
    plt.plot(hours, reserve_total, marker="o", label="Reachable reserve", color="#0284c7")
    plt.axhline(sim.R, color="#dc2626", linestyle="--", label=f"Requirement R={sim.R:.0f} MW")
    plt.fill_between(hours, reserve_total, sim.R, where=(reserve_margin < 0), color="#dc2626", alpha=0.15, label="Violation")
    plt.text(
        0.01,
        0.99,
        f"min margin={np.min(reserve_margin):.3e} MW ({'OK' if ok['reserve'] else 'FAIL'})",
        transform=plt.gca().transAxes,
        va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.9),
    )
    plt.title("05 Reserve Margin")
    plt.xlabel("Hour")
    plt.ylabel("Reserve (MW)")
    plt.legend(loc="best")
    _save_dual_resolution(fig, out_dir, "05_reserve_margin")
    plt.close(fig)

    if hist and len(hist.get("gen_cost", [])) > 0:
        fig = plt.figure(figsize=(9, 4.5))
        total_cost_hist = np.array(hist["gen_cost"]) + np.array(hist["storage_cost"]) + np.array(hist["su_sd_cost"])
        plt.plot(total_cost_hist, color="#374151")
        plt.title("06 Convergence: Total Cost")
        plt.xlabel("Iteration")
        plt.ylabel("Cost ($)")
        _save_dual_resolution(fig, out_dir, "06_convergence")
        plt.close(fig)


def generate_comparison_artifacts(method_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    _set_plot_style()
    if len(method_results) == 0:
        raise ValueError("method_results must not be empty.")
    methods = [r["method"] for r in method_results]
    if "MILP" in methods:
        baseline_idx = methods.index("MILP")
    else:
        baseline_idx = 0
        print(f"[WARN] MILP baseline not found, using '{methods[0]}' as comparison baseline.")
    baseline_cost = method_results[baseline_idx]["total_cost_usd"]
    baseline_time = (
        method_results[baseline_idx]["solve_time_sec"]
        if method_results[baseline_idx]["solve_time_sec"] > 0
        else MIN_BASELINE_TIME
    )
    baseline_balance = max(method_results[baseline_idx].get("balance_mm_mw", 0.0), MIN_BASELINE_DENOMINATOR)
    baseline_reserve = max(method_results[baseline_idx].get("reserve_violation_mw", 0.0), MIN_BASELINE_DENOMINATOR)
    baseline_ramp = max(method_results[baseline_idx].get("ramp_violation_mw", 0.0), MIN_BASELINE_DENOMINATOR)

    rows = []
    for r in method_results:
        cost_save = (baseline_cost - r["total_cost_usd"]) / baseline_cost * 100.0 if baseline_cost != 0 else 0.0
        rows.append(
            {
                "method": r["method"],
                "total_cost_usd": float(r["total_cost_usd"]),
                "balance_mm_mw": float(r.get("balance_mm_mw", np.nan)),
                "reserve_violation_mw": float(r.get("reserve_violation_mw", np.nan)),
                "ramp_violation_mw": float(r.get("ramp_violation_mw", np.nan)),
                "solve_time_sec": float(r["solve_time_sec"]),
                "cost_saving_pct_vs_baseline": float(cost_save),
            }
        )

    csv_path = os.path.join(out_dir, "08_method_comparison_table.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    categories = [
        "Cost",
        "BalanceMM",
        "ReserveVio",
        "RampVio",
        "Time",
    ]
    values = []
    for r in rows:
        values.append(
            [
                (r["total_cost_usd"] / baseline_cost * 100.0) if baseline_cost != 0 else 0.0,
                (r["balance_mm_mw"] / baseline_balance * 100.0) if baseline_balance != 0 else 0.0,
                (r["reserve_violation_mw"] / baseline_reserve * 100.0) if baseline_reserve != 0 else 0.0,
                (r["ramp_violation_mw"] / baseline_ramp * 100.0) if baseline_ramp != 0 else 0.0,
                (r["solve_time_sec"] / baseline_time * 100.0) if baseline_time != 0 else 0.0,
            ]
        )
    values = np.array(values)

    fig = plt.figure(figsize=(11, 6))
    y = np.arange(len(categories))
    bar_h = BAR_HEIGHT_SCALE / len(methods)
    for i, m in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * bar_h
        plt.barh(y + offset, values[i], height=bar_h, label=m)
    plt.axvline(100.0, color="#111827", linestyle="--", linewidth=1.2, label="Baseline=100%")
    plt.yticks(y, categories)
    plt.xlabel("Relative value (%)")
    plt.title("07 Method Comparison (Grouped Horizontal Bars)")
    plt.legend(loc="lower right")
    _save_dual_resolution(fig, out_dir, "07_method_comparison")
    plt.close(fig)
    return rows


def export_consistency_report(method_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    load_ref = method_results[0]["load_mw"]
    same_load = all(np.allclose(r["load_mw"], load_ref, atol=1e-6) for r in method_results)
    same_horizon = all(int(r["T"]) == int(method_results[0]["T"]) for r in method_results)

    best_cost = min(method_results, key=lambda r: r["total_cost_usd"])["method"]
    fastest = min(method_results, key=lambda r: r["solve_time_sec"])["method"]
    least_vio = min(
        method_results,
        key=lambda r: (r.get("balance_mm_mw", 0.0) + r.get("reserve_violation_mw", 0.0) + r.get("ramp_violation_mw", 0.0)),
    )["method"]

    report_path = os.path.join(out_dir, "08_consistency_check.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Consistency Check Report\n")
        f.write("========================\n")
        f.write(f"same_load_profile: {same_load}\n")
        f.write(f"same_horizon: {same_horizon}\n")
        f.write(f"lowest_cost_method: {best_cost}\n")
        f.write(f"fastest_method: {fastest}\n")
        f.write(f"least_total_violation_method: {least_vio}\n")
    return report_path


def parse_args():
    parser = argparse.ArgumentParser(description="UC+DED dispatch simulator (PSO) with unified comparison artifacts.")
    parser.add_argument(
        "--mode",
        choices=["aligned", "realistic"],
        default="aligned",
        help="aligned: 与MILP基线对齐（无网损/无阀点）；realistic: 启用网损与阀点。",
    )
    parser.add_argument("--num-particles", type=int, default=120, help="PSO粒子数")
    parser.add_argument("--max-iter", type=int, default=200, help="PSO迭代次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--milp-json", type=str, default="", help="Optional MILP unified-result JSON path")
    parser.add_argument("--dl-json", type=str, default="", help="Optional deep-learning unified-result JSON path")
    parser.add_argument("--rl-json", type=str, default="", help="Optional RL unified-result JSON path (defaults to current PSO run)")
    parser.add_argument("--out-root", type=str, default="outputs", help="Output root directory")
    return parser.parse_args()


def main():
    args = parse_args()
    PLoad = np.array(
        [
            272.563258729604,
            264.888219494172,
            261.799350631702,
            258.697623468531,
            253.748073897436,
            257.3254,
            267.583538636364,
            286.960179242424,
            290.062693484849,
            312.776684848485,
            309.970979242424,
            306.384639545455,
            302.196933636364,
            295.091917121212,
            292.778442727273,
            293.351041666667,
            298.876115909091,
            317.191548181818,
            319.602692121212,
            323.599087121212,
            317.339661212121,
            305.464569545455,
            298.74788265035,
            289.784661613054,
        ],
        dtype=float,
    )

    aligned_mode = args.mode == "aligned"
    sim = RealisticDispatchSimulator(
        PLoad,
        R_fixed=80.0,
        use_losses=not aligned_mode,
        use_valve_point=not aligned_mode,
        seed=args.seed,
    )
    t0 = time.perf_counter()
    (u, Pg, Pch, Pdis), hist = sim.run(num_particles=args.num_particles, max_iter=args.max_iter)
    solve_time = time.perf_counter() - t0

    fit, det = sim.evaluate(
        Pg.reshape(1, sim.T, sim.G),
        u.reshape(1, sim.T, sim.G),
        Pch.reshape(1, sim.T),
        Pdis.reshape(1, sim.T),
    )
    det = {k: float(v[0]) if isinstance(v, np.ndarray) and v.ndim == 1 else v for k, v in det.items()}
    det["total_cost"] = float(det["gen_cost"] + det["storage_cost"] + det["su_sd_cost"])

    pso_result = build_unified_result("RL-PSO", sim, PLoad, u, Pg, Pch, Pdis, det, solve_time)

    print("=" * 95)
    if aligned_mode:
        print("对齐调度仿真（UC + DED + 储能 + 备用 + 爬坡，无网损/无阀点，与MILP基线一致）")
    else:
        print("真实调度仿真（UC + DED + 储能 + 备用 + 爬坡 + 网损 + 阀点）")
    print("=" * 95)
    print(f"Best fitness:         {float(fit[0]):.4f}")
    print(f"Solve time:           {solve_time:.4f} s")
    print(f"Total gen cost:      ${det['gen_cost']:.4f}")
    print(f"Storage cycling cost:${det['storage_cost']:.4f}")
    print(f"Startup/shutdown:    ${det['su_sd_cost']:.4f}")
    print(f"Balance mismatch sum:{pso_result['balance_mm_mw']:.6e} MW")
    print(f"Reserve violation:   {pso_result['reserve_violation_mw']:.6e} MW")
    print(f"Ramp violation:      {pso_result['ramp_violation_mw']:.6e} MW")
    print(f"SOC end:             {det['soc_end']:.3f} MWh (target {sim.soc0:.3f})")

    soc = det["soc"][0]
    print("\nHour | u(G1 G2 G3) |  Pg(G1 G2 G3)   |  Pch  Pdis | SOC")
    for t in range(sim.T):
        loss = sim.loss(Pg[t, :].reshape(1, -1))[0]
        lhs = Pg[t, :].sum() + Pdis[t] - Pch[t]
        rhs = PLoad[t] + loss
        print(
            f"{t:02d}   | {u[t,0]}  {u[t,1]}  {u[t,2]}   | "
            f"{Pg[t,0]:7.2f} {Pg[t,1]:7.2f} {Pg[t,2]:7.2f} | "
            f"{Pch[t]:5.2f} {Pdis[t]:5.2f} | {soc[t]:7.2f}   (bal={lhs-rhs:+.3e})"
        )

    print(f"\n运行模式: {args.mode}  (use_losses={sim.use_losses}, use_valve_point={sim.use_valve_point})")

    scenario_dir = "aligned" if aligned_mode else "realistic"
    out_dir = os.path.join(args.out_root, scenario_dir)
    dispatch_dir = os.path.join(out_dir, "dispatch")
    compare_dir = os.path.join(out_dir, "compare")
    os.makedirs(dispatch_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    method_results = []
    if args.milp_json:
        method_results.append(load_unified_result_json(args.milp_json))
    if args.dl_json:
        method_results.append(load_unified_result_json(args.dl_json))
    if args.rl_json:
        method_results.append(load_unified_result_json(args.rl_json))
    else:
        method_results.append(pso_result)

    plot_dispatch_figures(sim, pso_result, hist, dispatch_dir)
    comparison_rows = generate_comparison_artifacts(method_results, compare_dir)
    report_path = export_consistency_report(method_results, compare_dir)

    print(f"[OK] Dispatch figures saved: {os.path.abspath(dispatch_dir)}")
    print(f"[OK] Comparison table/chart saved: {os.path.abspath(compare_dir)}")
    print(f"[OK] Consistency report: {report_path}")
    print(f"Methods in comparison: {[r['method'] for r in method_results]}")
    print(f"Comparison rows: {len(comparison_rows)}")


if __name__ == "__main__":
    main()
