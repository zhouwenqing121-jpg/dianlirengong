import os
import numpy as np
import matplotlib

# Headless-safe backend (CI/servers)
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def plot_ppt_figures(sim, PLoad, u, Pg, Pch, Pdis, det, hist, out_dir="ppt_figures"):
    os.makedirs(out_dir, exist_ok=True)

    T = sim.T
    G = sim.G
    hours = np.arange(T)

    soc = det["soc"][0]

    losses = np.array([sim.loss(Pg[t, :].reshape(1, -1))[0] for t in range(T)])
    lhs = Pg.sum(axis=1) + Pdis - Pch
    rhs = PLoad + losses
    bal_res = lhs - rhs

    rgen = np.array([sim.reserve_gen(Pg[t, :].reshape(1, -1))[0] for t in range(T)])
    rst = np.array([sim.reserve_storage(np.array([Pdis[t]]), np.array([soc[t]]))[0] for t in range(T)])
    rtot = rgen + rst

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 2.2,
        }
    )

    fig = plt.figure(figsize=(9, 4.5))
    plt.plot(hours, PLoad, marker="o")
    plt.title("24h Load Profile")
    plt.xlabel("Hour")
    plt.ylabel("Load (MW)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_load.png"))
    plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    labels = [f"G{i + 1}" for i in range(G)]
    plt.stackplot(hours, Pg.T, labels=labels, alpha=0.85)
    net_storage = Pdis - Pch
    plt.plot(hours, net_storage, color="k", linestyle="--", label="Storage net (Pdis-Pch)")
    plt.plot(hours, PLoad, color="red", marker=".", label="Load")
    plt.title("Dispatch Stack (Generators) + Storage Net + Load")
    plt.xlabel("Hour")
    plt.ylabel("Power (MW)")
    plt.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_dispatch_stack.png"))
    plt.close(fig)

    fig = plt.figure(figsize=(10, 4.5))
    plt.bar(hours - 0.15, Pch, width=0.3, label="Charge (Pch)", color="#1f77b4")
    plt.bar(hours + 0.15, Pdis, width=0.3, label="Discharge (Pdis)", color="#ff7f0e")
    plt.title("Storage Power (Charge/Discharge)")
    plt.xlabel("Hour")
    plt.ylabel("Power (MW)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_storage_power.png"))
    plt.close(fig)

    fig = plt.figure(figsize=(10, 4.5))
    plt.plot(np.arange(T + 1), soc, marker="o")
    plt.axhline(sim.soc0, color="k", linestyle="--", alpha=0.7, label="SOC target (SOC0)")
    plt.title("Storage State of Charge (SOC)")
    plt.xlabel("Hour")
    plt.ylabel("SOC (MWh)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_soc.png"))
    plt.close(fig)

    fig = plt.figure(figsize=(10, 4.5))
    plt.plot(hours, rtot, marker="o", label="Reachable reserve (Gen + Storage)")
    plt.axhline(sim.R, color="red", linestyle="--", label=f"Reserve requirement R={sim.R:.0f} MW")
    plt.fill_between(hours, rtot, sim.R, where=(rtot < sim.R), color="red", alpha=0.15, label="Violation")
    plt.title("Reachable Spinning Reserve")
    plt.xlabel("Hour")
    plt.ylabel("Reserve (MW)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_reserve.png"))
    plt.close(fig)

    fig = plt.figure(figsize=(10, 4.5))
    plt.axhline(0.0, color="k", linewidth=1.5)
    plt.plot(hours, bal_res, marker="o")
    plt.title("Power Balance Residual: (ΣPg + Pdis - Pch) - (Load + Loss)")
    plt.xlabel("Hour")
    plt.ylabel("Residual (MW)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_balance_residual.png"))
    plt.close(fig)

    fig = plt.figure(figsize=(9, 4.5))
    total_cost_hist = np.array(hist["gen_cost"]) + np.array(hist["storage_cost"]) + np.array(hist["su_sd_cost"])
    plt.plot(total_cost_hist)
    plt.title("PSO Convergence: Total Cost (Gen + Storage + SU/SD)")
    plt.xlabel("Iteration")
    plt.ylabel("Cost ($)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_convergence.png"))
    plt.close(fig)

    print(f"[OK] PPT figures saved to: {os.path.abspath(out_dir)}")
    print("Saved files: 01_load.png .. 07_convergence.png")


def main():
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

    sim = RealisticDispatchSimulator(PLoad, R_fixed=80.0, use_losses=True, use_valve_point=True, seed=42)
    (u, Pg, Pch, Pdis), hist = sim.run(num_particles=120, max_iter=200)

    fit, det = sim.evaluate(
        Pg.reshape(1, sim.T, sim.G),
        u.reshape(1, sim.T, sim.G),
        Pch.reshape(1, sim.T),
        Pdis.reshape(1, sim.T),
    )
    det = {k: float(v[0]) if isinstance(v, np.ndarray) and v.ndim == 1 else v for k, v in det.items()}

    print("=" * 95)
    print("真实调度仿真（UC + DED + 储能 + 备用 + 爬坡 + 网损 + 阀点）")
    print("=" * 95)
    print(f"Best fitness:         {float(fit[0]):.4f}")
    print(f"Total gen cost:      ${det['gen_cost']:.4f}")
    print(f"Storage cycling cost:${det['storage_cost']:.4f}")
    print(f"Startup/shutdown:    ${det['su_sd_cost']:.4f}")
    print(f"Balance mismatch sum:{det['balance_mm']:.6e} MW")
    print(f"Reserve violation:   {det['reserve_vio']:.6e} MW")
    print(f"Ramp violation:      {det['ramp_vio']:.6e} MW")
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

    sim_cmp = RealisticDispatchSimulator(PLoad, R_fixed=80.0, use_losses=False, use_valve_point=False, seed=42)
    (u_cmp, Pg_cmp, Pch_cmp, Pdis_cmp), _ = sim_cmp.run(num_particles=120, max_iter=200)
    fit_cmp, det_cmp = sim_cmp.evaluate(
        Pg_cmp.reshape(1, sim_cmp.T, sim_cmp.G),
        u_cmp.reshape(1, sim_cmp.T, sim_cmp.G),
        Pch_cmp.reshape(1, sim_cmp.T),
        Pdis_cmp.reshape(1, sim_cmp.T),
    )
    print("\n差异原因分析：")
    print("1) 真实调度默认启用网损和阀点，MILP基线未启用这两项。")
    print("2) 真实调度用PSO启发式搜索，MILP基线是确定性优化求解。")
    print(f"3) 在同假设（无网损/无阀点）下，本脚本PSO目标值: {float(fit_cmp[0]):.4f}")
    print(
        f"   分解成本: Gen=${float(det_cmp['gen_cost'][0]):.4f}, "
        f"Storage=${float(det_cmp['storage_cost'][0]):.4f}, SU/SD=${float(det_cmp['su_sd_cost'][0]):.4f}"
    )

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=110)
    axs = axs.ravel()
    axs[0].plot(np.array(hist["gen_cost"]) + np.array(hist["storage_cost"]) + np.array(hist["su_sd_cost"]), lw=2)
    axs[0].set_title("Total Cost (Gen + Storage + SU/SD)")
    axs[0].grid(True)
    axs[1].semilogy(np.maximum(hist["balance_mm"], 1e-18), lw=2)
    axs[1].set_title("Balance mismatch (log)")
    axs[1].grid(True)
    axs[2].semilogy(np.maximum(hist["reserve_vio"], 1e-18), lw=2)
    axs[2].set_title("Reserve violation (log)")
    axs[2].grid(True)
    axs[3].plot(hist["soc_end"], lw=2)
    axs[3].set_title("End SOC (MWh)")
    axs[3].grid(True)
    plt.tight_layout()
    fig.savefig("summary.png")
    plt.close(fig)

    plot_ppt_figures(sim, PLoad, u, Pg, Pch, Pdis, det, hist, out_dir="ppt_figures")


if __name__ == "__main__":
    main()
