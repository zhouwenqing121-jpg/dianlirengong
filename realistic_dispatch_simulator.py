"""
粒子群算法(PSO)求解IEEE 30节点电力系统日前调度
数据与MATLAB MILP版本完全一致，仅优化算法不同。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ================== 1. 初始化与参数设置 ==================
np.random.seed(42)
T = 24  # 调度周期（小时）

# ================== 2. 数据录入 ==================
# --- 风电预测功率 (MW) ---
P_WY = 0.1 * np.array([356.40, 334.63, 343.96, 337.74, 322.18, 315.96, 300.41, 284.85,
                        281.74, 263.08, 225.75, 138.64, 154.19, 123.09, 116.86, 175.97,
                        228.86, 253.74, 309.74, 294.19, 303.52, 319.07, 331.52, 325.30])

# --- 系统总电负荷 (MW) ---
PLoad = 0.2 * np.array([261.15, 253.70, 248.52, 245.63, 243.35, 247.00, 256.74, 275.30,
                         304.50, 333.55, 330.50, 326.70, 322.30, 314.84, 312.56, 313.32,
                         319.25, 338.72, 338.83, 310.43, 301.45, 290.20, 286.55, 277.58])

# --- 火电机组参数 ---
a       = np.array([0.00003,  0.000014])   # 二次系数
b_coef  = np.array([0.165,    0.166])      # 一次系数
c_coef  = np.array([3.5,      7.0])        # 常数项
P_GMAX  = np.array([30.0,     40.0])       # 最大出力 (MW)
P_GMIN  = np.array([5.0,       5.0])       # 最小出力 (MW)
R_U     = np.array([4.0,       6.0])       # 爬坡上限 (MW/h)
R_D     = np.array([4.0,       6.0])       # 爬坡下限 (MW/h)
coal_price = 800.0                         # 煤价 (元/MWh)

# ================== 3. IEEE 30节点拓扑与Bbus矩阵 ==================
baseMVA = 100
nbus = 30

# 基准负荷分布（用于按比例分配各节点负荷）
BusLoadBase = np.array([0, 21.7, 2.4, 7.6, 94.2, 0, 22.8, 30, 0, 5.8,
                         0, 11.2, 0, 6.2, 8.2, 3.5, 9, 3.2, 9.5, 2.2,
                         17.5, 0, 3.2, 8.7, 0, 3.5, 0, 0, 2.4, 10.6])
TotalBaseLoad = BusLoadBase.sum()
LoadRatio = BusLoadBase / TotalBaseLoad

# P_Load_Node[bus, t]: 各节点各时刻实际负荷 (MW)
P_Load_Node = np.outer(LoadRatio, PLoad)  # (30, 24)

# IEEE 30节点支路数据 [首节点, 末节点, 电抗X]  (节点编号从1开始)
BranchData = np.array([
    [1,  2,  0.0575], [1,  3,  0.1852], [2,  4,  0.1737], [3,  4,  0.0379],
    [2,  5,  0.1983], [2,  6,  0.1763], [4,  6,  0.0414], [5,  7,  0.1160],
    [6,  7,  0.0820], [6,  8,  0.0420], [6,  9,  0.2080], [6,  10, 0.5560],
    [9,  11, 0.2080], [9,  10, 0.1100], [4,  12, 0.2560], [12, 13, 0.1400],
    [12, 14, 0.2559], [12, 15, 0.1304], [12, 16, 0.1987], [14, 15, 0.1997],
    [16, 17, 0.1923], [15, 18, 0.2185], [18, 19, 0.1292], [19, 20, 0.0680],
    [10, 20, 0.2090], [10, 17, 0.0845], [10, 21, 0.0749], [10, 22, 0.1499],
    [21, 22, 0.0236], [15, 23, 0.2020], [22, 24, 0.1790], [23, 24, 0.2700],
    [24, 25, 0.3292], [25, 26, 0.3800], [25, 27, 0.2087], [28, 27, 0.3960],
    [27, 29, 0.4153], [27, 30, 0.6027], [29, 30, 0.4533], [8,  28, 0.2000],
    [6,  28, 0.0599],
])

f_bus   = (BranchData[:, 0] - 1).astype(int)  # 0-indexed首节点
t_bus   = (BranchData[:, 1] - 1).astype(int)  # 0-indexed末节点
x_line  = BranchData[:, 2]
nbranch = len(BranchData)

# 构建节点导纳矩阵 Bbus
Bbus = np.zeros((nbus, nbus))
for k in range(nbranch):
    val = 1.0 / x_line[k]
    i, j = f_bus[k], t_bus[k]
    Bbus[i, j] -= val
    Bbus[j, i] -= val
    Bbus[i, i] += val
    Bbus[j, j] += val

# 去掉参考节点(bus 0)后的简约B矩阵，预先求逆以加速DC潮流
B_red     = Bbus[1:, 1:]             # (29, 29)
B_red_inv = np.linalg.inv(B_red)     # 预计算，避免循环内反复求逆

# ================== 4. DC潮流辅助函数 ==================

def solve_dc_pf(P_inj):
    """
    直流潮流求解。
    P_inj: (30,) 各节点净注入功率 (MW), bus 0 为参考节点。
    返回 theta: (30,) 相角 (rad), theta[0] = 0。
    """
    P_inj_red   = P_inj[1:] / baseMVA          # 标幺化，去掉参考节点
    theta_red   = B_red_inv @ P_inj_red
    theta       = np.concatenate([[0.0], theta_red])
    return theta


def compute_branch_flows(theta):
    """由相角计算各支路潮流 (MW)。"""
    return (theta[f_bus] - theta[t_bus]) / x_line * baseMVA


# ================== 5. 适应度函数（目标 + 惩罚） ==================
PENALTY_BALANCE  = 1e8   # 功率不平衡惩罚系数
PENALTY_RAMP     = 1e7   # 爬坡越限惩罚系数
PENALTY_FLOW     = 1e5   # 支路越限惩罚系数
FLOW_LIMIT       = 600.0 # 支路潮流上限 (MW)

# 预计算各时刻的上下界向量（避免循环内重复创建）
_lb = np.empty(4 * T)
_ub = np.empty(4 * T)
for _t in range(T):
    _lb[_t]          = P_GMIN[0]
    _ub[_t]          = P_GMAX[0]
    _lb[T   + _t]    = P_GMIN[1]
    _ub[T   + _t]    = P_GMAX[1]
    _lb[2*T + _t]    = 0.0
    _ub[2*T + _t]    = P_WY[_t]
    _lb[3*T + _t]    = 0.0
    _ub[3*T + _t]    = P_WY[_t] / 2.0


def fitness(x):
    """
    计算粒子的适应度（目标值 + 惩罚项）。
    x 布局: [P_G1(0..T-1), P_G2(0..T-1), P_net1(0..T-1), P_net2(0..T-1)]
    共 4*T = 96 个变量。
    """
    P_G1   = x[0:T]
    P_G2   = x[T:2*T]
    P_net1 = x[2*T:3*T]
    P_net2 = x[3*T:4*T]

    # ---- 目标成本 ----
    # 火电燃煤成本
    coal1 = (a[0] * P_G1**2 + b_coef[0] * P_G1 + c_coef[0]) * coal_price
    coal2 = (a[1] * P_G2**2 + b_coef[1] * P_G2 + c_coef[1]) * coal_price
    Cost_Coal = coal1.sum() + coal2.sum()

    # 风电运维成本
    Cost_OM = 18.0 * (P_net1 + P_net2).sum()

    # 弃风惩罚成本
    Cost_Curtail = (300.0 * (P_WY - P_net1)).sum() + (300.0 * (P_WY / 2.0 - P_net2)).sum()

    obj = Cost_Coal + Cost_OM + Cost_Curtail

    # ---- 惩罚：爬坡约束 ----
    ramp_up1   = np.maximum(0.0, np.diff(P_G1) - R_U[0])
    ramp_down1 = np.maximum(0.0, -np.diff(P_G1) - R_D[0])
    ramp_up2   = np.maximum(0.0, np.diff(P_G2) - R_U[1])
    ramp_down2 = np.maximum(0.0, -np.diff(P_G2) - R_D[1])
    pen_ramp = PENALTY_RAMP * (
        (ramp_up1**2).sum() + (ramp_down1**2).sum() +
        (ramp_up2**2).sum() + (ramp_down2**2).sum()
    )

    # ---- 惩罚：节点功率平衡 & 支路越限 ----
    pen_balance = 0.0
    pen_flow    = 0.0

    for t in range(T):
        # 计算各节点净注入 (MW)
        P_inj = -P_Load_Node[:, t].copy()
        P_inj[0]  += P_G1[t]    # G1 在 bus 1 (index 0)
        P_inj[4]  += P_G2[t]    # G2 在 bus 5 (index 4)
        P_inj[7]  += P_net1[t]  # W1 在 bus 8 (index 7)
        P_inj[10] += P_net2[t]  # W2 在 bus 11 (index 10)

        # 全网功率不平衡惩罚
        bal_err = P_inj.sum()
        pen_balance += PENALTY_BALANCE * bal_err**2

        # DC潮流与支路越限惩罚
        theta  = solve_dc_pf(P_inj)
        flows  = compute_branch_flows(theta)
        viol   = np.maximum(0.0, np.abs(flows) - FLOW_LIMIT)
        pen_flow += PENALTY_FLOW * (viol**2).sum()

    return obj + pen_ramp + pen_balance + pen_flow


# ================== 6. 粒子群优化 (PSO) ==================

def pso_dispatch(n_particles=80, max_iter=400, w=0.7298, c1=1.4962, c2=1.4962):
    """
    标准PSO求解调度问题。
    参数:
        n_particles : 种群规模
        max_iter    : 最大迭代次数
        w           : 惯性权重
        c1          : 个体学习因子
        c2          : 社会学习因子
    返回:
        gbest_pos   : 全局最优位置
        gbest_val   : 全局最优适应度
        history     : 每代最优适应度记录
    """
    n_var = 4 * T

    # --- 初始化位置与速度 ---
    pos = _lb + np.random.rand(n_particles, n_var) * (_ub - _lb)
    vel_range = 0.1 * (_ub - _lb)
    vel = -vel_range + 2.0 * np.random.rand(n_particles, n_var) * vel_range

    # --- 初始评估 ---
    pbest_pos = pos.copy()
    pbest_val = np.array([fitness(pos[i]) for i in range(n_particles)])

    gbest_idx = int(np.argmin(pbest_val))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = float(pbest_val[gbest_idx])

    history = [gbest_val]

    print(f"  [PSO] 初始最优适应度: {gbest_val:.4e}")

    # --- 主迭代 ---
    for it in range(max_iter):
        r1 = np.random.rand(n_particles, n_var)
        r2 = np.random.rand(n_particles, n_var)

        # 速度与位置更新
        vel = (w * vel
               + c1 * r1 * (pbest_pos - pos)
               + c2 * r2 * (gbest_pos  - pos))

        pos = pos + vel

        # 越界修复：截断到可行域
        pos = np.clip(pos, _lb, _ub)

        # 评估并更新个体/全局最优
        for i in range(n_particles):
            val = fitness(pos[i])
            if val < pbest_val[i]:
                pbest_val[i] = val
                pbest_pos[i] = pos[i].copy()

        best_i = int(np.argmin(pbest_val))
        if pbest_val[best_i] < gbest_val:
            gbest_val = float(pbest_val[best_i])
            gbest_pos = pbest_pos[best_i].copy()

        history.append(gbest_val)

        if (it + 1) % 50 == 0:
            print(f"  [PSO] 迭代 {it+1:4d}/{max_iter}, 当前最优: {gbest_val:.4e}")

    return gbest_pos, gbest_val, history


# ================== 7. 运行PSO求解 ==================
print("=" * 55)
print("  粒子群算法 (PSO) — IEEE 30节点日前调度优化")
print("=" * 55)

best_x, best_fitness, cost_history = pso_dispatch(
    n_particles=80,
    max_iter=400,
)

# 提取最优解
Val_PG1   = best_x[0:T]
Val_PG2   = best_x[T:2*T]
Val_PW1   = best_x[2*T:3*T]
Val_PW2   = best_x[3*T:4*T]

# 分项成本
Cost_Coal_Total  = (
    (a[0] * Val_PG1**2 + b_coef[0] * Val_PG1 + c_coef[0]).sum() * coal_price
    + (a[1] * Val_PG2**2 + b_coef[1] * Val_PG2 + c_coef[1]).sum() * coal_price
)
Cost_Wind_OM     = 18.0 * (Val_PW1 + Val_PW2).sum()
Cost_Curtail_Val = (
    (300.0 * (P_WY - Val_PW1)).sum()
    + (300.0 * (P_WY / 2.0 - Val_PW2)).sum()
)
Cost_Total = Cost_Coal_Total + Cost_Wind_OM + Cost_Curtail_Val

print("\n✅ PSO 求解完成！")
print(f"   火电燃煤成本  : {Cost_Coal_Total:>14.2f} 元")
print(f"   风电运维成本  : {Cost_Wind_OM:>14.2f} 元")
print(f"   弃风惩罚成本  : {Cost_Curtail_Val:>14.2f} 元")
print(f"   总 成 本      : {Cost_Total:>14.2f} 元  ({Cost_Total/10000:.2f} 万元)")
print("=" * 55)

# ================== 8. 结果可视化 ==================

Val_TotalWind_Gen   = Val_PW1 + Val_PW2
Val_TotalWind_Avail = P_WY + P_WY / 2.0
t_axis = np.arange(1, T + 1)

# -------- 图1：成本分析 --------
fig1, axes1 = plt.subplots(1, 2, figsize=(11, 4.5))
fig1.patch.set_facecolor('white')
fig1.suptitle('Cost Analysis  (PSO)', fontsize=14, fontweight='bold')

cost_data   = np.maximum([Cost_Coal_Total, Cost_Wind_OM, Cost_Curtail_Val], 0)
labels_cost = ['Coal Cost', 'Wind O&M', 'Curtailment']
colors3     = ['#D95319', '#77AC30', '#A2142F']

# 饼图
ax = axes1[0]
wedges, texts, autotexts = ax.pie(
    cost_data, labels=labels_cost, colors=colors3,
    autopct='%1.1f%%', startangle=90,
    explode=[0, 0, 0.08], pctdistance=0.8,
)
ax.set_title(f'Cost Breakdown\n(Total: {Cost_Total/10000:.2f} wan yuan)', fontsize=11)

# 柱状图
ax = axes1[1]
bars = ax.bar(labels_cost, cost_data, color=colors3, width=0.5)
ax.set_ylabel('Cost (yuan)')
ax.grid(axis='y', linestyle='--', alpha=0.6)
for bar, val in zip(bars, cost_data):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
            f'{val:.0f}', ha='center', va='bottom', fontsize=9)
ax.set_title('Cost Items', fontsize=11)

fig1.tight_layout()
fig1.savefig('cost_analysis_pso.png', dpi=150, bbox_inches='tight')
print("  已保存: cost_analysis_pso.png")

# -------- 图2：电功率平衡堆积图 --------
fig2, ax2 = plt.subplots(figsize=(12, 5.5))
fig2.patch.set_facecolor('white')

stack_data = np.column_stack([Val_PG1, Val_PG2, Val_PW1, Val_PW2])
colors4    = ['#D95319', '#EDB120', '#77AC30', '#4DBEEE']
labels4    = ['Thermal G1', 'Thermal G2', 'Wind W1', 'Wind W2']

bottom = np.zeros(T)
for k in range(4):
    ax2.bar(t_axis, stack_data[:, k], bottom=bottom,
            color=colors4[k], label=labels4[k], width=0.65, alpha=0.9)
    bottom += stack_data[:, k]

# 负荷曲线
ax2.plot(t_axis, PLoad, 'k-o', linewidth=2, markersize=4,
         label='System Load', zorder=5)

# 最大可能出力（含弃风）
Total_Gen_Cap = Val_PG1 + Val_PG2 + Val_TotalWind_Avail
ax2.plot(t_axis, Total_Gen_Cap, 'r--', linewidth=1.5,
         label='Max Available (incl. curtailment)', zorder=5)

ax2.set_xlabel('Time (h)', fontsize=12)
ax2.set_ylabel('Power (MW)', fontsize=12)
ax2.set_title('System Power Balance & Unit Dispatch  (PSO)', fontsize=13, fontweight='bold')
ax2.set_xlim(0.5, T + 0.5)
ax2.set_ylim(0, max(Total_Gen_Cap) * 1.15)
ax2.legend(loc='upper center', ncol=6, fontsize=9,
           bbox_to_anchor=(0.5, 1.0), framealpha=0.85)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

fig2.tight_layout()
fig2.savefig('power_balance_pso.png', dpi=150, bbox_inches='tight')
print("  已保存: power_balance_pso.png")

# -------- 图3：PSO收敛曲线 --------
fig3, ax3 = plt.subplots(figsize=(8, 4))
fig3.patch.set_facecolor('white')

ax3.plot(np.arange(len(cost_history)), cost_history,
         color='#0072BD', linewidth=1.5)
ax3.set_xlabel('Iteration', fontsize=12)
ax3.set_ylabel('Best Fitness (yuan)', fontsize=12)
ax3.set_title('PSO Convergence Curve', fontsize=13, fontweight='bold')
ax3.grid(linestyle='--', alpha=0.6)

fig3.tight_layout()
fig3.savefig('convergence_pso.png', dpi=150, bbox_inches='tight')
print("  已保存: convergence_pso.png")

plt.close('all')
print("  绘图完成。")
