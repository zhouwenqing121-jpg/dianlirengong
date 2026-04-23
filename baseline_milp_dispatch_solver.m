%% ================== 1. 初始化与参数设置 ==================
clear; clc; close all;
warning('off','all'); % 关闭不必要的警告

% 调度周期
T = 24;

%% ================== 2. 数据录入 (扁平化防止格式错误) ==================
% --- 风电预测功率 (MW) ---
P_WY = 0.1 * [356.40, 334.63, 343.96, 337.74, 322.18, 315.96, 300.41, 284.85, ...
              281.74, 263.08, 225.75, 138.64, 154.19, 123.09, 116.86, 175.97, ...
              228.86, 253.74, 309.74, 294.19, 303.52, 319.07, 331.52, 325.30];

% --- 系统总电负荷 (MW) ---
PLoad = 0.2 * [261.15, 253.70, 248.52, 245.63, 243.35, 247.00, 256.74, 275.30, ...
               304.50, 333.55, 330.50, 326.70, 322.30, 314.84, 312.56, 313.32, ...
               319.25, 338.72, 338.83, 310.43, 301.45, 290.20, 286.55, 277.58];

% --- 数据完整性检查 (DEBUG) ---
if any(isnan(P_WY)), error('错误: P_WY 数据中包含 NaN'); end
if any(isnan(PLoad)), error('错误: PLoad 数据中包含 NaN'); end

% --- 火电机组参数 ---
a = [0.00003, 0.000014];
b = [0.165,   0.166];
c = [3.5,     7];
P_GMAX = [30, 40];
P_GMIN = [5,  5];
R_U = [4, 6];
R_D = [4, 6];
coal_price = 800;

%% ================== 3. 电网拓扑与B矩阵构建 ==================
baseMVA = 100; 
nbus = 30;

% IEEE 30节点基准负荷 (用于分配负荷)
BusLoadBase = [0; 21.7; 2.4; 7.6; 94.2; 0; 22.8; 30; 0; 5.8; 
               0; 11.2; 0; 6.2; 8.2; 3.5; 9; 3.2; 9.5; 2.2; 
               17.5; 0; 3.2; 8.7; 0; 3.5; 0; 0; 2.4; 10.6];
TotalBaseLoad = sum(BusLoadBase);

% 防止分母为0
if TotalBaseLoad == 0, error('TotalBaseLoad 不能为0'); end

% 计算各节点实际负荷
LoadRatio = BusLoadBase / TotalBaseLoad;
P_Load_Node = repmat(LoadRatio, 1, T) .* repmat(PLoad, nbus, 1);

% 检查负荷矩阵是否有NaN
if any(any(isnan(P_Load_Node))), error('计算出的 P_Load_Node 包含 NaN，请检查负荷数据'); end

% IEEE 30节点支路数据 [首节点, 末节点, 电抗X]
BranchData = [
    1,2,0.0575; 1,3,0.1852; 2,4,0.1737; 3,4,0.0379; 2,5,0.1983; 
    2,6,0.1763; 4,6,0.0414; 5,7,0.1160; 6,7,0.0820; 6,8,0.0420; 
    6,9,0.2080; 6,10,0.5560; 9,11,0.2080; 9,10,0.1100; 4,12,0.2560; 
    12,13,0.1400; 12,14,0.2559; 12,15,0.1304; 12,16,0.1987; 14,15,0.1997; 
    16,17,0.1923; 15,18,0.2185; 18,19,0.1292; 19,20,0.0680; 10,20,0.2090; 
    10,17,0.0845; 10,21,0.0749; 10,22,0.1499; 21,22,0.0236; 15,23,0.2020; 
    22,24,0.1790; 23,24,0.2700; 24,25,0.3292; 25,26,0.3800; 25,27,0.2087; 
    28,27,0.3960; 27,29,0.4153; 27,30,0.6027; 29,30,0.4533; 8,28,0.2000; 
    6,28,0.0599
];

f = BranchData(:,1);
o = BranchData(:,2);
x = BranchData(:,3);
nbranch = size(BranchData, 1);

% 检查电抗是否为0 (防止除以0产生Inf/NaN)
if any(x == 0), error('错误: 存在电抗值为0的支路，会导致计算Inf'); end

% 构建导纳矩阵 Bbus
Bbus = zeros(nbus, nbus);
for k = 1:nbranch
    if isnan(x(k)), error(['第 ' num2str(k) ' 条支路电抗为 NaN']); end
    
    val = 1.0 / x(k); % 导纳
    i = f(k);
    j = o(k);
    
    Bbus(i, j) = Bbus(i, j) - val;
    Bbus(j, i) = Bbus(j, i) - val;
    Bbus(i, i) = Bbus(i, i) + val;
    Bbus(j, j) = Bbus(j, j) + val;
end

% 检查Bbus是否有NaN
if any(any(isnan(Bbus))), error('Bbus 矩阵计算结果包含 NaN'); end

%% ================== 4. 优化模型构建 ==================
% 决策变量
P_G = sdpvar(2, T, 'full');        
P_net_1 = sdpvar(1, T, 'full');    
P_net_2 = sdpvar(1, T, 'full');    
thetaone = sdpvar(30, T, 'full');  

% 辅助变量
cost_c_COAL = sdpvar(2, T, 'full'); 
cost_c_rs = sdpvar(2, T, 'full');   

C = [];

% --- 1. 火电机组约束 ---
for t = 1:T
    for i = 1:2
        C = [C, P_GMIN(i) <= P_G(i,t) <= P_GMAX(i)];
        % 成本计算
        C = [C, cost_c_COAL(i,t) == a(i)*P_G(i,t)^2 + b(i)*P_G(i,t) + c(i)];
        C = [C, cost_c_rs(i,t) == cost_c_COAL(i,t) * coal_price];
    end
    
    % 爬坡
    if t < T
        for i = 1:2
            C = [C, P_G(i,t+1) - P_G(i,t) <= R_U(i)];
            C = [C, P_G(i,t) - P_G(i,t+1) <= R_D(i)];
        end
    end
end

% --- 2. 风电约束 ---
for t = 1:T
    C = [C, 0 <= P_net_1(t) <= P_WY(t)];
    C = [C, 0 <= P_net_2(t) <= P_WY(t)/2];
end

% --- 3. 潮流约束 (修正NaN问题的核心点) ---
for t = 1:T
    % 节点注入功率：先从负荷开始 (Double类型)
    P_inj_load = -P_Load_Node(:, t); 
    
    % 构造发电机注入向量 (Sdpvar类型)
    % 使用 Sparse 或直接索引避免混淆
    P_inj_gen = zeros(nbus, 1); % Double zero
    % 这一步必须小心，不能让 double + sdpvar 时混入 NaN
    
    % G1 (Bus 1), G2 (Bus 5), W1 (Bus 8), W2 (Bus 11)
    % 建立稀疏关联
    % P_inj = P_gen + P_wind - P_load
    
    % 等式右边：直流潮流 B * theta (Sdpvar)
    DC_Flow = Bbus * thetaone(:,t) * baseMVA;
    
    % 逐个节点列写平衡方程，避免构建大数组时的 NaN 风险
    % 节点1: G1 - Load
    C = [C, P_G(1,t) + P_inj_load(1) == DC_Flow(1)];
    
    % 节点5: G2 - Load
    C = [C, P_G(2,t) + P_inj_load(5) == DC_Flow(5)];
    
    % 节点8: Wind1 - Load
    C = [C, P_net_1(t) + P_inj_load(8) == DC_Flow(8)];
    
    % 节点11: Wind2 - Load
    C = [C, P_net_2(t) + P_inj_load(11) == DC_Flow(11)];
    
    % 其他节点: - Load
    other_nodes = setdiff(1:30, [1, 5, 8, 11]);
    C = [C, P_inj_load(other_nodes) == DC_Flow(other_nodes)];
    
    % 支路潮流约束
    theta_diff = thetaone(f, t) - thetaone(o, t);
    C = [C, -600 <= (theta_diff ./ x * baseMVA) <= 600];
end

% 参考节点相角
C = [C, thetaone(1,:) == 0];
C = [C, -pi <= thetaone <= pi];

%% ================== 5. 目标与求解 ==================
Cost_Coal_Total = sum(sum(cost_c_rs));
Cost_Wind_OM = sum(18 * (P_net_1 + P_net_2));
Cost_Curtail = sum(300 * (P_WY - P_net_1)) + sum(300 * (P_WY/2 - P_net_2));

Objective = Cost_Coal_Total + Cost_Wind_OM + Cost_Curtail;

% 设置求解器
ops = sdpsettings('solver', 'gurobi', 'verbose', 1, 'gurobi.MIPGap', 1e-4);
% 如果报错没有gurobi，可以尝试:
% ops = sdpsettings('solver', 'cplex'); 

sol = optimize(C, Objective, ops);

if sol.problem == 0
    disp('✅ 求解成功！');
    fprintf('总成本: %.2f 元\n', value(Objective));
else
    disp('❌ 求解失败');
    disp(sol.info);
    % 如果显示 Infeasible，说明约束冲突
end
%% ================== 7. 结果可视化 ==================
if sol.problem == 0
    % --- 1. 数据提取 (将YALMIP变量转换为数值) ---
    % 功率数据
    Val_PG1 = value(P_G(1,:));
    Val_PG2 = value(P_G(2,:));
    Val_PW1 = value(P_net_1);
    Val_PW2 = value(P_net_2);
    Val_TotalWind_Gen = Val_PW1 + Val_PW2;      % 实际风电出力
    Val_TotalWind_Avail = P_WY + P_WY/2;        % 风电可用上限 (预测值)
    
    % 成本数据
    Val_Cost_Coal = value(Cost_Coal_Total);
    Val_Cost_OM   = value(Cost_Wind_OM);
    Val_Cost_Curtail = value(Cost_Curtail);
    Val_Cost_Total = value(Objective);

    %% --- 图表 1: 各部分成本对比图 ---
    figure('Name', '成本分析', 'Color', 'w', 'Position', [100, 100, 900, 400]);
    
    % 子图1: 饼图 (占比)
    subplot(1, 2, 1);
    cost_data = [Val_Cost_Coal, Val_Cost_OM, Val_Cost_Curtail];
    % 避免负值或极小值影响绘图
    cost_data = max(cost_data, 0); 
    
    explode = [0, 0, 0.1]; % 突出显示弃风成本
    p = pie(cost_data, explode);
    
    % 设置饼图颜色和标签
    colormap(gca, [0.8500 0.3250 0.0980;  % 火电-橙红
                   0.4660 0.6740 0.1880;  % 运维-绿
                   0.6350 0.0780 0.1840]); % 弃风-深红
                   
    legend({'火电燃煤成本', '风电运维成本', '弃风惩罚成本'}, 'Location', 'southoutside');
    title(sprintf('总成本构成 (总计: %.2f 万元)', Val_Cost_Total/10000), 'FontSize', 12, 'FontWeight', 'bold');
    
    % 子图2: 柱状图 (具体数值)
    subplot(1, 2, 2);
    b = bar([1, 2, 3], cost_data, 'FaceColor', 'flat');
    b.CData(1,:) = [0.8500 0.3250 0.0980];
    b.CData(2,:) = [0.4660 0.6740 0.1880];
    b.CData(3,:) = [0.6350 0.0780 0.1840];
    
    set(gca, 'XTickLabel', {'燃煤成本', '运维成本', '弃风惩罚'});
    ylabel('金额 (元)');
    grid on;
    % 在柱子上显示数值
    xtips = b.XEndPoints;
    ytips = b.YEndPoints;
    labels = string(round(b.YData, 0));
    text(xtips, ytips, labels, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    title('各分项成本具体数值', 'FontSize', 12);

    %% --- 图表 2: 电功率平衡图 (堆积图) ---
    figure('Name', '功率平衡', 'Color', 'w', 'Position', [100, 550, 1000, 500]);
    
    t_axis = 1:T;
    % 准备堆积数据: 行对应时间，列对应[G1, G2, W1, W2]
    stack_data = [Val_PG1', Val_PG2', Val_PW1', Val_PW2'];
    
    % 绘制堆积柱状图
    bar_handle = bar(t_axis, stack_data, 'stacked', 'BarWidth', 0.6);
    
    % 设置颜色
    bar_handle(1).FaceColor = [0.8500 0.3250 0.0980]; % G1 - 橙红
    bar_handle(2).FaceColor = [0.9290 0.6940 0.1250]; % G2 - 黄色
    bar_handle(3).FaceColor = [0.4660 0.6740 0.1880]; % W1 - 绿色
    bar_handle(4).FaceColor = [0.3010 0.7450 0.9330]; % W2 - 浅蓝
    
    hold on;
    
    % 绘制负荷曲线
    plot(t_axis, PLoad, 'k-o', 'LineWidth', 2, 'MarkerFaceColor', 'k', 'DisplayName', '系统总负荷');
    
    % 绘制风电可用上限 (展示弃风情况)
    % 将 G1+G2 的基底加上 总风电预测，画出虚线，表示"如果风电全发，总发电曲线会在哪里"
    % 这是一种高级展示方法，可以直观看到柱状图顶部和虚线的差距即为弃风
    Total_Gen_Capacity = Val_PG1 + Val_PG2 + Val_TotalWind_Avail;
    plot(t_axis, Total_Gen_Capacity, 'r--', 'LineWidth', 1.5, 'DisplayName', '最大可能发电出力 (含弃风)');
    
    % 图表装饰
    xlabel('时间 (h)', 'FontSize', 12);
    ylabel('功率 (MW)', 'FontSize', 12);
    title('系统电功率平衡与机组出力分配', 'FontSize', 14, 'FontWeight', 'bold');
    xlim([0.5, 24.5]);
    ylim([0, max(Total_Gen_Capacity)*1.1]); % 留一点顶部空间
    
    legend({'火电机组 G1', '火电机组 G2', '风电场 W1', '风电场 W2', '系统总负荷', '潜在最大出力'}, ...
           'Location', 'northoutside', 'Orientation', 'horizontal');
    grid on;
    box on;
    
    fprintf('绘图完成。\n');
    
else
    disp('由于求解失败，无法生成图表。');
end
