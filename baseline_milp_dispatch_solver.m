function results = baseline_milp_dispatch_solver(rlResult)
% baseline_milp_dispatch_solver
% 使用普通求解器 intlinprog 做 UC+ED+储能 调度，作为强化学习/启发式算法对照组。
% 说明：该 MILP 基线保留核心工程约束（最小开停机、爬坡、备用、储能 SOC），
%      发电成本用分段线性化二次成本；不包含网损与阀点项（便于使用标准 MILP 求解器）。
%
% 用法：
%   results = baseline_milp_dispatch_solver();
%   rlResult = struct('total_cost',1.23e5,'balance_mm',0,'reserve_vio',0,'ramp_vio',0,'soc_end',150);
%   results = baseline_milp_dispatch_solver(rlResult);

if nargin < 1
    rlResult = [];
end

%% ------------------------ 数据（与现有脚本一致） ------------------------
loadProfile = [ ...
    272.563258729604;264.888219494172;261.799350631702;258.697623468531; ...
    253.748073897436;257.3254;267.583538636364;286.960179242424; ...
    290.062693484849;312.776684848485;309.970979242424;306.384639545455; ...
    302.196933636364;295.091917121212;292.778442727273;293.351041666667; ...
    298.876115909091;317.191548181818;319.602692121212;323.599087121212; ...
    317.339661212121;305.464569545455;298.74788265035;289.784661613054];
T = numel(loadProfile);

% [a,b,c,~,~,Pmin,Pmax,RU,RD,startCost,stopCost,minUp,minDown]
gen = [ ...
    561.0,7.92,0.001562,300.0,0.0315,100.0,600.0,80.0,80.0,2500.0,500.0,3,3; ...
    310.0,7.85,0.001940,200.0,0.0420,100.0,400.0,60.0,60.0,1800.0,400.0,2,2; ...
    78.0,7.97,0.004820,150.0,0.0630,50.0,200.0,40.0,40.0,900.0,250.0,1,1];

G = size(gen,1);
a = gen(:,1); b = gen(:,2); c = gen(:,3);
Pmin = gen(:,6); Pmax = gen(:,7);
RU = gen(:,8); RD = gen(:,9);
startCost = gen(:,10); stopCost = gen(:,11);
minUp = round(gen(:,12)); minDown = round(gen(:,13));

Rreq = 80.0;
PchMax = 100.0;
PdisMax = 100.0;
Emax = 300.0;
E0 = 150.0;
etaCh = 0.95;
etaDis = 0.95;
cycleCost = 1.0;

% 分段线性化配置（越大越精细）
nSeg = 8;

%% ------------------------ 变量索引 ------------------------
% 决策变量（全部打平到 x）：
% Pg(g,t), u(g,t), y(g,t), z(g,t), seg(g,s,t), Rg(g,t), Pch(t), Pdis(t), E(t=1..T+1), Rs(t), smode(t)

nPg = G*T;
nu = G*T;
ny = G*T;
nzv = G*T;
nSegVar = G*nSeg*T;
nRg = G*T;
nPch = T;
nPdis = T;
nE = T+1;
nRs = T;
nSMode = T;

offset = 0;
idx.Pg = reshape(offset + (1:nPg), [G,T]); offset = offset + nPg;
idx.u = reshape(offset + (1:nu), [G,T]); offset = offset + nu;
idx.y = reshape(offset + (1:ny), [G,T]); offset = offset + ny;
idx.z = reshape(offset + (1:nzv), [G,T]); offset = offset + nzv;
idx.seg = reshape(offset + (1:nSegVar), [G,nSeg,T]); offset = offset + nSegVar;
idx.Rg = reshape(offset + (1:nRg), [G,T]); offset = offset + nRg;
idx.Pch = offset + (1:nPch); offset = offset + nPch;
idx.Pdis = offset + (1:nPdis); offset = offset + nPdis;
idx.E = offset + (1:nE); offset = offset + nE;
idx.Rs = offset + (1:nRs); offset = offset + nRs;
idx.smode = offset + (1:nSMode); offset = offset + nSMode;

nVar = offset;

%% ------------------------ 目标函数 ------------------------
f = zeros(nVar,1);

% 分段成本：f(P)=a+bP+cP^2 在 [Pmin,Pmax] 上做增量线性化
segCap = zeros(G,nSeg);
segSlope = zeros(G,nSeg);
baseOnCost = zeros(G,1);

for g = 1:G
    bp = linspace(Pmin(g), Pmax(g), nSeg+1);
    fg = a(g) + b(g)*bp + c(g)*(bp.^2);
    baseOnCost(g) = fg(1);
    for s = 1:nSeg
        segCap(g,s) = bp(s+1)-bp(s);
        if segCap(g,s) <= 0
            error('分段线性化失败：segCap 必须为正，当前 g=%d, s=%d', g, s);
        end
        segSlope(g,s) = (fg(s+1)-fg(s))/segCap(g,s);
    end
end

for t = 1:T
    for g = 1:G
        f(idx.u(g,t)) = f(idx.u(g,t)) + baseOnCost(g);
        f(idx.y(g,t)) = f(idx.y(g,t)) + startCost(g);
        f(idx.z(g,t)) = f(idx.z(g,t)) + stopCost(g);
        for s = 1:nSeg
            f(idx.seg(g,s,t)) = f(idx.seg(g,s,t)) + segSlope(g,s);
        end
    end
    f(idx.Pch(t)) = f(idx.Pch(t)) + cycleCost;
    f(idx.Pdis(t)) = f(idx.Pdis(t)) + cycleCost;
end

%% ------------------------ 约束组装 ------------------------
A = sparse([],[],[],0,nVar,0);
bvec = [];
Aeq = sparse([],[],[],0,nVar,0);
beq = [];

% 1) 功率上下限 + 分段关系
for t = 1:T
    for g = 1:G
        % Pg = Pmin*u + sum(seg)
        row = sparse(1,nVar);
        row(idx.Pg(g,t)) = 1;
        row(idx.u(g,t)) = -Pmin(g);
        row(idx.seg(g,:,t)) = -1;
        Aeq = [Aeq; row]; %#ok<AGROW>
        beq = [beq; 0]; %#ok<AGROW>

        % Pg <= Pmax*u
        row = sparse(1,nVar);
        row(idx.Pg(g,t)) = 1;
        row(idx.u(g,t)) = -Pmax(g);
        A = [A; row]; %#ok<AGROW>
        bvec = [bvec; 0]; %#ok<AGROW>

        % Rg <= RU*u
        row = sparse(1,nVar);
        row(idx.Rg(g,t)) = 1;
        row(idx.u(g,t)) = -RU(g);
        A = [A; row]; %#ok<AGROW>
        bvec = [bvec; 0]; %#ok<AGROW>

        % Rg <= Pmax*u - Pg  => Rg + Pg - Pmax*u <= 0
        row = sparse(1,nVar);
        row(idx.Rg(g,t)) = 1;
        row(idx.Pg(g,t)) = 1;
        row(idx.u(g,t)) = -Pmax(g);
        A = [A; row]; %#ok<AGROW>
        bvec = [bvec; 0]; %#ok<AGROW>

        % seg_s <= cap_s * u
        for s = 1:nSeg
            row = sparse(1,nVar);
            row(idx.seg(g,s,t)) = 1;
            row(idx.u(g,t)) = -segCap(g,s);
            A = [A; row]; %#ok<AGROW>
            bvec = [bvec; 0]; %#ok<AGROW>
        end
    end
end

% 2) 启停逻辑：u(t)-u(t-1)=y(t)-z(t), 假设初始 u(0)=0
for g = 1:G
    for t = 1:T
        row = sparse(1,nVar);
        row(idx.u(g,t)) = 1;
        row(idx.y(g,t)) = -1;
        row(idx.z(g,t)) = 1;
        if t > 1
            row(idx.u(g,t-1)) = -1;
        end
        Aeq = [Aeq; row]; %#ok<AGROW>
        beq = [beq; 0]; %#ok<AGROW>
    end
end

% 3) 最小开机时间
for g = 1:G
    U = minUp(g);
    for t = 1:T
        k0 = max(1, t-U+1);
        row = sparse(1,nVar);
        row(idx.y(g,k0:t)) = 1;
        row(idx.u(g,t)) = -1;
        A = [A; row]; %#ok<AGROW>
        bvec = [bvec; 0]; %#ok<AGROW>
    end
end

% 4) 最小停机时间
for g = 1:G
    D = minDown(g);
    for t = 1:T
        k0 = max(1, t-D+1);
        row = sparse(1,nVar);
        row(idx.z(g,k0:t)) = 1;
        row(idx.u(g,t)) = 1;
        A = [A; row]; %#ok<AGROW>
        bvec = [bvec; 1]; %#ok<AGROW>
    end
end

% 5) 爬坡约束（状态切换时用 big-M 放宽）
for g = 1:G
    M = Pmax(g);
    for t = 2:T
        % Pg_t - Pg_{t-1} <= RU + M*(1-u_{t-1})
        row = sparse(1,nVar);
        row(idx.Pg(g,t)) = 1;
        row(idx.Pg(g,t-1)) = -1;
        row(idx.u(g,t-1)) = M;
        A = [A; row]; %#ok<AGROW>
        bvec = [bvec; RU(g) + M]; %#ok<AGROW>

        % Pg_{t-1} - Pg_t <= RD + M*(1-u_t)
        row = sparse(1,nVar);
        row(idx.Pg(g,t-1)) = 1;
        row(idx.Pg(g,t)) = -1;
        row(idx.u(g,t)) = M;
        A = [A; row]; %#ok<AGROW>
        bvec = [bvec; RD(g) + M]; %#ok<AGROW>
    end
end

% 6) 储能充放电互斥 + SOC 动态
for t = 1:T
    % Pch <= PchMax * smode
    row = sparse(1,nVar);
    row(idx.Pch(t)) = 1;
    row(idx.smode(t)) = -PchMax;
    A = [A; row]; %#ok<AGROW>
    bvec = [bvec; 0]; %#ok<AGROW>

    % Pdis <= PdisMax * (1-smode) => Pdis + PdisMax*smode <= PdisMax
    row = sparse(1,nVar);
    row(idx.Pdis(t)) = 1;
    row(idx.smode(t)) = PdisMax;
    A = [A; row]; %#ok<AGROW>
    bvec = [bvec; PdisMax]; %#ok<AGROW>

    % E(t+1) = E(t) + etaCh*Pch - Pdis/etaDis
    row = sparse(1,nVar);
    row(idx.E(t+1)) = 1;
    row(idx.E(t)) = -1;
    row(idx.Pch(t)) = -etaCh;
    row(idx.Pdis(t)) = 1/etaDis;
    Aeq = [Aeq; row]; %#ok<AGROW>
    beq = [beq; 0]; %#ok<AGROW>

    % Rs <= PdisMax - Pdis => Rs + Pdis <= PdisMax
    row = sparse(1,nVar);
    row(idx.Rs(t)) = 1;
    row(idx.Pdis(t)) = 1;
    A = [A; row]; %#ok<AGROW>
    bvec = [bvec; PdisMax]; %#ok<AGROW>

    % Rs <= etaDis * E(t)
    row = sparse(1,nVar);
    row(idx.Rs(t)) = 1;
    row(idx.E(t)) = -etaDis;
    A = [A; row]; %#ok<AGROW>
    bvec = [bvec; 0]; %#ok<AGROW>
end

% SOC 首末一致
row = sparse(1,nVar); row(idx.E(1)) = 1; Aeq = [Aeq; row]; beq = [beq; E0]; %#ok<AGROW>
row = sparse(1,nVar); row(idx.E(T+1)) = 1; Aeq = [Aeq; row]; beq = [beq; E0]; %#ok<AGROW>

% 7) 功率平衡（MILP基线中不含网损）
for t = 1:T
    row = sparse(1,nVar);
    row(idx.Pg(:,t)) = 1;
    row(idx.Pdis(t)) = 1;
    row(idx.Pch(t)) = -1;
    Aeq = [Aeq; row]; %#ok<AGROW>
    beq = [beq; loadProfile(t)]; %#ok<AGROW>
end

% 8) 备用约束：sum(Rg)+Rs >= Rreq
for t = 1:T
    row = sparse(1,nVar);
    row(idx.Rg(:,t)) = -1;
    row(idx.Rs(t)) = -1;
    A = [A; row]; %#ok<AGROW>
    bvec = [bvec; -Rreq]; %#ok<AGROW>
end

%% ------------------------ 上下界/整数变量 ------------------------
lb = zeros(nVar,1);
ub = inf(nVar,1);

for t = 1:T
    for g = 1:G
        ub(idx.Pg(g,t)) = Pmax(g);
        ub(idx.Rg(g,t)) = max(RU(g), Pmax(g));
        for s = 1:nSeg
            ub(idx.seg(g,s,t)) = segCap(g,s);
        end
    end
    ub(idx.Pch(t)) = PchMax;
    ub(idx.Pdis(t)) = PdisMax;
    ub(idx.Rs(t)) = PdisMax;
end
for t = 1:T+1
    ub(idx.E(t)) = Emax;
end

intcon = [idx.u(:); idx.y(:); idx.z(:); idx.smode(:)]';
ub(intcon) = 1;

%% ------------------------ 求解 ------------------------
opts = optimoptions('intlinprog', ...
    'Display','iter', ...
    'Heuristics','advanced', ...
    'CutGeneration','advanced', ...
    'RelativeGapTolerance',1e-4, ...
    'MaxTime',300);

[x, fval, exitflag, output] = intlinprog(f, intcon, A, bvec, Aeq, beq, lb, ub, opts);

if exitflag <= 0
    warning('intlinprog 未找到最优可行解，exitflag=%d', exitflag);
end

%% ------------------------ 结果还原与评估 ------------------------
Pg = reshape(x(idx.Pg), [G,T])';
uRaw = reshape(x(idx.u), [G,T])';
yRaw = reshape(x(idx.y), [G,T])';
zRaw = reshape(x(idx.z), [G,T])';
intTol = 1e-6;
if max(abs(uRaw(:) - round(uRaw(:)))) > intTol || ...
   max(abs(yRaw(:) - round(yRaw(:)))) > intTol || ...
   max(abs(zRaw(:) - round(zRaw(:)))) > intTol
    warning('存在接近但非整数的二进制解，已按最近整数处理。');
end
u = round(uRaw);
y = round(yRaw);
z = round(zRaw);
Pch = x(idx.Pch(:));
Pdis = x(idx.Pdis(:));
E = x(idx.E(:));
Rg = reshape(x(idx.Rg), [G,T])';
Rs = x(idx.Rs(:));

balanceResidual = sum(Pg,2) + Pdis - Pch - loadProfile;
reserveMargin = sum(Rg,2) + Rs - Rreq;

rampVio = 0;
for g = 1:G
    for t = 2:T
        if u(t-1,g) == 1 && u(t,g) == 1
            rampVio = rampVio + max(0, Pg(t,g)-Pg(t-1,g)-RU(g));
            rampVio = rampVio + max(0, Pg(t-1,g)-Pg(t,g)-RD(g));
        end
    end
end

results = struct();
results.total_cost = fval;
results.Pg = Pg;
results.u = u;
results.y = y;
results.z = z;
results.Pch = Pch;
results.Pdis = Pdis;
results.E = E;
results.Rg = Rg;
results.Rs = Rs;
results.balance_mm = sum(abs(balanceResidual));
results.reserve_vio = sum(max(0,-reserveMargin));
results.ramp_vio = rampVio;
results.soc_end = E(end);
results.exitflag = exitflag;
results.output = output;

fprintf('\n===============================================================\n');
fprintf('非学习算法对照组（MILP + intlinprog）\n');
fprintf('===============================================================\n');
fprintf('Objective (MILP):      %.4f\n', results.total_cost);
fprintf('Balance mismatch sum:  %.6e MW\n', results.balance_mm);
fprintf('Reserve violation sum: %.6e MW\n', results.reserve_vio);
fprintf('Ramp violation sum:    %.6e MW\n', results.ramp_vio);
fprintf('SOC end:               %.3f MWh (target %.3f)\n', results.soc_end, E0);

fprintf('\nHour | u(G1 G2 G3) |  Pg(G1 G2 G3)   |  Pch  Pdis |   E\n');
for t = 1:T
    fprintf('%02d   | %d  %d  %d   | %7.2f %7.2f %7.2f | %5.2f %5.2f | %7.2f\n', ...
        t-1, u(t,1), u(t,2), u(t,3), Pg(t,1), Pg(t,2), Pg(t,3), Pch(t), Pdis(t), E(t));
end

% 与学习算法结果对照（可选）
if ~isempty(rlResult)
    fprintf('\n====================== 对照实验结果 ======================\n');
    fprintf('%-24s %-16s %-16s\n', '指标', '学习算法', 'MILP基线');
    fprintf('%-24s %-16.4f %-16.4f\n', 'total_cost', rlResult.total_cost, results.total_cost);
    fprintf('%-24s %-16.6e %-16.6e\n', 'balance_mm', rlResult.balance_mm, results.balance_mm);
    fprintf('%-24s %-16.6e %-16.6e\n', 'reserve_vio', rlResult.reserve_vio, results.reserve_vio);
    fprintf('%-24s %-16.6e %-16.6e\n', 'ramp_vio', rlResult.ramp_vio, results.ramp_vio);
    fprintf('%-24s %-16.3f %-16.3f\n', 'soc_end', rlResult.soc_end, results.soc_end);
end
end
