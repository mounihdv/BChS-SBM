% ODE_C2_Trajectories_Sim.m
% Simulate c=2 mean-field ODE trajectories for:
%  - 3 regimes (disordered, modular symmetric-order sector, bipartite contrarian sector)
%  - 2 ICs (symmetric, antisymmetric)
% Saves trajectories to ModelRuns/ODE_C2_Trajectories.mat

clear; clc;

if ~exist('ModelRuns','dir'), mkdir('ModelRuns'); end

%% --- Choose representative parameter points (edit if you want) ---
% Regime 1: disordered (p > 0.25)
par(1).name = 'Disordered';
par(1).a = 0.80;
par(1).p = 0.35;

% Regime 2: modular + low-p (symmetric ordering sector)
par(2).name = 'ModularLowP';
par(2).a = 0.95;   % > 0.5 (assortative)
par(2).p = 0.05;   % < 0.25

% Regime 3: bipartite contrarian (antisymmetric ordering sector)
par(3).name = 'BipartiteHighP';
par(3).a = 0.05;   % < 0.5 (disassortative)
par(3).p = 0.95;   % > 0.5

%% --- Integration settings ---
Tmax   = 100;               % extend if needed
tspan  = [0 Tmax];
RelTol = 1e-8;
AbsTol = 1e-10;
opts   = odeset('RelTol',RelTol,'AbsTol',AbsTol);

% IC seed sizes (keep small but nonzero)
eps_m  = 5e-2;             % magnetisation seed magnitude
s0     = 2/3;              % activity seed (start near active disordered)

%% --- Build and run simulations ---
IC(1).name = 'SymIC';
IC(2).name = 'AntiIC';

Traj = struct();
Traj.meta.c = 2;
Traj.meta.Tmax = Tmax;
Traj.meta.RelTol = RelTol;
Traj.meta.AbsTol = AbsTol;
Traj.meta.eps_m = eps_m;
Traj.meta.s0 = s0;
Traj.par = par;
Traj.IC = IC;

for r = 1:numel(par)
    a = par(r).a;
    p = par(r).p;
    Pi = [a, 1-a; 1-a, a];

    for ic = 1:2
        if ic == 1
            [m0, s0vec] = ic_sym_from_fspace(s0, eps_m);
        else
            [m0, s0vec] = ic_antisym_from_fspace(s0, eps_m);
        end

        y0 = [m0(:); s0vec(:)];
        f  = @(t,y) bchs_sbm_rhs_c2(t, y, Pi, p);

        [t, Y] = ode15s(f, tspan, y0, opts);

        % unpack
        m = Y(:,1:2);
        s = Y(:,3:4);

        Traj.data(r,ic).t = t;
        Traj.data(r,ic).m = m;
        Traj.data(r,ic).s = s;
        Traj.data(r,ic).a = a;
        Traj.data(r,ic).p = p;
        Traj.data(r,ic).regime = par(r).name;
        Traj.data(r,ic).ic = IC(ic).name;
    end
    fprintf('Done regime %d/%d (%s)\n', r, numel(par), par(r).name);
end

outFile = fullfile('ModelRuns','ODE_C2_Trajectories.mat');
save(outFile, 'Traj');
fprintf('Saved trajectories to %s\n', outFile);

%% =========================
% Local functions (end of script)
%% =========================

function dydt = bchs_sbm_rhs_c2(~, y, Pi, p)
    % y = [m1 m2 s1 s2]^T
    m = y(1:2);
    s = y(3:4);

    M = Pi*m;
    S = Pi*s;

    % Verified mean-field ODEs:
    % mdot = (1-2p)(1-s/2).*M - (S/2).*m
    % sdot = S(1-3s/2) + (1-2p)/2 * (m.*M)
    mdot = (1-2*p).*(1 - 0.5*s).*M - 0.5*S.*m;
    sdot = S.*(1 - 1.5*s) + 0.5*(1-2*p).*(m.*M);

    dydt = [mdot; sdot];
end

function [m0, s0vec] = ic_sym_from_fspace(s0, eps_m)
    % Symmetric IC: both groups identical in f-space, then map to (m,s)
    s0 = min(max(s0,0),1);
    m  = min(eps_m, s0);

    fplus  = 0.5*(s0 + m);
    fminus = 0.5*(s0 - m);

    m0    = [fplus - fminus; fplus - fminus];
    s0vec = [fplus + fminus; fplus + fminus];
end

function [m0, s0vec] = ic_antisym_from_fspace(s0, eps_m)
    % Antisymmetric IC: m2=-m1, same activity s in both groups
    s0 = min(max(s0,0),1);
    m  = min(eps_m, s0);

    % group 1: +m, group 2: -m in f-space
    fplus1  = 0.5*(s0 + m);  fminus1 = 0.5*(s0 - m);
    fplus2  = 0.5*(s0 - m);  fminus2 = 0.5*(s0 + m);

    m0    = [fplus1 - fminus1; fplus2 - fminus2];
    s0vec = [fplus1 + fminus1; fplus2 + fminus2];
end