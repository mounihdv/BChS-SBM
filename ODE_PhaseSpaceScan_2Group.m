% ODE_Scan_C2_3IC.m
% Scan full (a,p) in [0,1]x[0,1] for c=2 using mean-field ODEs in (m,s)
% 3 IC types:
%   1) symmetric IC
%   2) antisymmetric IC
%   3) random IC (MC-like multinomial initial fractions), ensemble averaged
%
% Outputs: grids for O and O_intra for each IC type, saved to .mat

clear; clc;

%% Settings
c = 2;

Na = 40;                 % grid resolution in a
Np = 40;                 % grid resolution in p
a_vals = linspace(0,1,Na);
p_vals = linspace(0,1,Np);

% --- NEW: two different integration horizons ---
Tmax_single = 100;        % for symmetric + antisymmetric single-run ICs
Tmax_random = 100;        % for random IC ensemble runs

RelTol = 1e-7;
AbsTol = 1e-9;

tailFrac = 0.20;         % average over last 20% of trajectory points

% IC controls
eps_m = 5e-2;            % small seed magnetisation for symmetric/antisymmetric IC
s0    = 2/3;             % start near active disordered activity
% Random IC (MC-like)
nIC_random = 30;         % ensemble size for random IC scan
nGroupForMCIC = 100;     % "agents per group" for multinomial draw (large reduces noise)

% Output filename
outFile = fullfile('ModelRuns','ODE_Scan_C2_3IC.mat');
if ~exist('ModelRuns','dir'), mkdir('ModelRuns'); end

%% Preallocate result grids
Grid_O_sym     = zeros(Np, Na);
Grid_Intra_sym = zeros(Np, Na);

Grid_O_anti     = zeros(Np, Na);
Grid_Intra_anti = zeros(Np, Na);

Grid_O_rand     = zeros(Np, Na);
Grid_Intra_rand = zeros(Np, Na);

%% ODE options
opts = odeset('RelTol',RelTol,'AbsTol',AbsTol);

%% Main scan
for ia = 1:Na
    a = a_vals(ia);

    % Two-group mixing matrix
    Pi = [a, 1-a; 1-a, a];

    for ip = 1:Np
        p = p_vals(ip);

        % ---------------------------
        % 1) Symmetric IC (single-run horizon)
        % ---------------------------
        [m0_sym, s0_sym] = ic_sym_from_fspace(c, s0, eps_m);
        [O_sym, Oin_sym] = integrate_and_measure(Pi, p, m0_sym, s0_sym, Tmax_single, opts, tailFrac);

        Grid_O_sym(ip, ia)     = O_sym;
        Grid_Intra_sym(ip, ia) = Oin_sym;

        % ---------------------------
        % 2) Antisymmetric IC (single-run horizon)
        % ---------------------------
        [m0_anti, s0_anti] = ic_antisym_from_fspace(c, s0, eps_m);
        [O_anti, Oin_anti] = integrate_and_measure(Pi, p, m0_anti, s0_anti, Tmax_single, opts, tailFrac);

        Grid_O_anti(ip, ia)     = O_anti;
        Grid_Intra_anti(ip, ia) = Oin_anti;

        % ---------------------------
        % 3) Random IC (MC-like), ensemble average (random horizon)
        % ---------------------------
        O_acc = 0; Oin_acc = 0;

        for k = 1:nIC_random
            [m0_r, s0_r] = ic_random_multinomial_fspace(c, nGroupForMCIC);
            [O_r, Oin_r] = integrate_and_measure(Pi, p, m0_r, s0_r, Tmax_random, opts, tailFrac);

            O_acc   = O_acc   + O_r;
            Oin_acc = Oin_acc + Oin_r;
        end

        Grid_O_rand(ip, ia)     = O_acc  / nIC_random;
        Grid_Intra_rand(ip, ia) = Oin_acc / nIC_random;

    end
    fprintf('Done a = %.3f (%d/%d)\n', a, ia, Na);
end

%% Save
save(outFile, ...
    'a_vals','p_vals', ...
    'Grid_O_sym','Grid_Intra_sym', ...
    'Grid_O_anti','Grid_Intra_anti', ...
    'Grid_O_rand','Grid_Intra_rand', ...
    'Na','Np','c', ...
    'Tmax_single','Tmax_random','RelTol','AbsTol','tailFrac', ...
    'eps_m','s0','nIC_random','nGroupForMCIC');

fprintf('Saved: %s\n', outFile);

%% =========================
%  Local functions
% =========================

function dydt = bchs_sbm_rhs(~, y, Pi, p)
    % y = [m; s], both c x 1
    c = size(Pi,1);
    m = y(1:c);
    s = y(c+1:2*c);

    M = Pi*m;
    S = Pi*s;

    % Mean-field ODEs (verified against your derivation)
    mdot = (1-2*p).*(1 - 0.5*s).*M - 0.5*S.*m;
    sdot = S.*(1 - 1.5*s) + 0.5*(1-2*p).*(m.*M);

    dydt = [mdot; sdot];
end

function [O, Ointra] = integrate_and_measure(Pi, p, m0, s0, Tmax, opts, tailFrac)
    c = numel(m0);
    y0 = [m0(:); s0(:)];

    % Integrate
    f = @(t,y) bchs_sbm_rhs(t,y,Pi,p);
    [~, Y] = ode15s(f, [0 Tmax], y0, opts);

    % Tail-average
    nT = size(Y,1);
    i0 = max(1, floor((1-tailFrac)*nT));
    m_tail = mean(Y(i0:end, 1:c), 1);  % 1 x c

    % Order parameters (c=2 generalises trivially)
    O      = abs(mean(m_tail));
    Ointra = mean(abs(m_tail));

    % Numerical sanity (reporting only; does not affect dynamics)
    O      = min(max(O,0),1);
    Ointra = min(max(Ointra,0),1);
end

function [m0, s0vec] = ic_sym_from_fspace(c, s0, eps_m)
    % Build a symmetric IC in f-space, then map to (m,s)
    % Same (f+,f-,f0) in both groups, with small magnetisation seed.
    s0 = min(max(s0, 0), 1);
    m  = min(eps_m, s0);  % ensure |m|<=s

    fplus  = 0.5*(s0 + m);
    fminus = 0.5*(s0 - m);
    f0     = 1 - s0; %#ok<NASGU>

    % replicate across groups
    fplus_g  = repmat(fplus,  c, 1);
    fminus_g = repmat(fminus, c, 1);
    % map to m,s
    m0    = fplus_g - fminus_g;
    s0vec = fplus_g + fminus_g;

    m0    = m0(:);
    s0vec = s0vec(:);
end

function [m0, s0vec] = ic_antisym_from_fspace(c, s0, eps_m)
    % Build an antisymmetric IC in f-space, then map to (m,s)
    % For c=2: m2=-m1, s2=s1.
    if c ~= 2
        error('Antisymmetric IC is defined here for c=2 only.');
    end

    s0 = min(max(s0, 0), 1);
    m  = min(eps_m, s0);

    % Group 1 has +m, Group 2 has -m, same s
    fplus1  = 0.5*(s0 + m);
    fminus1 = 0.5*(s0 - m);

    fplus2  = 0.5*(s0 - m);
    fminus2 = 0.5*(s0 + m);

    m0    = [fplus1 - fminus1; fplus2 - fminus2];
    s0vec = [fplus1 + fminus1; fplus2 + fminus2];
end

function [m0, s0vec] = ic_random_multinomial_fspace(c, nPerGroup)
    % MC-like random IC:
    % For each group, draw counts from uniform opinions {-1,0,+1} => probs [1/3,1/3,1/3]
    % Convert to fractions, then map to (m,s).
    probs = [1/3, 1/3, 1/3];

    m0    = zeros(c,1);
    s0vec = zeros(c,1);

    for g = 1:c
        counts = multinomial_counts(nPerGroup, probs); % [nPlus, nMinus, nZero]
        nPlus  = counts(1);
        nMinus = counts(2);
        nZero  = counts(3); %#ok<NASGU>

        fplus  = nPlus / nPerGroup;
        fminus = nMinus / nPerGroup;

        m0(g)    = fplus - fminus;
        s0vec(g) = fplus + fminus;
    end
end

function counts = multinomial_counts(n, probs)
    % Returns counts for categories 1..K with probabilities probs (sum=1).
    % No Statistics Toolbox required.
    K = numel(probs);
    edges = [0, cumsum(probs)];
    edges(end) = 1;
    r = rand(n,1);
    idx = discretize(r, edges);
    counts = zeros(1,K);
    for k = 1:K
        counts(k) = sum(idx == k);
    end
end