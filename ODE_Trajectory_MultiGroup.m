% Sim_Traj_Cmulti_4IC_3Regimes.m
% Generate ODE trajectories for 4 ICs x 3 regimes (general c)
% Stores time series of:
%   O(t)        = |mean(m_g)|
%   O_intra(t)  = mean(|m_g|)
%   std(m_g)(t)
%
% Output:
%   ModelRuns/ODE_Cmulti_Trajectories_4IC.mat  (struct Traj4)

clear; clc;

%% =========================
% Settings
%% =========================
c = 100;                 % even c recommended
Tmax = 120;
RelTol = 1e-7;
AbsTol = 1e-9;

% IC controls
eps_m = 5e-2;
s0    = 2/3;

% Random IC controls
nGroupForMCIC = 100;
seedBase = 24680;

% Regimes: [a, p]
% Row 1: high p (disordered)
% Row 2: low p, low a (global order)
% Row 3: low p, high a (modular order)
regimes = [ ...
    0.90, 0.40;   % disordered (high p)
    0.10, 0.10;   % global order (low p, low a)
    0.90, 0.10];  % modular order (low p, high a)

% Output
if ~exist('ModelRuns','dir'), mkdir('ModelRuns'); end
outFile = fullfile('ModelRuns','ODE_Cmulti_Trajectories_4IC.mat');

if mod(c,2) ~= 0
    warning('c is odd: Contrast/Almost-Contrast ICs are defined assuming even c.');
end

opts = odeset('RelTol',RelTol,'AbsTol',AbsTol);

%% =========================
% Allocate struct
%% =========================
% rows = regimes (3), cols = IC types (4)
% IC order:
%   1 Uniform
%   2 Contrast
%   3 Random
%   4 Almost-Contrast
Traj4 = struct();
Traj4.c = c;
Traj4.Tmax = Tmax;
Traj4.eps_m = eps_m;
Traj4.s0 = s0;
Traj4.regimes = regimes;
Traj4.IC_names = {'Uniform IC','Contrast IC','Random IC','Almost-Contrast IC'};

for r = 1:3
    for ic = 1:4
        Traj4.data(r,ic).a = [];
        Traj4.data(r,ic).p = [];
        Traj4.data(r,ic).t = [];
        Traj4.data(r,ic).m = [];
        Traj4.data(r,ic).s = [];
        Traj4.data(r,ic).O = [];
        Traj4.data(r,ic).Ointra = [];
        Traj4.data(r,ic).StdMg = [];
    end
end

%% =========================
% Run all 12 cases
%% =========================
for r = 1:3
    a = regimes(r,1);
    p = regimes(r,2);

    Pi = build_pi_planted(c, a);

    for ic = 1:4
        % --- Build IC ---
        switch ic
            case 1
                [m0, s0vec] = ic_uniform_from_fspace(c, s0, eps_m);

            case 2
                [m0, s0vec] = ic_contrast_from_fspace(c, s0, eps_m);

            case 3
                seed = seedBase + 100*r + ic;
                [m0, s0vec] = ic_random_multinomial_fspace(c, nGroupForMCIC, seed);

            case 4
                [m0, s0vec] = ic_almost_contrast_from_fspace(c, s0, eps_m);

            otherwise
                error('Unknown IC index');
        end

        % --- Integrate ---
        [t, Y] = integrate_full(Pi, p, m0, s0vec, Tmax, opts);

        % unpack
        m = Y(:,1:c);
        s = Y(:,c+1:2*c);

        % observables
        O      = abs(mean(m,2));
        Ointra = mean(abs(m),2);
        StdMg  = std(m,0,2);

        % store
        Traj4.data(r,ic).a = a;
        Traj4.data(r,ic).p = p;
        Traj4.data(r,ic).t = t;
        Traj4.data(r,ic).m = m;
        Traj4.data(r,ic).s = s;
        Traj4.data(r,ic).O = O;
        Traj4.data(r,ic).Ointra = Ointra;
        Traj4.data(r,ic).StdMg = StdMg;

        fprintf('Done regime %d/3, IC %d/4  (a=%.2f, p=%.2f)\n', r, ic, a, p);
    end
end

%% Save
save(outFile, 'Traj4');
fprintf('Saved: %s\n', outFile);

%% =========================
% Local functions
%% =========================

function Pi = build_pi_planted(c, a)
    if c <= 1
        Pi = 1;
        return;
    end
    b = (1-a)/(c-1);
    Pi = b*ones(c) + (a-b)*eye(c);
end

function dydt = bchs_sbm_rhs(~, y, Pi, p)
    c = size(Pi,1);
    m = y(1:c);
    s = y(c+1:2*c);

    M = Pi*m;
    S = Pi*s;

    mdot = (1-2*p).*(1 - 0.5*s).*M - 0.5*S.*m;
    sdot = S.*(1 - 1.5*s) + 0.5*(1-2*p).*(m.*M);

    dydt = [mdot; sdot];
end

function [t, Y] = integrate_full(Pi, p, m0, s0, Tmax, opts)
    y0 = [m0(:); s0(:)];
    f = @(t,y) bchs_sbm_rhs(t,y,Pi,p);
    [t, Y] = ode15s(f, [0 Tmax], y0, opts);
end

function [m0, s0vec] = ic_uniform_from_fspace(c, s0, eps_m)
    s0 = min(max(s0,0),1);
    m  = min(eps_m,s0);

    fplus  = 0.5*(s0 + m);
    fminus = 0.5*(s0 - m);

    m0    = repmat(fplus - fminus, c, 1);
    s0vec = repmat(fplus + fminus, c, 1);
end

function [m0, s0vec] = ic_contrast_from_fspace(c, s0, eps_m)
    s0 = min(max(s0,0),1);
    m  = min(eps_m,s0);

    fplusP  = 0.5*(s0 + m);   fminusP = 0.5*(s0 - m);
    fplusN  = 0.5*(s0 - m);   fminusN = 0.5*(s0 + m);

    m0    = zeros(c,1);
    s0vec = zeros(c,1);

    nHalf = floor(c/2);

    m0(1:nHalf)        = (fplusP - fminusP);
    s0vec(1:nHalf)     = (fplusP + fminusP);

    m0(nHalf+1:end)    = (fplusN - fminusN);
    s0vec(nHalf+1:end) = (fplusN + fminusN);
end

function [m0, s0vec] = ic_almost_contrast_from_fspace(c, s0, eps_m)
    s0 = min(max(s0,0),1);
    m  = min(eps_m,s0);

    fplusP  = 0.5*(s0 + m);   fminusP = 0.5*(s0 - m);
    fplusN  = 0.5*(s0 - m);   fminusN = 0.5*(s0 + m);

    m0    = zeros(c,1);
    s0vec = zeros(c,1);

    nPos = floor(c/2) + 1;

    m0(1:nPos)         = (fplusP - fminusP);
    s0vec(1:nPos)      = (fplusP + fminusP);

    m0(nPos+1:end)     = (fplusN - fminusN);
    s0vec(nPos+1:end)  = (fplusN + fminusN);
end

function [m0, s0vec] = ic_random_multinomial_fspace(c, nPerGroup, seed)
    rng(seed, 'twister');

    m0    = zeros(c,1);
    s0vec = zeros(c,1);

    for g = 1:c
        counts = multinomial_counts_fast(nPerGroup); % [nPlus, nMinus, nZero]
        nPlus  = counts(1);
        nMinus = counts(2);

        fplus  = nPlus / nPerGroup;
        fminus = nMinus / nPerGroup;

        m0(g)    = fplus - fminus;
        s0vec(g) = fplus + fminus;
    end
end

function counts = multinomial_counts_fast(n)
    r = rand(n,1);
    counts = histcounts(r, [0, 1/3, 2/3, 1]); % [+, -, 0]
end