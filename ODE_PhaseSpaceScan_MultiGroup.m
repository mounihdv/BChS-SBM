% ODE_Scan_Cmulti_4IC_FullPlane_FAST.m
% Fast full-plane scan for general c using mean-field ODEs in (m,s)
% 4 IC types:
%   1) Uniform IC        : all groups +eps_m
%   2) Contrast IC       : half +eps_m, half -eps_m (requires even c)
%   3) Random IC         : MC-like multinomial initial fractions, ensemble averaged
%   4) Almost-Contrast   : (c/2+1) groups +eps_m, remaining -eps_m (even c assumed)
%
% Parallelisation:
%   - Single parfor over ALL grid points (idx = 1..Na*Np) with 5 workers.
%   - Deterministic random IC via seeds per (idx,k).
%
% Output:
%   Saves ModelRuns/ODE_Scan_C{c}_4IC_FullPlane.mat with Np x Na grids.

%clear; clc;

%% Settings
c = 100;                 % choose even c for Contrast / Almost-Contrast

Na = 40;                 % resolution in a
Np = 40;                 % resolution in p
a_vals = linspace(0,1,Na);
p_vals = linspace(0,1,Np);

Tmax_single = 120;
Tmax_random = 120;

RelTol = 1e-7;
AbsTol = 1e-9;
tailFrac = 0.20;

eps_m = 1e-1;
s0    = 2/3;

nIC_random     = 30;
nGroupForMCIC  = 100;

seedBase = 12345;

outFile = fullfile('ModelRuns', sprintf('ODE_Scan_C%d_4IC_FullPlane.mat', c));
if ~exist('ModelRuns','dir'), mkdir('ModelRuns'); end

if mod(c,2) ~= 0
    warning('c is odd: Contrast/Almost-Contrast ICs are defined assuming even c.');
end

%% ODE options
opts = odeset('RelTol',RelTol,'AbsTol',AbsTol);

%% Parallel pool (5 workers)
pobj = gcp('nocreate');
if isempty(pobj)
    parpool('local', 5);
else
    if pobj.NumWorkers ~= 5
        delete(pobj);
        parpool('local', 5);
    end
end

%% Flat storage (fast + parfor-safe)
Ntot = Na * Np;

O_uni_f      = zeros(Ntot,1);
Oin_uni_f    = zeros(Ntot,1);

O_con_f      = zeros(Ntot,1);
Oin_con_f    = zeros(Ntot,1);

O_rand_f     = zeros(Ntot,1);
Oin_rand_f   = zeros(Ntot,1);

O_almost_f   = zeros(Ntot,1);
Oin_almost_f = zeros(Ntot,1);

elapsed_f    = zeros(Ntot,1);

%% Main scan (fastest: parfor over all grid points)
parfor idx = 1:Ntot
    t0 = tic;

    % Map linear index -> (ip, ia)
    [ip, ia] = ind2sub([Np, Na], idx);

    a = a_vals(ia);
    p = p_vals(ip);

    % Build Pi for this a
    Pi = build_pi_planted(c, a);

    % --- Precompute the three deterministic ICs once per point ---
    [m0_u, s0_u] = ic_uniform_from_fspace(c, s0, eps_m);
    [m0_c, s0_c] = ic_contrast_from_fspace(c, s0, eps_m);
    [m0_a, s0_a] = ic_almost_contrast_from_fspace(c, s0, eps_m);

    % 1) Uniform
    [Ou, Oinu] = integrate_and_measure(Pi, p, m0_u, s0_u, Tmax_single, opts, tailFrac);
    O_uni_f(idx)   = Ou;
    Oin_uni_f(idx) = Oinu;

    % 2) Contrast
    [Oc, Oinc] = integrate_and_measure(Pi, p, m0_c, s0_c, Tmax_single, opts, tailFrac);
    O_con_f(idx)   = Oc;
    Oin_con_f(idx) = Oinc;

    % 3) Random (ensemble)
    O_acc = 0; Oin_acc = 0;
    for k = 1:nIC_random
        seed = seedBase + 1000000*idx + k;  % deterministic per (idx,k)
        [m0_r, s0_r] = ic_random_multinomial_fspace(c, nGroupForMCIC, seed);
        [Or, Oinr]   = integrate_and_measure(Pi, p, m0_r, s0_r, Tmax_random, opts, tailFrac);
        O_acc   = O_acc   + Or;
        Oin_acc = Oin_acc + Oinr;
    end
    O_rand_f(idx)   = O_acc / nIC_random;
    Oin_rand_f(idx) = Oin_acc / nIC_random;

    % 4) Almost-contrast
    [Oa, Oina] = integrate_and_measure(Pi, p, m0_a, s0_a, Tmax_single, opts, tailFrac);
    O_almost_f(idx)   = Oa;
    Oin_almost_f(idx) = Oina;

    elapsed_f(idx) = toc(t0);
end

%% Reshape back to grids (Np x Na)
Grid_O_uni        = reshape(O_uni_f,      [Np, Na]);
Grid_Intra_uni    = reshape(Oin_uni_f,    [Np, Na]);

Grid_O_con        = reshape(O_con_f,      [Np, Na]);
Grid_Intra_con    = reshape(Oin_con_f,    [Np, Na]);

Grid_O_rand       = reshape(O_rand_f,     [Np, Na]);
Grid_Intra_rand   = reshape(Oin_rand_f,   [Np, Na]);

Grid_O_almost     = reshape(O_almost_f,   [Np, Na]);
Grid_Intra_almost = reshape(Oin_almost_f, [Np, Na]);

ElapsedMap        = reshape(elapsed_f,    [Np, Na]);

%% Save
save(outFile, ...
    'a_vals','p_vals', ...
    'Grid_O_uni','Grid_Intra_uni', ...
    'Grid_O_con','Grid_Intra_con', ...
    'Grid_O_rand','Grid_Intra_rand', ...
    'Grid_O_almost','Grid_Intra_almost', ...
    'ElapsedMap', ...
    'Na','Np','c', ...
    'Tmax_single','Tmax_random','RelTol','AbsTol','tailFrac', ...
    'eps_m','s0','nIC_random','nGroupForMCIC','seedBase');

fprintf('Saved: %s\n', outFile);

%% =========================
% Local functions
%% =========================

function Pi = build_pi_planted(c, a)
    if c <= 1
        Pi = 1;
        return;
    end
    b  = (1-a)/(c-1);
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

function [O, Ointra] = integrate_and_measure(Pi, p, m0, s0, Tmax, opts, tailFrac)
    c  = numel(m0);
    y0 = [m0(:); s0(:)];

    f = @(t,y) bchs_sbm_rhs(t,y,Pi,p);
    [~, Y] = ode15s(f, [0 Tmax], y0, opts);

    nT = size(Y,1);
    i0 = max(1, floor((1-tailFrac)*nT));
    m_tail = mean(Y(i0:end, 1:c), 1);

    O      = abs(mean(m_tail));
    Ointra = mean(abs(m_tail));

    O      = min(max(O,0),1);
    Ointra = min(max(Ointra,0),1);
end

function [m0, s0vec] = ic_uniform_from_fspace(c, s0, eps_m)
    s0 = min(max(s0, 0), 1);
    m  = min(eps_m, s0);

    fplus  = 0.5*(s0 + m);
    fminus = 0.5*(s0 - m);

    m0    = repmat(fplus - fminus,  c, 1);
    s0vec = repmat(fplus + fminus, c, 1);
end

function [m0, s0vec] = ic_contrast_from_fspace(c, s0, eps_m)
    s0 = min(max(s0, 0), 1);
    m  = min(eps_m, s0);

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
    s0 = min(max(s0, 0), 1);
    m  = min(eps_m, s0);

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

    % probs = [1/3,1/3,1/3] for (+,-,0)
    m0    = zeros(c,1);
    s0vec = zeros(c,1);

    for g = 1:c
        counts = multinomial_counts_fast(nPerGroup);
        nPlus  = counts(1);
        nMinus = counts(2);

        fplus  = nPlus / nPerGroup;
        fminus = nMinus / nPerGroup;

        m0(g)    = fplus - fminus;
        s0vec(g) = fplus + fminus;
    end
end

function counts = multinomial_counts_fast(n)
    % Fast multinomial for uniform probs (1/3,1/3,1/3), no toolbox:
    % Use one rand vector + histcounts on bins [0,1/3,2/3,1]
    r = rand(n,1);
    counts = histcounts(r, [0, 1/3, 2/3, 1]);  % [nPlus, nMinus, nZero]
end