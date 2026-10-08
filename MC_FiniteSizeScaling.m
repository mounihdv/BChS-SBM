%% FiniteSizeScaling_CScan_BChS_SBM.m
% =========================================================================
% Finite-size scaling of the BChS model on planted-partition SBMs,
% SCALING THE NUMBER OF GROUPS c AT FIXED GROUP SIZE.
%
% Design
% ------
%   * Every group has exactly nGroup = 100 members.
%   * The number of groups is scanned: c = [13 25 51 101 151].
%   * Total size N = nGroup*c = [1300 2500 5100 10100 15100] follows.
%
% IMPORTANT ANALYSIS NOTE (why a is NOT the right collapse variable here)
% ----------------------------------------------------------------------
% The contrast eigenvalue of the mixing matrix is
%
%   lambda(a,c) = (a*c - 1)/(c - 1),
%
% so at fixed disagreement probability p the critical point in a DRIFTS
% with c:  a_c(c) = lambda* + (1 - lambda*)/c,  lambda* = 1/(2(1-2p)) on
% the existence line. Binder curves for different c therefore do NOT
% cross at a common a, and a collapse in (a - a_c) N^{1/nubar} with one
% a_c cannot work. Instead, analyse in lambda (or q = (1-2p)*lambda):
% lambda_c is c-independent, curves cross at lambda_c, and the collapse
% variable is (lambda - lambda_c) N^{1/nubar}. For convenience this
% script saves lambda_vec (per (a,p) pair) in every output file.
%
% Saved per realization (time moments over the measurement window), for
% each of the three order parameters:
%   Res_O,     Res_O2,     Res_O4      (global <|m|>, <m^2>, <m^4>)
%   Res_Intra, Res_Intra2, Res_Intra4  (I = mean_g|m_g| and powers)
%   Res_Std,   Res_Std2,   Res_Std4    (R = std(m_g) and powers)
%
% Output files:  ModelRuns/FSS_CScan_Run<RunID>_c<c>_Final.mat
% (one per c; N is stored inside and is always nGroup*c).
% =========================================================================
RunID = 21;

%% ========== 1. PHYSICS PARAMETERS ==========

nGroup   = 100;                    % members per group (fixed)
c_values = [13 25 51 101 151];     % number of groups (the size variable)
N_values = nGroup * c_values;      % total agents per system

k_avg  = 20;

% Fixed disagreement probabilities
p_values = 0.025;

% Self-mixing probability scan. The same a grid is used for every c;
% the c-dependent critical point is handled at analysis time via lambda.
n_a     = 40;
a_vals  = linspace(0.65, 1, n_a);

%% ========== 2. SIMULATION PARAMETERS ==========

numiter = 40;           % realizations per parameter point

%% ========== 3. PARAMETER GRID ==========

[A_GRID, P_GRID] = ndgrid(a_vals, p_values);
a_vec = A_GRID(:);
p_vec = P_GRID(:);
nPairs = numel(a_vec);

% Guidance printout: predicted existence-line a_c for every (p,c), from
% lambda* = 1/(2(1-2p)) and a_c = lambda* + (1-lambda*)/c. Check that the
% scanned a range brackets these before committing compute time.
fprintf('Predicted existence-line a_c(p,c):\n');
for ip = 1:numel(p_values)
    p0 = p_values(ip);
    lambdaStar = 1/(2*(1-2*p0));
    for ic = 1:numel(c_values)
        aC = lambdaStar + (1-lambdaStar)/c_values(ic);
        fprintf('  p=%.4f, c=%4d:  a_c ~ %.5f\n',p0,c_values(ic),aC);
    end
end

%% ========== 4. DIRECTORIES AND LOGGING ==========

outdir  = fullfile('ModelRuns', 'FSS_CScan_Checkpoints');
if ~exist(outdir, 'dir'), mkdir(outdir); end
if ~exist('ModelRuns', 'dir'), mkdir('ModelRuns'); end

logfile = sprintf('progress_log_FSS_CScan_Run%d.txt', RunID);

%% ========== 5. PARALLEL POOL ==========

pool = gcp('nocreate');
if isempty(pool)
    parpool('local', 4);
end

%% ========== 6. MAIN LOOP OVER GROUP COUNTS ==========

chunkSize = 12;   % checkpoint every 10 parameter pairs

for iC = 1:length(c_values)
    c = c_values(iC);
    N = nGroup * c;
    Tss     = N*100;          % burn-in steps (scales with N = 100c)
    T_full  = N*200;          % measurement steps

    % lambda for every (a,p) pair at this c: the analysis-side scan
    % coordinate with a c-independent critical value.
    lambda_vec = (a_vec*c - 1) / (c - 1);

    fprintf('\n====== c = %d, N = %d (size %d/%d) ======\n', ...
        c, N, iC, length(c_values));

    % --- Try to resume from checkpoint for this c ---
    fnameSearch = sprintf('FSS_CScan_Run%d_c%d_upTo_*.mat', RunID, c);
    checkpoint_files = dir(fullfile(outdir, fnameSearch));

    resumed = false;
    if ~isempty(checkpoint_files)
        [~, ix]   = max([checkpoint_files.datenum]);
        lastfile  = fullfile(checkpoint_files(ix).folder, checkpoint_files(ix).name);
        S         = load(lastfile);

        newMomentFields = {'Res_Intra2','Res_Intra4','Res_Std2','Res_Std4'};
        hasNewFormat = all(cellfun(@(f) isfield(S,f), newMomentFields));

        if hasNewFormat
            Res_O       = S.Res_O;
            Res_O2      = S.Res_O2;
            Res_O4      = S.Res_O4;
            Res_Intra   = S.Res_Intra;
            Res_Intra2  = S.Res_Intra2;
            Res_Intra4  = S.Res_Intra4;
            Res_Std     = S.Res_Std;
            Res_Std2    = S.Res_Std2;
            Res_Std4    = S.Res_Std4;
            itertimes   = S.itertimes;

            last_stopIdx = S.stopIdx;
            startIdx0    = last_stopIdx + 1;
            resumed = true;

            fprintf('  Resuming from %s (last stopIdx = %d)\n', lastfile, last_stopIdx);

            if startIdx0 > nPairs
                fprintf('  c=%d already complete. Skipping.\n', c);
                continue;
            end
        else
            warning(['Checkpoint %s is from an old script version ' ...
                '(missing Intra2/Intra4/Std2/Std4). Restarting c=%d ' ...
                'from scratch so the new moments are complete.'], ...
                lastfile, c);
        end
    end

    if ~resumed
        fprintf('  No usable checkpoint: starting from scratch.\n');

        Res_O       = zeros(nPairs, numiter);
        Res_O2      = zeros(nPairs, numiter);
        Res_O4      = zeros(nPairs, numiter);
        Res_Intra   = zeros(nPairs, numiter);
        Res_Intra2  = zeros(nPairs, numiter);
        Res_Intra4  = zeros(nPairs, numiter);
        Res_Std     = zeros(nPairs, numiter);
        Res_Std2    = zeros(nPairs, numiter);
        Res_Std4    = zeros(nPairs, numiter);
        itertimes   = zeros(1, nPairs);

        startIdx0 = 1;
    end

    tCStart = tic;

    % --- Chunked parfor with checkpoints ---
    for startIdx = startIdx0:chunkSize:nPairs

        stopIdx = min(startIdx + chunkSize - 1, nPairs);

        parfor idx = startIdx:stopIdx
            tIdxStart = tic;

            a = a_vec(idx);
            p = p_vec(idx);

            loc_O   = zeros(1, numiter);
            loc_O2  = zeros(1, numiter);
            loc_O4  = zeros(1, numiter);
            loc_I   = zeros(1, numiter);
            loc_I2  = zeros(1, numiter);
            loc_I4  = zeros(1, numiter);
            loc_S   = zeros(1, numiter);
            loc_S2  = zeros(1, numiter);
            loc_S4  = zeros(1, numiter);

            for kk = 1:numiter
                % Generate SBM from physical parameters
                EL = SBM_edgelist_Physical(N, c, k_avg, a);

                % Run BChS dynamics, collect moments of all three order
                % parameters (global, intra-group, modular spread)
                [O, O2, O4, Omod, Omod2, Omod4, Ostd, Ostd2, Ostd4] = ...
                    RunBCS_Moments(EL, p, Tss, T_full, c, N);

                loc_O(kk)   = O;
                loc_O2(kk)  = O2;
                loc_O4(kk)  = O4;
                loc_I(kk)   = Omod;
                loc_I2(kk)  = Omod2;
                loc_I4(kk)  = Omod4;
                loc_S(kk)   = Ostd;
                loc_S2(kk)  = Ostd2;
                loc_S4(kk)  = Ostd4;
            end

            Res_O(idx, :)      = loc_O;
            Res_O2(idx, :)     = loc_O2;
            Res_O4(idx, :)     = loc_O4;
            Res_Intra(idx, :)  = loc_I;
            Res_Intra2(idx, :) = loc_I2;
            Res_Intra4(idx, :) = loc_I4;
            Res_Std(idx, :)    = loc_S;
            Res_Std2(idx, :)   = loc_S2;
            Res_Std4(idx, :)   = loc_S4;

            itertimes(idx) = toc(tIdxStart);

            fid = fopen(logfile, 'a');
            if fid ~= -1
                fprintf(fid, 'c=%d | N=%d | idx=%d | a=%.3f, p=%.3f | t=%.1fmin | total=%.1fmin\n', ...
                    c, N, idx, a, p, itertimes(idx)/60, toc(tCStart)/60);
                fclose(fid);
            end
        end

        % --- Checkpoint after each chunk ---
        chkFile = fullfile(outdir, sprintf('FSS_CScan_Run%d_c%d_upTo_%d.mat', RunID, c, stopIdx));
        save(chkFile, ...
            'Res_O', 'Res_O2', 'Res_O4', ...
            'Res_Intra', 'Res_Intra2', 'Res_Intra4', ...
            'Res_Std', 'Res_Std2', 'Res_Std4', ...
            'a_vals', 'p_values', 'a_vec', 'p_vec', 'lambda_vec', ...
            'N', 'c', 'nGroup', 'c_values', 'N_values', 'k_avg', ...
            'numiter', 'Tss', 'T_full', ...
            'itertimes', 'startIdx', 'stopIdx', '-v7.3');

        fid = fopen(logfile, 'a');
        if fid ~= -1
            fprintf(fid, '==== c=%d chunk %d-%d done | %.1f min ====\n', ...
                c, startIdx, stopIdx, toc(tCStart)/60);
            fclose(fid);
        end
    end

    % --- Final save for this c ---
    finalFile = fullfile('ModelRuns', sprintf('FSS_CScan_Run%d_c%d_Final.mat', RunID, c));
    save(finalFile, ...
        'Res_O', 'Res_O2', 'Res_O4', ...
        'Res_Intra', 'Res_Intra2', 'Res_Intra4', ...
        'Res_Std', 'Res_Std2', 'Res_Std4', ...
        'a_vals', 'p_values', 'a_vec', 'p_vec', 'lambda_vec', ...
        'N', 'c', 'nGroup', 'c_values', 'N_values', 'k_avg', ...
        'numiter', 'Tss', 'T_full', ...
        'itertimes', '-v7.3');

    fprintf('  c=%d complete. Saved: %s (%.1f min)\n', c, finalFile, toc(tCStart)/60);
end

fprintf('\nAll group counts complete.\n');

%% =====================================================================
%  LOCAL FUNCTIONS
%  =====================================================================

function EL = SBM_edgelist_Physical(n, c, k_avg, a)
% Generates SBM graph from physical parameters (a, k_avg)
% a = self-mixing probability, k_avg = target average degree
    if mod(n, c) ~= 0, error('N must be divisible by c'); end
    n_block = n / c;

    p_in  = (a * k_avg) / (n_block - 1);
    p_out = ((1 - a) * k_avg) / (n - n_block);
    p_in  = min(p_in, 1);
    p_out = max(min(p_out, 1), 0);

    expE = 0.5 * n * k_avg;
    EL   = zeros(ceil(1.2 * expE), 2, 'uint32');
    k    = 0;
    starts = (0:c-1)' * n_block + 1;

    % Internal edges
    for g = 1:c
        EL_block = generate_block_edges(n_block, p_in);
        if ~isempty(EL_block)
            new_edges = EL_block + (starts(g) - 1);
            m = size(new_edges, 1);
            EL(k+1:k+m, :) = new_edges;
            k = k + m;
        end
    end

    % External edges
    if p_out > 0
        is_sparse = p_out < 0.05;
        for g1 = 1:c
            for g2 = g1+1:c
                if is_sparse
                    n_pairs = n_block^2;
                    n_edges = binornd(n_pairs, p_out);
                    if n_edges > 0
                        lin_idx = randperm(n_pairs, n_edges)';
                        [ii, jj] = ind2sub([n_block, n_block], lin_idx);
                        idx_range = k + (1:n_edges);
                        EL(idx_range, 1) = starts(g1) + ii - 1;
                        EL(idx_range, 2) = starts(g2) + jj - 1;
                        k = k + n_edges;
                    end
                else
                    R = rand(n_block, n_block) < p_out;
                    [ii, jj] = find(R);
                    m = numel(ii);
                    if m > 0
                        idx_range = k + (1:m);
                        EL(idx_range, 1) = starts(g1) + ii - 1;
                        EL(idx_range, 2) = starts(g2) + jj - 1;
                        k = k + m;
                    end
                end
            end
        end
    end
    EL = EL(1:k, :);
end

function edges = generate_block_edges(n, p)
    R = triu(rand(n, n) < p, 1);
    [ii, jj] = find(R);
    edges = [ii, jj];
end

function [O, O2, O4, Omod, Omod2, Omod4, Ostd, Ostd2, Ostd4] = ...
    RunBCS_Moments(EL, p, Tss, T_full, c, N)
% Runs BChS dynamics. Returns time-moments for finite-size scaling of all
% three order parameters:
%   O,  O2,  O4        = <|m|>,  <m^2>,  <m^4>       (global order)
%   Omod, Omod2, Omod4 = <I>, <I^2>, <I^4>,  I = mean_g |m_g|
%   Ostd, Ostd2, Ostd4 = <R>, <R^2>, <R^4>,  R = std(m_g)
    nEL    = size(EL, 1);
    invN   = 1 / N;
    invT   = 1 / T_full;
    nMod   = c;
    modLen = N / nMod;
    invModSize = 1 / modLen;
    mod_id = ceil((1:N) / modLen).';

    o0 = randi(3, N, 1) - 2;

    % ---- Burn-in ----
    randE = randi(nEL, 1, Tss);
    mu    = (rand(1, Tss) >= p) * 2 - 1;
    dir   = rand(1, Tss) < 0.5;

    for t = 1:Tss
        e = randE(t);
        if dir(t), ii = EL(e, 2); jj = EL(e, 1);
        else,      ii = EL(e, 1); jj = EL(e, 2); end
        new_val = o0(ii) + mu(t) * o0(jj);
        if new_val > 1, new_val = 1; elseif new_val < -1, new_val = -1; end
        o0(ii) = new_val;
    end

    % ---- Initialise running observables ----
    global_mean = mean(o0);
    mod_means   = accumarray(mod_id, o0, [nMod 1], @mean);
    sum_sq_mod  = sum(mod_means.^2);
    sum_abs_mod = sum(abs(mod_means));

    acc_O = 0;   acc_O2 = 0;   acc_O4 = 0;
    acc_I = 0;   acc_I2 = 0;   acc_I4 = 0;
    acc_R = 0;   acc_R2 = 0;   acc_R4 = 0;

    % ---- Measurement phase ----
    randE = randi(nEL, 1, T_full);
    mu    = (rand(1, T_full) >= p) * 2 - 1;
    dir   = rand(1, T_full) < 0.5;

    for t = 1:T_full
        e = randE(t);
        if dir(t), ii = EL(e, 2); jj = EL(e, 1);
        else,      ii = EL(e, 1); jj = EL(e, 2); end

        old_op = o0(ii);
        new_op = old_op + mu(t) * o0(jj);
        if new_op > 1, new_op = 1; elseif new_op < -1, new_op = -1; end

        if new_op ~= old_op
            o0(ii) = new_op;
            delta  = new_op - old_op;
            global_mean = global_mean + delta * invN;

            m_idx = mod_id(ii);
            old_m = mod_means(m_idx);
            new_m = old_m + delta * invModSize;
            mod_means(m_idx) = new_m;
            sum_sq_mod  = sum_sq_mod  - old_m^2    + new_m^2;
            sum_abs_mod = sum_abs_mod - abs(old_m) + abs(new_m);
        end

        % Global order parameter and its powers
        ThisO = abs(global_mean);
        acc_O  = acc_O  + ThisO;
        acc_O2 = acc_O2 + ThisO^2;
        acc_O4 = acc_O4 + ThisO^4;

        % Intra-group order parameter I = mean_g |m_g| and its powers
        ThisI = sum_abs_mod / nMod;
        acc_I  = acc_I  + ThisI;
        acc_I2 = acc_I2 + ThisI^2;
        acc_I4 = acc_I4 + ThisI^4;

        % Modular spread R = std(m_g) (population std across groups)
        var_m = (sum_sq_mod / nMod) - (global_mean^2);
        if var_m < 0, var_m = 0; end
        ThisR = sqrt(var_m);
        acc_R  = acc_R  + ThisR;
        acc_R2 = acc_R2 + ThisR^2;
        acc_R4 = acc_R4 + ThisR^4;
    end

    O     = acc_O  * invT;
    O2    = acc_O2 * invT;
    O4    = acc_O4 * invT;
    Omod  = acc_I  * invT;
    Omod2 = acc_I2 * invT;
    Omod4 = acc_I4 * invT;
    Ostd  = acc_R  * invT;
    Ostd2 = acc_R2 * invT;
    Ostd4 = acc_R4 * invT;
end
