%% Packaged multigroup MC phase-space driver
% Corrected: uniform edge sampling with random endpoint selection.
% Defaults match the Figure 2 scan; older checkpoints are rejected.
% Helper functions are included at the end of this file.
if ~exist('Figures','dir'), mkdir('Figures'); end

Runid = 20;
N = 10000;
c = 100;
p_in = 0.9;

%rng(42);

numiter = 150;
nS      = 42;

ps      = linspace(0, 0.5, nS);
p_outs  = 10.^linspace(-3, -1, nS);

[PP, POUT] = ndgrid(ps, p_outs);
P_vec      = PP(:);
Pout_vec   = POUT(:);

nPairs = numel(P_vec);

Tss    = 2e6;
T_full = 2e6;

updateRule = 'uniform_edge_random_endpoint_v1';
simConfig = struct('N',N, 'c',c, 'p_in',p_in, ...
    'ps',ps, 'p_outs',p_outs, 'Tss',Tss, 'T_full',T_full, ...
    'numiter',numiter, 'updateRule',updateRule);

logfile = sprintf('progress_log_Run%d.txt',Runid);
outdir  = fullfile('ModelRuns', 'Checkpoints');

if ~exist(outdir,'dir'), mkdir(outdir); end

fnameSearch = sprintf('Run%d_all_upTo_*.mat',Runid);

% =========================
% 1) Try to resume from latest checkpoint
% =========================
checkpoint_files = dir(fullfile(outdir, fnameSearch));

if ~isempty(checkpoint_files)
    % Pick latest by datenum
    [~,ix]   = max([checkpoint_files.datenum]);
    lastfile = fullfile(checkpoint_files(ix).folder, checkpoint_files(ix).name);
    S        = load(lastfile);
    assert(isfield(S,'simConfig') && isequal(S.simConfig,simConfig), ...
        ['Checkpoint is from an older update rule or different settings. ' ...
         'Move incompatible Run%d checkpoints out of %s and start a fresh run.'], ...
        Runid, outdir);
    assert(isequal(size(S.O_tot),[nPairs,numiter]), ...
        'Checkpoint array dimensions do not match this scan.');
    assert(isscalar(S.stopIdx) && S.stopIdx == floor(S.stopIdx) && ...
        S.stopIdx >= 0 && S.stopIdx <= nPairs, 'Invalid checkpoint stopIdx.');

    % Restore arrays and metadata
    O_tot    = S.O_tot;
    O_tot_2  = S.O_tot_2;
    O_tot_3  = S.O_tot_3;
    O_tot_4  = S.O_tot_4;

    O_intra    = S.O_intra;
    O_intra_2  = S.O_intra_2;
    O_intra_3  = S.O_intra_3;
    O_intra_4  = S.O_intra_4;

    itertimes = S.itertimes;

    % Sanity checks on parameters
    assert(isequal(ps,     S.ps),     'ps mismatch with checkpoint.');
    assert(isequal(p_outs, S.p_outs), 'p_outs mismatch with checkpoint.');
    assert(nPairs == numel(S.P_vec),  'P_vec size mismatch.');

    % Continue from next index after previous stopIdx
    last_stopIdx = S.stopIdx;
    startIdx0    = last_stopIdx + 1;

    fprintf('Resuming from %s (last stopIdx = %d)\n', lastfile, last_stopIdx);

    % Append to log instead of deleting
    tGlobalStart = tic;      % new wall clock for this session
else
    % =========================
    % Fresh start (no checkpoint found)
    % =========================
    fprintf('No checkpoint found: starting from scratch.\n');

    if exist(logfile,'file'), delete(logfile); end

    O_tot    = zeros(nPairs, numiter);
    O_tot_2  = zeros(nPairs, numiter);
    O_tot_3  = zeros(nPairs, numiter);
    O_tot_4  = zeros(nPairs, numiter);

    O_intra    = zeros(nPairs, numiter);
    O_intra_2  = zeros(nPairs, numiter);
    O_intra_3  = zeros(nPairs, numiter);
    O_intra_4  = zeros(nPairs, numiter);

    itertimes = zeros(1, nPairs);

    startIdx0   = 1;         % start from the beginning
    tGlobalStart = tic;
end

chunkSize = 42;                 % checkpoint every 42 parameter pairs

pool = gcp('nocreate');
if isempty(pool), parpool('local',6); end

% =========================
% 2) Main loop, starting from startIdx0
% =========================
for startIdx = startIdx0:chunkSize:nPairs

    stopIdx = min(startIdx + chunkSize - 1, nPairs);

    parfor idx = startIdx:stopIdx
        tIdxStart = tic;

        p     = P_vec(idx);
        p_out = Pout_vec(idx);

        O_tot_loc    = zeros(1, numiter);
        O_tot_2_loc  = zeros(1, numiter);
        O_tot_3_loc  = zeros(1, numiter);
        O_tot_4_loc  = zeros(1, numiter);

        O_intra_loc    = zeros(1, numiter);
        O_intra_2_loc  = zeros(1, numiter);
        O_intra_3_loc  = zeros(1, numiter);
        O_intra_4_loc  = zeros(1, numiter);

        for kk = 1:numiter
            EL = SBM_edgelist_my(N, c, p_in, p_out);

            % You can swap in the incremental version here if desired:
            [O, O_mod] = RunBCS_SBM_optimized_inc(EL, p, Tss, T_full, N, c);
            %[O, O_mod] = RunBCS_SBM_optimized(EL, p, Tss, T_full);

            O_tot_loc(kk)    = O(1);
            O_tot_2_loc(kk)  = O(2);
            O_tot_3_loc(kk)  = O(3);
            O_tot_4_loc(kk)  = O(4);

            O_intra_loc(kk)   = O_mod(1);
            O_intra_2_loc(kk) = O_mod(2);
            O_intra_3_loc(kk) = O_mod(3);
            O_intra_4_loc(kk) = O_mod(4);
        end

        O_tot(idx,:)    = O_tot_loc;
        O_tot_2(idx,:)  = O_tot_2_loc;
        O_tot_3(idx,:)  = O_tot_3_loc;
        O_tot_4(idx,:)  = O_tot_4_loc;

        O_intra(idx,:)    = O_intra_loc;
        O_intra_2(idx,:)  = O_intra_2_loc;
        O_intra_3(idx,:)  = O_intra_3_loc;
        O_intra_4(idx,:)  = O_intra_4_loc;

        itertimes(idx) = toc(tIdxStart);

        % fine-grain log per idx
        fid = fopen(logfile,'a');
        if fid ~= -1
            fprintf(fid, 'idx=%d finished | p=%.4g, p_out=%.4g | t=%.1fmin | total=%.1fmin\n', ...
                idx, p, p_out, itertimes(idx)/60, toc(tGlobalStart)/60);
            fclose(fid);
        end
    end

    % checkpoint after each chunk
    chkFile = fullfile(outdir, sprintf('Run%d_all_upTo_%d.mat',Runid, stopIdx));
    save(chkFile, ...
        'O_tot','O_tot_2','O_tot_3','O_tot_4', ...
        'O_intra','O_intra_2','O_intra_3','O_intra_4', ...
        'ps','p_outs','P_vec','Pout_vec','Tss','T_full', ...
        'itertimes','nS','numiter','N','c','p_in','updateRule','simConfig', ...
        'startIdx','stopIdx','-v7.3');

    % coarse log per chunk
    fid = fopen(logfile,'a');
    if fid ~= -1
        fprintf(fid, '==== chunk %d-%d finished | total=%.1fmin ====\n', ...
            startIdx, stopIdx, toc(tGlobalStart)/60);
        fclose(fid);
    end
end

fulfname = sprintf('Run%d_all_final.mat',Runid);
% final save
finalfile = fullfile('ModelRuns',fulfname);
save(finalfile, ...
    'O_tot','O_tot_2','O_tot_3','O_tot_4', ...
    'O_intra','O_intra_2','O_intra_3','O_intra_4', ...
    'ps','p_outs','P_vec','Pout_vec','Tss','T_full', ...
    'itertimes','nS','numiter','N','c','p_in','updateRule','simConfig','-v7.3');

O_tot_grid     = reshape(O_tot,     [nS, nS, numiter]);
O_tot_2_grid   = reshape(O_tot_2,   [nS, nS, numiter]);
O_tot_3_grid   = reshape(O_tot_3,   [nS, nS, numiter]);
O_tot_4_grid   = reshape(O_tot_4,   [nS, nS, numiter]);

O_intra_grid   = reshape(O_intra,   [nS, nS, numiter]);
O_intra_2_grid = reshape(O_intra_2, [nS, nS, numiter]);
O_intra_3_grid = reshape(O_intra_3, [nS, nS, numiter]);
O_intra_4_grid = reshape(O_intra_4, [nS, nS, numiter]);




O_tot_thisRun = mean(O_tot_grid,3);
O_intra_thisRun = mean(O_intra_grid,3);

% nS = 45;

figure()
imagesc(O_tot_thisRun)
xticks(1:nS)
xticklabels(num2str(p_outs'))
yticks(1:nS)
yticklabels(num2str(ps'))
xlabel('p_out')
ylabel('p')
title('O')
set(gcf,'Position',[257.0000   49.8000  870.4000  732.8000])
colormap(flipud(jet)); clim([0 1]); colorbar
exportgraphics(gcf,sprintf('Figures/PhasePlot_Otot_Run%d.png',Runid))

figure()
imagesc(O_intra_thisRun)
xticks(1:nS)
xticklabels(num2str(p_outs'))
yticks(1:nS)
yticklabels(num2str(ps'))
xlabel('p_out')
ylabel('p')
title('O-intragroup')
set(gcf,'Position',[257.0000   49.8000  870.4000  732.8000])
colormap(flipud(jet)); clim([0 1]); colorbar
exportgraphics(gcf,sprintf('Figures/PhasePlot_Ointra_Run%d.png',Runid))


%% Recovered helper: SBM_edgelist_my.m

function EL = SBM_edgelist_my(n, c, p_in, p_out)
% Generate undirected SBM edge list with equal-sized blocks (no adjacency).
% n     - total number of nodes
% c     - number of clusters/modules
% p_in  - within-module edge probability
% p_out - between-module edge probability
%
% Returns:
%   EL  - (M x 2) edge list with i < j

    % ---- basic checks and block layout ----
    assert(c >= 1 && c == floor(c) && mod(n,c) == 0, ...
        'Use a positive integer c with n divisible by c.');
    assert(p_in >= 0 && p_in <= 1 && p_out >= 0 && p_out <= 1, ...
        'Edge probabilities must lie in [0,1].');
    b = n / c;

    starts = (0:c-1)' * b + 1;   % block starts: 1, b+1, 2b+1, ...

    % ---- rough expected edges for preallocation ----
    % average probability per pair for equal blocks
    p_eff = (p_in + (c-1)*p_out) / c;
    expE  = 0.5 * n * (n-1) * p_eff;
    EL    = zeros(ceil(1.1*expE), 2, 'uint32');
    k     = 0;

    % ---- within-block edges ----
    for a = 1:c
        ia = starts(a) : starts(a)+b-1;
        R = rand(b,b) < p_in;   % Bernoulli matrix
        R = triu(R,1);          % no self-loops, i<j
        [ii,jj] = find(R);      % local indices in this block
        m = numel(ii);
        if m > 0
            idx = k + (1:m);
            EL(idx,1) = ia(ii);
            EL(idx,2) = ia(jj);
            k = k + m;
        end
    end

    % ---- between-block edges ----
    for a = 1:c
        ia = starts(a) : starts(a)+b-1;
        for bb = a+1:c
            ib = starts(bb) : starts(bb)+b-1;
            R = rand(b,b) < p_out;
            [ii,jj] = find(R);
            m = numel(ii);
            if m > 0
                idx = k + (1:m);
                EL(idx,1) = ia(ii);
                EL(idx,2) = ib(jj);
                k = k + m;
            end
        end
    end

    % ---- trim edge list ----
    EL = EL(1:k,:);
end


%% Recovered helper: RunBCS_SBM_optimized_inc.m
function [O, Omod] = RunBCS_SBM_optimized_inc(EL, p, Tss, T_full, N, c)

    % Initial opinions
    assert(mod(N,c) == 0, 'N must be divisible by c.');
    o0  = randi(3, N, 1) - 2;
    nEL = size(EL,1);
    assert(nEL > 0, 'The generated network has no edges.');
    invN = 1 / N;
    invT = 1 / T_full;

    % Equal-sized groups of consecutive node indices.
    nMod = c;
    mod_id = ceil((1:N) / (N/c)).';
    invModSize = 1 / (N / nMod);

    % ---------- Burn-in (transient) ----------
    randE_burnin = randi(nEL, 1, Tss);
    mu_burnin    = (rand(1, Tss) >= p) * 2 - 1;
    dir_burnin   = rand(1, Tss) < 0.5;

    for t = 1:Tss
        e  = randE_burnin(t);
        if dir_burnin(t)
            ii = EL(e,2); jj = EL(e,1);
        else
            ii = EL(e,1); jj = EL(e,2);
        end

        old = o0(ii);
        new = old + mu_burnin(t) * o0(jj);
        new = max(-1, min(1, new));
        o0(ii) = new;
    end

    clear randE_burnin mu_burnin dir_burnin;

    % ---------- Initialize observables ----------
    % Global and modular means at t = 0 (just after burn-in)
    global_mean = mean(o0);
    mod_means   = accumarray(mod_id, o0, [nMod 1], @mean);

    Os = 0;  O2s = 0;  O3s = 0;  O4s = 0;
    O_mods = 0;  O_mods_2 = 0;  O_mods_3 = 0;  O_mods_4 = 0;

    % ---------- Main measurement loop ----------
    randE_main = randi(nEL, 1, T_full);
    mu_main    = (rand(1, T_full) >= p) * 2 - 1;
    dir_main   = rand(1, T_full) < 0.5;

    for t = 1:T_full
        e  = randE_main(t);
        if dir_main(t)
            ii = EL(e,2); jj = EL(e,1);
        else
            ii = EL(e,1); jj = EL(e,2);
        end

        old = o0(ii);
        new = old + mu_main(t) * o0(jj);
        new = max(-1, min(1, new));

        if new ~= old
            o0(ii) = new;

            % Incremental global mean
            delta = new - old;
            global_mean = global_mean + delta * invN;

            % Incremental module mean
            m = mod_id(ii);
            mod_means(m) = mod_means(m) + delta * invModSize;
        end

        ThisO     = abs(global_mean);
        ThisO_mod = mean(abs(mod_means));

        Os   = Os   + ThisO;
        O2s  = O2s  + ThisO^2;
        O3s  = O3s  + ThisO^3;
        O4s  = O4s  + ThisO^4;

        O_mods   = O_mods   + ThisO_mod;
        O_mods_2 = O_mods_2 + ThisO_mod^2;
        O_mods_3 = O_mods_3 + ThisO_mod^3;
        O_mods_4 = O_mods_4 + ThisO_mod^4;
    end

    % ---------- Averages over time ----------
    O       = [Os,      O2s,      O3s,      O4s     ] * invT;
    Omod    = [O_mods,  O_mods_2, O_mods_3, O_mods_4] * invT;
end

