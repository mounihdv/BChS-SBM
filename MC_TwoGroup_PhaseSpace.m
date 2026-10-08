function MC_TwoGroup_PhaseSpace(sector)
% Figure 3 MC scans consolidated from recovered MonteCarlo_Run5 and Run6.
% Usage: MC_TwoGroup_PhaseSpace('bipartite'), ('consensus'), or ('both').
% Default is 'both'; this starts the full, expensive recovered scans.
if nargin == 0, sector = 'both'; end
switch lower(char(sector))
    case 'bipartite', runIDs = 5;
    case 'consensus', runIDs = 6;
    case 'both', runIDs = [5, 6];
    otherwise, error('sector must be bipartite, consensus, or both.');
end
for id = runIDs
    run_sector(id);
end
end

function run_sector(RunID)
N = 10000;
k_avg = 20;
c = 2;
c_val = c;
if RunID == 5
    n_grid = 30;
    NumIter = 30;
    a_vals = linspace(0, 0.5, n_grid);
    p_vals = linspace(0.5, 1, n_grid);
else
    n_grid = 40;
    NumIter = 40;
    a_vals = linspace(0.745, 0.99, n_grid);
    p_vals = linspace(0, 0.5, n_grid);
end

% Time Steps
Tss     = 2e6;               
T_full  = 2e6;               

% Output Setup
outdir  = 'ModelRuns';
if ~exist(outdir, 'dir'), mkdir(outdir); end

logfile = fullfile(outdir, sprintf('Log_Run%d_C%d.txt', RunID, c_val));

% 2. PREPARE PARAMETER GRID
[A_GRID, P_GRID] = ndgrid(a_vals, p_vals);
a_vec  = A_GRID(:);
p_vec  = P_GRID(:);
nTotal = numel(a_vec);

% 3. CHECKPOINT RECOVERY SYSTEM
% Look for existing checkpoints to resume from
chkPattern = fullfile(outdir, sprintf('Run%d_C%d_Checkpoint_*.mat', RunID, c_val));
chkFiles   = dir(chkPattern);

if ~isempty(chkFiles)
    % Find latest checkpoint
    [~, latestIdx] = max([chkFiles.datenum]);
    latestFile = fullfile(chkFiles(latestIdx).folder, chkFiles(latestIdx).name);
    
    fprintf('Found checkpoint: %s\n', chkFiles(latestIdx).name);
    fprintf('Resuming simulation...\n');
    
    % Load previous progress
    load(latestFile, 'Res_O', 'Res_O2', 'Res_O4', 'Res_Intra', 'Res_Std', 'last_idx');
    start_idx = last_idx + 1;
    
    % Log resumption
    fid = fopen(logfile, 'a');
    fprintf(fid, '--- Resuming from Index %d at %s ---\n', start_idx, datestr(now));
    fclose(fid);
else
    % Start Fresh
    fprintf('No checkpoint found. Starting fresh.\n');
    Res_O     = zeros(nTotal, NumIter);
    Res_O2    = zeros(nTotal, NumIter); 
    Res_O4    = zeros(nTotal, NumIter); 
    Res_Intra = zeros(nTotal, NumIter);
    Res_Std   = zeros(nTotal, NumIter); 
    start_idx = 1;
    % Init Log
    fid = fopen(logfile, 'w');
    fprintf(fid, '--- Starting Run %d (C=%d) at %s ---\n', RunID, c_val, datestr(now));
    fprintf(fid, 'Total Points: %d, Iterations: %d\n', nTotal, NumIter);
    fclose(fid);
end

% 4. MAIN LOOP (Chunked for Safety)
tGlobalStart = tic;
ChunkSize = 20; % Save every 60 points (Adjusted for your small n_grid=100 run)

poolSize = 6;
pool = gcp('nocreate');
if isempty(pool), parpool('local', poolSize); end

% Loop over Chunks
for chunk_start = start_idx : ChunkSize : nTotal
    
    chunk_end = min(chunk_start + ChunkSize - 1, nTotal);
    fprintf('Processing Chunk: %d to %d of %d\n', chunk_start, chunk_end, nTotal);
    
    % Parallel Loop for this Chunk
    parfor idx = chunk_start : chunk_end
        a = a_vec(idx);
        p = p_vec(idx);
        
        % Local buffers
        loc_O  = zeros(1, NumIter);
        loc_O2 = zeros(1, NumIter);
        loc_O4 = zeros(1, NumIter);
        loc_I  = zeros(1, NumIter);
        loc_S  = zeros(1, NumIter);
        
        for kk = 1:NumIter
            % A. Generate SBM
            EL = SBM_edgelist_Physical(N, c, k_avg, a);
            
            % B. Run Dynamics 
            [O, O2, O4, Omod, Ostd] = RunBCS_Moments(EL, p, Tss, T_full, c, N);
            
            loc_O(kk)  = O;
            loc_O2(kk) = O2;
            loc_O4(kk) = O4;
            loc_I(kk)  = Omod;
            loc_S(kk)  = Ostd;
        end
        
        % Assign to Global Arrays
        Res_O(idx, :)     = loc_O;
        Res_O2(idx, :)    = loc_O2;
        Res_O4(idx, :)    = loc_O4;
        Res_Intra(idx, :) = loc_I;
        Res_Std(idx, :)   = loc_S;
    end
    
    % 5. SAVE CHECKPOINT (After every chunk)
    last_idx = chunk_end;
    ChkFile = fullfile(outdir, sprintf('Run%d_C%d_Checkpoint_Idx%d.mat', RunID, c_val, last_idx));
    
    save(ChkFile, ...
         'Res_O', 'Res_O2', 'Res_O4', 'Res_Intra', 'Res_Std', ...
         'a_vals', 'p_vals', 'c_val', 'a_vec', 'p_vec', ...
         'N', 'k_avg', 'last_idx', '-v7.3');
     
    % Log Progress
    fid = fopen(logfile, 'a');
    fprintf(fid, 'Finished Chunk %d-%d. Time Elapsed: %.1f min\n', ...
            chunk_start, chunk_end, toc(tGlobalStart)/60);
    fclose(fid);
    
    % Cleanup old checkpoints to save space (Optional)
    if chunk_start > ChunkSize
        prev_idx = chunk_start - 1;
        % You might need to adjust logic if ChunkSize changes, but this is basic cleanup
        OldFiles = dir(fullfile(outdir, sprintf('Run%d_C%d_Checkpoint_Idx*.mat', RunID, c_val)));
        for k=1:length(OldFiles)
            if ~contains(OldFiles(k).name, sprintf('Idx%d.mat', last_idx))
                delete(fullfile(OldFiles(k).folder, OldFiles(k).name));
            end
        end
    end
end

% 6. FINAL SAVE
FinalFile = fullfile(outdir, sprintf('Master_Run_%d_C%d_Moments.mat', RunID, c_val));
save(FinalFile, ...
     'Res_O', 'Res_O2', 'Res_O4', 'Res_Intra', 'Res_Std', ...
     'a_vals', 'p_vals', 'c_val', 'a_vec', 'p_vec',  ...
     'N', 'k_avg', '-v7.3');

fprintf('Done. Saved to %s\n', FinalFile);

% 7. PLOT RESULTS
try
    Plot_BCS_PhaseSpace(FinalFile);
catch
    warning('Could not auto-plot. Run Plot_BCS_PhaseSpace manually.');
end

end

%% --- LOCAL FUNCTIONS ---

function EL = SBM_edgelist_Physical(n, c, k_avg, a)
% Generates SBM graph from physical 'a' and 'k_avg'
    if mod(n, c) ~= 0, error('N must be divisible by c'); end
    n_block = n / c;
    p_in  = (a * k_avg) / (n_block - 1);
    p_out = ((1 - a) * k_avg) / (n - n_block);
    p_in = min(p_in, 1); 
    p_out = max(min(p_out, 1), 0);
    expE = 0.5 * n * k_avg;
    EL   = zeros(ceil(1.2 * expE), 2, 'uint32');
    k    = 0;
    starts = (0:c-1)' * n_block + 1;
    % Internal
    for g = 1:c
        EL_block = generate_block_edges(n_block, p_in);
        if ~isempty(EL_block)
            new_edges = EL_block + (starts(g) - 1);
            m = size(new_edges, 1);
            EL(k+1:k+m, :) = new_edges;
            k = k + m;
        end
    end
    % External
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
                        idx = k + (1:n_edges);
                        EL(idx, 1) = starts(g1) + ii - 1;
                        EL(idx, 2) = starts(g2) + jj - 1;
                        k = k + n_edges;
                    end
                else
                    R = rand(n_block, n_block) < p_out;
                    [ii, jj] = find(R);
                    m = numel(ii);
                    if m > 0
                        idx = k + (1:m);
                        EL(idx, 1) = starts(g1) + ii - 1;
                        EL(idx, 2) = starts(g2) + jj - 1;
                        k = k + m;
                    end
                end
            end
        end
    end
    EL = EL(1:k, :);
end

function edges = generate_block_edges(n, p)
    R = triu(rand(n,n) < p, 1);
    [ii, jj] = find(R);
    edges = [ii, jj];
end

function [O, O2, O4, Omod, Ostd] = RunBCS_Moments(EL, p, Tss, T_full, c, N)
% Runs BChS dynamics. Returns Moments (O, O2, O4) + Modular Metrics.
    nEL  = size(EL, 1);
    invN = 1 / N;
    invT = 1 / T_full;
    nMod   = c;                 
    modLen = N / nMod;          
    invModSize = 1 / modLen;
    mod_id = ceil((1:N) / modLen).'; 
    o0 = randi(3, N, 1) - 2; 
    % Burn-in
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
    % Observables
    global_mean = mean(o0);
    mod_means   = accumarray(mod_id, o0, [nMod 1], @mean);
    sum_sq_mod  = sum(mod_means.^2);
    acc_O = 0; acc_O2 = 0; acc_O4 = 0;
    acc_Mod = 0; acc_Std = 0;
    % Measurement
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
            m = mod_id(ii);
            old_m = mod_means(m);
            new_m = old_m + delta * invModSize;
            mod_means(m) = new_m;
            sum_sq_mod  = sum_sq_mod - old_m^2 + new_m^2;
        end
        % Metrics
        ThisO = abs(global_mean);
        acc_O   = acc_O + ThisO;
        acc_O2  = acc_O2 + ThisO^2;       
        acc_O4  = acc_O4 + ThisO^4;       
        acc_Mod = acc_Mod + mean(abs(mod_means));
        var_m = (sum_sq_mod / nMod) - (global_mean^2); 
        if var_m < 0, var_m = 0; end
        acc_Std = acc_Std + sqrt(var_m);
    end
    O    = acc_O * invT;
    O2   = acc_O2 * invT;
    O4   = acc_O4 * invT;
    Omod = acc_Mod * invT;
    Ostd = acc_Std * invT;
end