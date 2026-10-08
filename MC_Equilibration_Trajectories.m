%% BChS equilibration: five serial runs with overlapping trajectories.
% Self-contained: original SBM generator included below. No parallel pool.
% Each run uses a fresh network and random opinions, as in the original driver.
% One update = one edge draw and one attempted endpoint update.
% Curves are sampled instantaneous observables, NOT cumulative averages.

clear; clc;

%% Parameters
N           = 10000;
c           = 100;
p_in        = 0.9;
p_out       = 1e-3;
p           = 0.1;
nRuns       = 5;          % Change to 3 or 4 if desired
baseSeed    = 42;
T           = 5e6;
recordEvery = 1000;       % Record every 1000 attempted updates
Tss         = 2e6;        % Original transient cutoff (plot marker only)
Tmeasure    = 1e6;        % Original measurement interval

assert(mod(N,c) == 0, 'N must be divisible by c.');
assert(mod(T,recordEvery) == 0, 'T must be divisible by recordEvery.');

updates = (0:recordEvery:T).';
O       = zeros(numel(updates), nRuns);
O_intra = zeros(numel(updates), nRuns);
O_std   = zeros(numel(updates), nRuns); % Population spread of signed community means
seeds   = baseSeed + (0:nRuns-1);

%% Serial runs
for run = 1:nRuns
    rng(seeds(run), 'twister');
    fprintf('Run %d/%d: generating network...\n', run, nRuns);
    EL = SBM_edgelist_my(N, c, p_in, p_out);
    nEL = size(EL,1);
    assert(nEL > 0, 'The generated network has no edges.');
    s = randi(3, N, 1) - 2;  % Equal probabilities for -1, 0, +1

    mod_means = mean(reshape(s, N/c, c), 1);
    O(1,run)       = abs(mean(s));
    O_intra(1,run) = mean(abs(mod_means));
    O_std(1,run)   = std(mod_means, 1); % Normalization by c, not c-1

    % Same sampling and symmetric update rule as the supplied helper.
    randE = randi(nEL, 1, T);
    mu    = (rand(1, T) >= p) * 2 - 1;
    dir   = rand(1, T) < 0.5;

    % Continuous evolution: record the transient as well as later dynamics.
    for t = 1:T
        e = randE(t);
        if dir(t)
            ii = EL(e,2);
            jj = EL(e,1);
        else
            ii = EL(e,1);
            jj = EL(e,2);
        end

        s(ii) = max(-1, min(1, s(ii) + mu(t)*s(jj)));

        if mod(t, recordEvery) == 0
            k = t/recordEvery + 1;
            O(k,run) = abs(mean(s));
            % Each column holds one community of N/c consecutive agents.
            mod_means = mean(reshape(s, N/c, c), 1);
            O_intra(k,run) = mean(abs(mod_means));
            O_std(k,run)   = std(mod_means, 1);
        end

        if mod(t, 1e6) == 0
            fprintf('Run %d/%d: %.0f updates completed\n', run, nRuns, t);
        end
    end
    clear randE mu dir;
end

%% Save sampled trajectories in the current folder
save('BChS_Equilibration_Trajectories.mat', 'updates', 'O', 'O_intra', 'O_std', ...
    'N', 'c', 'p_in', 'p_out', 'p', 'nRuns', 'seeds', ...
    'T', 'recordEvery', 'Tss', 'Tmeasure');

%% Separate figures for global order, within-community order and modular spread
% Each figure overlays all runs. The same colour identifies a run in all three.
colors = lines(nRuns);
runLabels = arrayfun(@(r) sprintf('Run %d',r), 1:nRuns, ...
    'UniformOutput',false);

for observable = 1:3
    if observable == 1
        values = O;
        yLabelText = 'Global order, O';
        fileName = 'BChS_Equilibration_Global.png';
    elseif observable == 2
        values = O_intra;
        yLabelText = 'Within-community order, O_{intra}';
        fileName = 'BChS_Equilibration_Intra.png';
    else
        values = O_std;
        yLabelText = 'Community mean opinion spread, O_{std}';
        fileName = 'BChS_Equilibration_Std.png';
    end

    fig = figure('Color','w', 'Position',[150 150 1000 550]);
    hold on;
    % Shading marks the original measurement interval (2e6 to 3e6).
    patch([Tss Tss+Tmeasure Tss+Tmeasure Tss], [0 0 1 1], ...
        [0.9 0.9 0.9], 'EdgeColor','none', 'FaceAlpha',0.4, ...
        'HandleVisibility','off');
    hRun = gobjects(nRuns,1);
    for run = 1:nRuns
        hRun(run) = plot(updates, values(:,run), '-', ...
            'Color',colors(run,:), 'LineWidth',1.1);
    end
    xline(Tss, '--k', 'Transient cutoff', 'HandleVisibility','off');
    xlabel('Attempted updates');
    ylabel(yLabelText);
    legend(hRun, runLabels, 'Location','best');
    title(sprintf('N = %d, p = %.2f, p_{out} = %.3g', N, p, p_out));
    xlim([0 T]); ylim([0 1]);
    set(gca, 'FontSize',12, 'Layer','top');
    box on;
    print(fig, fileName, '-dpng', '-r300');
end
fprintf('Saved trajectories (.mat) and three separate figures (.png).\n');

%% Original network generator (unchanged)

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
