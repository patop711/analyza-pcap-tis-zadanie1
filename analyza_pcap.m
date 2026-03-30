function analyza_pcap(pcap_file, only_task1)
%ANALYZA_PCAP  Analýza reálneho IP toku – TIS Zadanie 1
%
%  Autor  : Patrik  (6 písmen → mu = 6 p/s)
%
%  Použitie:
%    analyza_pcap                      % iba Task 1 (bez pcap)
%    analyza_pcap('subor.pcap')        % plná analýza
%    analyza_pcap('subor.pcap', true)  % iba Task 1
%
%  Požiadavky:
%    - MATLAB R2020a+ (exportgraphics)
%    - Statistics and Machine Learning Toolbox  (exprnd, poisspdf, prctile)
%    - Pre načítanie PCAP: 5G Toolbox (pcapReader)
%      Alternatíva: exportujte PCAP do CSV cez tshark a upravte load_pcap().

if nargin < 1, pcap_file  = '';    end
if nargin < 2, only_task1 = false; end

%% ── Globálne nastavenia ──────────────────────────────────────────────────
NAME       = 'Patrik';
MU         = double(numel(NAME));   % 6 p/s  (počet písmen v mene)
N_SIM      = 10000;                 % počet simulovaných hodnôt
CHOSEN_BIN = 1.0;                   % vybrané vzorkovanie [s]

OUTPUT_DIR = 'output';
FIG_DIR    = fullfile(OUTPUT_DIR, 'figures');   % .fig (MATLAB figure)
JPG_DIR    = fullfile(OUTPUT_DIR, 'jpg');        % .jpg
TXT_DIR    = fullfile(OUTPUT_DIR, 'data');        % .txt

% vytvor výstupné adresáre
for d = {OUTPUT_DIR, FIG_DIR, JPG_DIR, TXT_DIR}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end

rng(42);   % reprodukovateľnosť

%% ── Task 1 – vždy ───────────────────────────────────────────────────────
task1_poisson_simulation(MU, N_SIM, JPG_DIR, FIG_DIR, TXT_DIR);

%% ── Task 2 & 3 – iba ak máme PCAP ───────────────────────────────────────
if ~only_task1
    if isempty(pcap_file)
        fprintf('\nPoznámka: nebol zadaný .pcap súbor – beží iba Task 1.\n');
        fprintf('Použitie: analyza_pcap(''subor.pcap'')\n');
    else
        if ~isfile(pcap_file)
            error('Súbor ''%s'' neexistuje.', pcap_file);
        end
        [times, protos] = load_pcap(pcap_file);
        task2_full_analysis(times, protos, CHOSEN_BIN, JPG_DIR, FIG_DIR, TXT_DIR);
        task3_segments(times, protos, CHOSEN_BIN, JPG_DIR, FIG_DIR, TXT_DIR);
    end
end

create_zip(OUTPUT_DIR);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  TASK 1 – Simulácia Poissonovho toku
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function task1_poisson_simulation(MU, N_SIM, JPG_DIR, FIG_DIR, TXT_DIR)
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('TASK 1 – Simulácia Poissonovho toku  (λ = %d p/s, N = %d)\n', MU, N_SIM);
fprintf('%s\n', repmat('=', 1, 60));

% ── Obr. 1 – rovnomerné rozdelenie ────────────────────────────────────
uniform_vals = rand(N_SIM, 1);

fig1 = figure('Visible', 'off', 'Position', [0 0 1100 400]);
plot(uniform_vals, '.', 'MarkerSize', 2, 'Color', [0.25 0.41 0.88]);
title(sprintf('%d náhodných čísel rovnomerného rozdelenia  U(0, 1)', N_SIM));
xlabel('Index');  ylabel('Hodnota');
ylim([0 1]);  grid on;
save_fig(fig1, 'task1_01_uniform', JPG_DIR, FIG_DIR);
save_txt(uniform_vals, 'task1_01_uniform', 'uniform_U(0,1)', TXT_DIR);
close(fig1);

% ── Obr. 2 – exponenciálne rozdelenie (medzery medzi príchodmi) ───────
exp_gaps = exprnd(1.0 / MU, N_SIM, 1);

fig2 = figure('Visible', 'off', 'Position', [0 0 1100 400]);
plot(exp_gaps, '.', 'MarkerSize', 2, 'Color', [1.0 0.55 0.0]);
title(sprintf('%d náhodných čísel exponenciálneho rozdelenia  (λ = %d p/s,  stredná hodnota = %.4f s)', ...
    N_SIM, MU, 1/MU));
xlabel('Index');  ylabel('Medzera  [s]');
grid on;
save_fig(fig2, 'task1_02_exponential', JPG_DIR, FIG_DIR);
save_txt(exp_gaps, 'task1_02_exponential', 'exponential_inter-arrival_time_s', TXT_DIR);
close(fig2);

% ── Vzorkovanie na 1-sekundové intervaly ──────────────────────────────
arrival_times = cumsum(exp_gaps);
t_end         = arrival_times(end);
t_bins        = (0 : ceil(t_end))';          % hrany histogramu
counts_1s     = histcounts(arrival_times, t_bins)';
bin_centers   = 0.5 * (t_bins(1:end-1) + t_bins(2:end));

% ── Obr. 3 – vzorkovaný Poissonov proces (priebeh + počty/s) ──────────
n_show = min(500, numel(arrival_times));
show_s = min(200, numel(counts_1s));

fig3 = figure('Visible', 'off', 'Position', [0 0 1300 700]);

subplot(2, 1, 1);
stairs(arrival_times(1:n_show), (1:n_show)', 'Color', [0.27 0.51 0.71], 'LineWidth', 0.9);
title(sprintf('Vzorkovaný Poissonov proces – kumulatívne príchody (prvých %d)', n_show));
xlabel('Čas  [s]');  ylabel('Kum. počet príchodov');
grid on;

subplot(2, 1, 2);
bar(bin_centers(1:show_s), counts_1s(1:show_s), 1.0, ...
    'FaceColor', [0.18 0.55 0.34], 'EdgeColor', [0.0 0.39 0.0], ...
    'FaceAlpha', 0.75, 'LineWidth', 0.3);
hold on;
yline(MU, 'r--', 'LineWidth', 1.5);
hold off;
title(sprintf('Počet príchodov za 1 sekundu  (prvých %d s)', show_s));
xlabel('Čas  [s]');  ylabel('Počet príchodov / s');
legend(sprintf('E[k] = λ = %d', MU), 'Location', 'northeast');
grid on;

sgtitle('Simulácia Poissonovho toku – vzorkovanie na 1 s', 'FontSize', 13, 'FontWeight', 'bold');
save_fig(fig3, 'task1_03_poisson_sampled', JPG_DIR, FIG_DIR);
save_txt([bin_centers, counts_1s], 'task1_03_counts_per_second', ...
    'time_start_s;counts_per_1s', TXT_DIR);
close(fig3);

% ── Obr. 4 – porovnanie simulácia vs. teoretické Poissonovo rozd. ─────
k_max     = max(max(counts_1s), MU + 5);
k_vals    = (0 : k_max)';
sim_cnt   = histcounts(counts_1s, (-0.5 : k_max + 0.5))';
sim_prob  = sim_cnt / sum(sim_cnt);
theo_prob = poisspdf(k_vals, MU);

fig4 = figure('Visible', 'off', 'Position', [0 0 1200 500]);
w = 0.38;
hold on;
bar(k_vals - w/2, sim_prob,  w, 'FaceColor', [0.27 0.51 0.71], ...
    'EdgeColor', [0 0 0.5],  'FaceAlpha', 0.80, 'LineWidth', 0.4);
bar(k_vals + w/2, theo_prob, w, 'FaceColor', [1.0 0.50 0.31], ...
    'EdgeColor', [0.55 0 0], 'FaceAlpha', 0.80, 'LineWidth', 0.4);
hold off;
title(sprintf('Porovnanie simulácie a teoretického Poissonovho rozdelenia  (λ = %d, N = %d)', ...
    MU, N_SIM));
xlabel('k  (príchody za 1 s)');  ylabel('Pravdepodobnosť  P(X = k)');
legend('Simulácia', sprintf('Poisson(λ = %d)', MU), 'Location', 'northeast');
grid on;

save_fig(fig4, 'task1_04_poisson_comparison', JPG_DIR, FIG_DIR);
save_txt([k_vals, sim_prob, theo_prob], 'task1_04_poisson_comparison', ...
    'k;sim_probability;theo_probability', TXT_DIR);
close(fig4);

fprintf('  Simulačný čas: %.1f s  |  bins: %d\n', t_end, numel(counts_1s));
fprintf('  Task 1 – hotovo.\n\n');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Pomocné funkcie – medzery a tok
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function gaps = compute_gaps(times)
gaps  = diff(times);
n_neg = sum(gaps < 0);
if n_neg > 0
    fprintf('    Odstraňujem %d záporných medzier\n', n_neg);
end
gaps = gaps(gaps >= 0);
end


function [rate, pdf_fit] = exp_fit_curve(gaps, x)
mu_g = mean(gaps);
if mu_g > 0
    rate = 1.0 / mu_g;
else
    rate = 1.0;
end
pdf_fit = rate * exp(-rate * x);
end


function plot_gaps_3(gaps, prefix, title_str, JPG_DIR, FIG_DIR, TXT_DIR)
n_bins = min(120, max(20, floor(numel(gaps) / 80)));
x_99   = prctile(gaps, 99);
if isempty(x_99) || x_99 == 0, x_99 = 1.0; end
x_fit  = linspace(0, x_99, 300)';
[rate, pdf_fit] = exp_fit_curve(gaps, x_fit);

% ── gaps01 kombinovaný (priebeh + histogram) ───────────────────────────
fig = figure('Visible', 'off', 'Position', [0 0 1400 400]);

subplot(1, 2, 1);
plot(gaps, '.', 'MarkerSize', 1.5, 'Color', [0.27 0.51 0.71]);
title('Priebeh medzier (bez záporných)');
xlabel('Index paketu');  ylabel('Medzera  [s]');
grid on;

subplot(1, 2, 2);
histogram(gaps, n_bins, 'Normalization', 'pdf', 'BinLimits', [0, x_99], ...
    'FaceColor', [0.27 0.51 0.71], 'EdgeColor', [0 0 0.5], ...
    'FaceAlpha', 0.75, 'LineWidth', 0.3);
hold on;
plot(x_fit, pdf_fit, 'r-', 'LineWidth', 2.0);
hold off;
title('Histogram medzier');
xlabel('Medzera  [s]');  ylabel('Hustota pravdepodobnosti');
legend(sprintf('Exp fit  λ = %.3f p/s', rate), 'Location', 'northeast');
xlim([0, x_99]);  grid on;

sgtitle(title_str, 'FontSize', 13, 'FontWeight', 'bold');
save_fig(fig, sprintf('%s_gaps01_combined', prefix), JPG_DIR, FIG_DIR);
close(fig);

% ── gaps02 histogram standalone ────────────────────────────────────────
fig2 = figure('Visible', 'off', 'Position', [0 0 900 500]);
histogram(gaps, n_bins, 'Normalization', 'pdf', 'BinLimits', [0, x_99], ...
    'FaceColor', [0.27 0.51 0.71], 'EdgeColor', [0 0 0.5], ...
    'FaceAlpha', 0.75, 'LineWidth', 0.3);
hold on;
plot(x_fit, pdf_fit, 'r-', 'LineWidth', 2.0);
hold off;
title(sprintf('%s – histogram medzier', title_str));
xlabel('Medzera  [s]');  ylabel('Hustota pravdepodobnosti');
legend(sprintf('Exp fit  λ = %.3f p/s', rate), 'Location', 'northeast');
xlim([0, x_99]);  grid on;
save_fig(fig2, sprintf('%s_gaps02_histogram', prefix), JPG_DIR, FIG_DIR);
close(fig2);

% ── gaps03 priebeh standalone ──────────────────────────────────────────
fig3 = figure('Visible', 'off', 'Position', [0 0 1300 400]);
plot(gaps, '.', 'MarkerSize', 1.5, 'Color', [0.27 0.51 0.71]);
title(sprintf('%s – priebeh medzier medzi paketmi', title_str));
xlabel('Index paketu');  ylabel('Medzera  [s]');
grid on;
save_fig(fig3, sprintf('%s_gaps03_timeseries', prefix), JPG_DIR, FIG_DIR);
close(fig3);

save_txt(gaps, sprintf('%s_gaps', prefix), 'inter_packet_gap_s', TXT_DIR);
end


function [counts, centers] = sample_flow(times, bin_size)
t_start = times(1);
t_end   = times(end);
bins    = (t_start : bin_size : t_end + bin_size)';
counts  = histcounts(times, bins)';
centers = 0.5 * (bins(1:end-1) + bins(2:end));
end


function plot_flow_and_hist(counts, centers, bin_size, prefix, title_str, JPG_DIR, FIG_DIR, TXT_DIR)
mu_est    = mean(counts);
k_max     = max([max(counts), floor(mu_est) + 3, 1]);
k_vals    = (0 : k_max)';
sim_cnt   = histcounts(counts, (-0.5 : k_max + 0.5))';
sim_prob  = sim_cnt / max(sum(sim_cnt), 1);
theo_prob = poisspdf(k_vals, max(mu_est, 1e-6));
w = 0.38;

% ── kombinovaný (tok + histogram vedľa seba) ───────────────────────────
fig = figure('Visible', 'off', 'Position', [0 0 1600 500]);

subplot(1, 2, 1);
plot(centers, counts, '-', 'LineWidth', 0.8, 'Color', [0.27 0.51 0.71]);
hold on;
fill([centers(1); centers; centers(end)], [0; counts; 0], ...
    [0.27 0.51 0.71], 'FaceAlpha', 0.25, 'EdgeColor', 'none');
hold off;
title(sprintf('Tok paketov  (bin = %g s)', bin_size));
xlabel('Čas  [s]');  ylabel(sprintf('Pakety / %g s', bin_size));
grid on;

subplot(1, 2, 2);
hold on;
bar(k_vals - w/2, sim_prob,  w, 'FaceColor', [0.27 0.51 0.71], ...
    'EdgeColor', [0 0 0.5],  'FaceAlpha', 0.80, 'LineWidth', 0.4);
bar(k_vals + w/2, theo_prob, w, 'FaceColor', [1.0 0.50 0.31], ...
    'EdgeColor', [0.55 0 0], 'FaceAlpha', 0.80, 'LineWidth', 0.4);
hold off;
title(sprintf('Histogram vs. Poisson  (bin = %g s)', bin_size));
xlabel(sprintf('k  (pakety / %g s)', bin_size));  ylabel('Pravdepodobnosť');
legend('Data', sprintf('Poisson(λ = %.2f)', mu_est), 'Location', 'northeast');
grid on;

sgtitle(title_str, 'FontSize', 13, 'FontWeight', 'bold');
save_fig(fig, sprintf('%s_bin%g_combined', prefix, bin_size), JPG_DIR, FIG_DIR);
close(fig);

% ── tok standalone ─────────────────────────────────────────────────────
fig2 = figure('Visible', 'off', 'Position', [0 0 1300 400]);
plot(centers, counts, '-', 'LineWidth', 0.8, 'Color', [0.27 0.51 0.71]);
hold on;
fill([centers(1); centers; centers(end)], [0; counts; 0], ...
    [0.27 0.51 0.71], 'FaceAlpha', 0.25, 'EdgeColor', 'none');
hold off;
title(sprintf('%s – tok paketov  (bin = %g s)', title_str, bin_size));
xlabel('Čas  [s]');  ylabel(sprintf('Pakety / %g s', bin_size));
grid on;
save_fig(fig2, sprintf('%s_bin%g_flow', prefix, bin_size), JPG_DIR, FIG_DIR);
close(fig2);

% ── histogram standalone ───────────────────────────────────────────────
fig3 = figure('Visible', 'off', 'Position', [0 0 1000 500]);
hold on;
bar(k_vals - w/2, sim_prob,  w, 'FaceColor', [0.27 0.51 0.71], ...
    'EdgeColor', [0 0 0.5],  'FaceAlpha', 0.80, 'LineWidth', 0.4);
bar(k_vals + w/2, theo_prob, w, 'FaceColor', [1.0 0.50 0.31], ...
    'EdgeColor', [0.55 0 0], 'FaceAlpha', 0.80, 'LineWidth', 0.4);
hold off;
title(sprintf('%s – histogram vs. Poisson  (bin = %g s)', title_str, bin_size));
xlabel(sprintf('k  (pakety / %g s)', bin_size));  ylabel('Pravdepodobnosť');
legend('Data', sprintf('Poisson(λ = %.2f)', mu_est), 'Location', 'northeast');
grid on;
save_fig(fig3, sprintf('%s_bin%g_hist', prefix, bin_size), JPG_DIR, FIG_DIR);
close(fig3);

% dáta
save_txt([centers, counts], sprintf('%s_bin%g_flow', prefix, bin_size), ...
    sprintf('time_s;packets_per_%gs', bin_size), TXT_DIR);
save_txt([k_vals, sim_prob, theo_prob], sprintf('%s_bin%g_hist', prefix, bin_size), ...
    sprintf('k;sim_prob;poisson_prob(mu=%.4f)', mu_est), TXT_DIR);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  TASK 2 – Celý záznam
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function task2_full_analysis(times, protos, CHOSEN_BIN, JPG_DIR, FIG_DIR, TXT_DIR)
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('TASK 2 – Analýza celého záznamu\n');
fprintf('%s\n', repmat('=', 1, 60));

% ── medzery (3 obrázky) ────────────────────────────────────────────────
fprintf('\n  Medzery – všetky pakety\n');
gaps_all = compute_gaps(times);
plot_gaps_3(gaps_all, 'task2_all', 'Všetky pakety', JPG_DIR, FIG_DIR, TXT_DIR);

% ── 3 vzorkovania (6 obrázkov) ────────────────────────────────────────
bin_sizes = [0.01, 0.1, 1.0];
fprintf('\n  Tri vzorkovania: [%.2f  %.1f  %.1f] s\n', bin_sizes);
for i = 1:numel(bin_sizes)
    bs = bin_sizes(i);
    fprintf('    bin = %g s\n', bs);
    [counts, centers] = sample_flow(times, bs);
    plot_flow_and_hist(counts, centers, bs, 'task2_all_sampling', ...
        'Všetky pakety', JPG_DIR, FIG_DIR, TXT_DIR);
end

% ── zvolené vzorkovanie → TCP / UDP / iné (6 obrázkov) ────────────────
fprintf('\n  Zvolené vzorkovanie: %g s  →  TCP / UDP / iné\n', CHOSEN_BIN);
proto_labels = {'tcp',        'udp',        'other'};
proto_names  = {'TCP pakety', 'UDP pakety', 'Zvyšok (nie TCP/UDP)'};
for i = 1:3
    mask   = strcmp(protos, proto_labels{i});
    t_prot = times(mask);
    if numel(t_prot) < 2
        fprintf('    %s: príliš málo paketov, preskočím\n', upper(proto_labels{i}));
        continue;
    end
    fprintf('    %s: %d paketov\n', proto_names{i}, numel(t_prot));
    [counts, centers] = sample_flow(t_prot, CHOSEN_BIN);
    plot_flow_and_hist(counts, centers, CHOSEN_BIN, ...
        sprintf('task2_%s', proto_labels{i}), proto_names{i}, JPG_DIR, FIG_DIR, TXT_DIR);
end

fprintf('  Task 2 – hotovo.\n\n');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  TASK 3 – Zaujímavý a nudný úsek
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [mask_int, mask_bor, t_int, t_bor, window] = find_segments(times, window)
bins   = (times(1) : window : times(end) + window)';
counts = histcounts(times, bins)';
starts = bins(1:end-1);

[~, idx_int] = max(counts);
nonzero      = find(counts > 0);
[~, tmp]     = min(counts(nonzero));
idx_bor      = nonzero(tmp);

t_int = starts(idx_int);
t_bor = starts(idx_bor);

fprintf('    Najzaujímavejší: [%.1f – %.1f] s  (%d pkt)\n', ...
    t_int, t_int + window, counts(idx_int));
fprintf('    Najnudnejší   : [%.1f – %.1f] s  (%d pkt)\n', ...
    t_bor, t_bor + window, counts(idx_bor));

mask_int = times >= t_int & times < t_int + window;
mask_bor = times >= t_bor & times < t_bor + window;
end


function task3_segments(times, protos, CHOSEN_BIN, JPG_DIR, FIG_DIR, TXT_DIR)
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('TASK 3 – Najzaujímavejší a najnudnejší úsek\n');
fprintf('%s\n', repmat('=', 1, 60));

duration = times(end) - times(1);
window   = min(10.0, duration / 10.0);
fprintf('  Dĺžka záznamu: %.1f s  |  okno: %.1f s\n', duration, window);

[mask_int, mask_bor, t_int, t_bor, ~] = find_segments(times, window);

seg_names  = {'interesting',   'boring'};
seg_masks  = {mask_int,        mask_bor};
seg_titles = {sprintf('Najzaujímavejší úsek  (t ≈ %.0f s)', t_int), ...
              sprintf('Najnudnejší úsek      (t ≈ %.0f s)', t_bor)};

fig_overview = figure('Visible', 'off', 'Position', [0 0 1600 900]);
sgtitle('Porovnanie medzier – zaujímavý vs. nudný úsek', ...
    'FontSize', 13, 'FontWeight', 'bold');

for seg_idx = 1:2
    seg_name  = seg_names{seg_idx};
    seg_mask  = seg_masks{seg_idx};
    seg_title = seg_titles{seg_idx};
    t_seg     = times(seg_mask);
    p_seg     = protos(seg_mask);

    if numel(t_seg) < 2
        fprintf('  POZOR: %s – príliš málo paketov, preskočím\n', seg_name);
        continue;
    end

    gaps_seg = compute_gaps(t_seg);
    clr = [0.27 0.51 0.71];  % steelblue pre interesting
    if strcmp(seg_name, 'boring'), clr = [1 0.55 0.0]; end  % darkorange

    % ── prehľadový panel (súhrnný obrázok fig_overview) ─────────────
    subplot(2, 2, (seg_idx - 1) * 2 + 1);
    plot(gaps_seg, '.', 'MarkerSize', 2.0, 'Color', clr);
    title(sprintf('%s – priebeh medzier', seg_title));
    xlabel('Index');  ylabel('Medzera  [s]');
    grid on;

    x_99     = prctile(gaps_seg, 99);
    if isempty(x_99) || x_99 == 0, x_99 = 1.0; end
    n_bins_s = min(60, max(10, floor(numel(gaps_seg) / 10)));
    x_fit    = linspace(0, x_99, 300)';
    [rate_s, pdf_s] = exp_fit_curve(gaps_seg, x_fit);

    subplot(2, 2, (seg_idx - 1) * 2 + 2);
    histogram(gaps_seg, n_bins_s, 'Normalization', 'pdf', 'BinLimits', [0, x_99], ...
        'FaceColor', clr, 'EdgeColor', [0 0 0.5], 'FaceAlpha', 0.75, 'LineWidth', 0.3);
    hold on;
    plot(x_fit, pdf_s, 'r-', 'LineWidth', 1.8);
    hold off;
    title(sprintf('%s – histogram medzier', seg_title));
    xlabel('Medzera  [s]');  ylabel('Hustota');
    legend(sprintf('Exp  λ = %.3f', rate_s), 'Location', 'northeast', 'FontSize', 8);
    xlim([0, x_99]);  grid on;

    % ── individuálne grafy medzier úseku ─────────────────────────────
    plot_gaps_3(gaps_seg, sprintf('task3_%s', seg_name), seg_title, ...
        JPG_DIR, FIG_DIR, TXT_DIR);

    % ── 4 datasety × 2 obrázky ────────────────────────────────────────
    ds_names  = {'all',  'tcp',  'udp',  'other'};
    ds_titles = {sprintf('%s – Všetky pakety', seg_title), ...
                 sprintf('%s – TCP',          seg_title), ...
                 sprintf('%s – UDP',          seg_title), ...
                 sprintf('%s – Zvyšok',       seg_title)};

    fig_grid = figure('Visible', 'off', 'Position', [0 0 1600 2200]);
    sgtitle(sprintf('%s – tok paketov  (bin = %g s)', seg_title, CHOSEN_BIN), ...
        'FontSize', 13, 'FontWeight', 'bold');

    for ds_idx = 1:4
        if strcmp(ds_names{ds_idx}, 'all')
            ds_mask = true(numel(t_seg), 1);
        else
            ds_mask = strcmp(p_seg, ds_names{ds_idx});
        end
        t_ds     = t_seg(ds_mask);
        ds_title = ds_titles{ds_idx};

        ax_flow = subplot(4, 2, (ds_idx - 1) * 2 + 1);
        ax_hist = subplot(4, 2, (ds_idx - 1) * 2 + 2);

        if numel(t_ds) < 2
            axes(ax_flow); %#ok<LAXES>
            text(0.5, 0.5, sprintf('Nedostatok dát\n(%s)', ds_names{ds_idx}), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'Units', 'normalized', 'FontSize', 12);
            axes(ax_hist); %#ok<LAXES>
            text(0.5, 0.5, sprintf('Nedostatok dát\n(%s)', ds_names{ds_idx}), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'Units', 'normalized', 'FontSize', 12);
            continue;
        end

        t_ds_norm = t_ds - t_ds(1);   % normalizácia na začiatok segmentu
        [counts_g, centers_g] = sample_flow(t_ds_norm, CHOSEN_BIN);

        % tok
        axes(ax_flow); %#ok<LAXES>
        plot(centers_g, counts_g, '-', 'LineWidth', 0.8, 'Color', [0.27 0.51 0.71]);
        hold on;
        fill([centers_g(1); centers_g; centers_g(end)], [0; counts_g; 0], ...
            [0.27 0.51 0.71], 'FaceAlpha', 0.25, 'EdgeColor', 'none');
        hold off;
        title(sprintf('%s – tok', ds_title));
        xlabel('Čas  [s]');  ylabel(sprintf('Pkt / %g s', CHOSEN_BIN));
        grid on;

        % histogram vs Poisson
        mu_ds    = mean(counts_g);
        k_max_ds = max([max(counts_g), floor(mu_ds) + 3, 1]);
        kv       = (0 : k_max_ds)';
        sc       = histcounts(counts_g, (-0.5 : k_max_ds + 0.5))';
        sp       = sc / max(sum(sc), 1);
        tp       = poisspdf(kv, max(mu_ds, 1e-6));

        w = 0.38;
        axes(ax_hist); %#ok<LAXES>
        hold on;
        bar(kv - w/2, sp, w, 'FaceColor', [0.27 0.51 0.71], ...
            'EdgeColor', [0 0 0.5],  'FaceAlpha', 0.80, 'LineWidth', 0.4);
        bar(kv + w/2, tp, w, 'FaceColor', [1.0 0.50 0.31], ...
            'EdgeColor', [0.55 0 0], 'FaceAlpha', 0.80, 'LineWidth', 0.4);
        hold off;
        title(sprintf('%s – histogram', ds_title));
        xlabel('k');  ylabel('Pravd.');
        legend('Data', sprintf('Poisson(%.1f)', mu_ds), ...
            'Location', 'northeast', 'FontSize', 7);
        grid on;

        plot_flow_and_hist(counts_g, centers_g, CHOSEN_BIN, ...
            sprintf('task3_%s_%s', seg_name, ds_names{ds_idx}), ...
            ds_title, JPG_DIR, FIG_DIR, TXT_DIR);
    end

    save_fig(fig_grid, sprintf('task3_%s_grid4x2', seg_name), JPG_DIR, FIG_DIR);
    close(fig_grid);
end

save_fig(fig_overview, 'task3_overview_gaps_2x2', JPG_DIR, FIG_DIR);
close(fig_overview);

fprintf('  Task 3 – hotovo.\n\n');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Načítanie PCAP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [times, protos] = load_pcap(pcap_file)
%LOAD_PCAP  Načíta .pcap súbor pomocou pcapReader (5G Toolbox).
%
%  Podporuje iba legacy PCAP formát (nie PCAP-NG).
%  Ak je vstupný súbor PCAP-NG, funkcia sa pokúsi ho automaticky
%  skonvertovať pomocou nástroja editcap (súčasť Wireshark).
%
%  Manuálna konverzia PCAP-NG → PCAP:
%    editcap -F pcap vstup.pcapng vystup.pcap
%
%  Ak pcapReader nie je dostupný, exportujte PCAP do CSV cez tshark:
%    tshark -r subor.pcap -T fields -e frame.time_epoch -e ip.proto \
%           -E separator=, -E header=y > subor.csv
%  a nahraďte telo tejto funkcie za:
%    T = readtable('subor.csv');
%    times  = T.frame_time_epoch;
%    protos = repmat({'other'}, height(T), 1);
%    protos(T.ip_proto == 6)  = {'tcp'};
%    protos(T.ip_proto == 17) = {'udp'};

fprintf('\nNačítavam PCAP: %s\n', pcap_file);

if ~exist('pcapReader', 'class')
    error(['Trieda pcapReader nie je dostupná.\n' ...
           'Vyžaduje MATLAB 5G Toolbox.\n' ...
           'Alternatíva: exportujte PCAP do CSV cez tshark – pozri komentár v load_pcap().']);
end

% ── Detekcia PCAP-NG podľa magického čísla ────────────────────────────────
% PCAP-NG:    prvé 4 bajty = 0x0A 0D 0D 0A  (Section Header Block type)
% Legacy PCAP: 0xD4C3B2A1 alebo 0xA1B2C3D4 (little/big-endian magic)
pcap_file = convert_pcapng_if_needed(pcap_file);

reader   = pcapReader(pcap_file);
pkt_info = read(reader);          % vráti tabuľku: Timestamp, PacketData, ...
n        = height(pkt_info);
times    = zeros(n, 1);
protos   = repmat({'other'}, n, 1);

for i = 1:n
    times(i) = double(pkt_info.Timestamp(i));
    raw      = pkt_info.PacketData{i};
    % Ethernet II rámec: EtherType na bajtoch 13–14 (1-indexed)
    % IPv4 (0x0800): Protocol field na bajte 24 (14 B Eth + 9 B do IP proto)
    if numel(raw) >= 24
        ethertype = uint16(raw(13)) * 256 + uint16(raw(14));
        if ethertype == hex2dec('0800')   % IPv4
            ip_proto = raw(24);
            if ip_proto == 6
                protos{i} = 'tcp';
            elseif ip_proto == 17
                protos{i} = 'udp';
            end
        end
    end
end

% zoradiť podľa časovej pečiatky
[times, order] = sort(times);
protos = protos(order);

n_tcp   = sum(strcmp(protos, 'tcp'));
n_udp   = sum(strcmp(protos, 'udp'));
n_other = sum(strcmp(protos, 'other'));
fprintf('  Celkový počet paketov : %d\n', n);
fprintf('  TCP / UDP / iné       : %d / %d / %d\n', n_tcp, n_udp, n_other);
fprintf('  Trvanie záznamu       : %.3f s\n', times(end) - times(1));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  PCAP-NG konverzia
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out_file = convert_pcapng_if_needed(pcap_file)
%CONVERT_PCAPNG_IF_NEEDED  Ak je súbor PCAP-NG, skonvertuje ho na PCAP.
%
%  PCAP-NG začína magickými bajtmi 0x0A 0D 0D 0A (Section Header Block).
%  pcapReader podporuje iba legacy PCAP, preto je potrebná konverzia.
%  Konverzia sa vykonáva pomocou editcap (Wireshark).

out_file = pcap_file;   % predvolene vrátime originálny súbor

% Prečítaj prvé 4 bajty a skontroluj PCAP-NG magic
fid = fopen(pcap_file, 'rb');
if fid == -1
    error('Nepodarilo sa otvoriť súbor: %s', pcap_file);
end
magic = fread(fid, 4, 'uint8=>uint8')';
fclose(fid);

PCAPNG_MAGIC = uint8([0x0A, 0x0D, 0x0D, 0x0A]);
if ~isequal(magic, PCAPNG_MAGIC)
    return;   % nie je PCAP-NG, nič netreba robiť
end

fprintf('  Detekovaný formát PCAP-NG – pokúšam sa o konverziu na PCAP...\n');

% Vytvor cestu pre dočasný .pcap súbor
[dir_, name_] = fileparts(pcap_file);
tmp_file = fullfile(dir_, [name_ '_converted.pcap']);

% Pokús sa spustiť editcap (súčasť Wireshark)
cmd = sprintf('editcap -F pcap "%s" "%s"', pcap_file, tmp_file);
[status, output] = system(cmd);

if status == 0 && isfile(tmp_file)
    fprintf('  Konverzia úspešná: %s\n', tmp_file);
    out_file = tmp_file;
else
    error(['Súbor "%s" je vo formáte PCAP-NG, ktorý pcapReader nepodporuje.\n\n' ...
           'Konverzia pomocou editcap zlyhala (editcap nie je v PATH alebo nie je nainštalovaný).\n\n' ...
           'Vykonajte konverziu manuálne:\n' ...
           '  editcap -F pcap "%s" vystup.pcap\n\n' ...
           'Potom spustite:\n' ...
           '  analyza_pcap(''vystup.pcap'')\n\n' ...
           'editcap nájdete v balíku Wireshark: https://www.wireshark.org/\n' ...
           'Výstup príkazu: %s'], pcap_file, pcap_file, output);
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  ZIP archív
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function create_zip(OUTPUT_DIR)
zip_path = fullfile(OUTPUT_DIR, 'analyza_pcap_patrik.zip');
fprintf('\nVytvárám ZIP archív: %s\n', zip_path);

all_files = dir(fullfile(OUTPUT_DIR, '**', '*'));
all_files = all_files(~[all_files.isdir]);

file_list = {};
for i = 1:numel(all_files)
    fp = fullfile(all_files(i).folder, all_files(i).name);
    if ~endsWith(fp, '.zip')
        file_list{end+1} = fp; %#ok<AGROW>
    end
end

if ~isempty(file_list)
    zip(zip_path, file_list);
end
fprintf('  ZIP uložený: %s\n', zip_path);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Ukladanie figúr a dát
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function save_fig(fig, name, JPG_DIR, FIG_DIR)
% .fig – natívny MATLAB formát
fig_path = fullfile(FIG_DIR, [name '.fig']);
savefig(fig, fig_path);

% .jpg – 150 DPI  (exportgraphics vyžaduje R2020a+)
jpg_path = fullfile(JPG_DIR, [name '.jpg']);
exportgraphics(fig, jpg_path, 'Resolution', 150);
fprintf('  → %s\n', jpg_path);
end


function save_txt(data, name, header, TXT_DIR)
path = fullfile(TXT_DIR, [name '.txt']);
% zapíš hlavičku
fid = fopen(path, 'w');
fprintf(fid, '# %s\n', header);
fclose(fid);
% pripoj dáta (bodkočiarka ako oddeľovač, 8 platných číslic)
writematrix(data, path, 'Delimiter', ';', 'WriteMode', 'append', 'FileType', 'text');
fprintf('  → %s\n', path);
end
