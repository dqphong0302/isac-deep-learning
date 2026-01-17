% compare_dl_vs_gsfdic_accurate.m
% -------------------------------------------------------------------------
% ACCURATE COMPARISON matching ISAC_anh_Phong.m logic exactly:
%   (A) GSFDIC with fb=4 (exact original algorithm)
%   (B) 1-pass + Deep Learning
%
% Key differences from simplified version:
%   - Uses CFAR target detection (cfar_ofdm_radar)
%   - H_comp calculation for fb > 1
%   - Full communication signal reconstruction
%   - RLS estimation within feedback loop
% -------------------------------------------------------------------------

clc; clear; close all;

addpath('./useful_function/');

%% Python setup
pe = pyenv;
fprintf("MATLAB is using Python: %s\n", string(pe.Executable));

%% Settings
onnxFile = "hcom1d_fixed.onnx";

if ~isfile(onnxFile)
    error('ONNX file not found: %s', onnxFile);
end

Nmc = 100;       % Monte Carlo trials
no_of_fb = 4;    % Feedback loops (same as original)

%% System params (same as ISAC_anh_Phong.m)
global Nc Ns fC c Delta_f Ts Te;

c = 3e8;
Delta_f = 78.125e3;
Nc = 1024;
Ns = 512;
fC = 5e9;
Te = 1/Delta_f;
Ts = Te;
delta_r = 1.1574;
r_max = 2500;
v_max = 75;

expandVector = @(v, m) v(:, ones(1, m));

n = 0:Nc-1; k = n';
DFT_matrix = 1/sqrt(Nc) * exp(-2i*pi*k*n/Nc);

%% Communication params
hcom_length = 4;
N_pilot = 2*Nc;  % 2048 pilots (same as original)

% Modulators
M = 2;
pskmodObj = comm.PSKModulator(M, 'BitInput', true);
pskdemodObj = comm.PSKDemodulator(M, 'BitOutput', true);

% Feature config for DL
B = 64;
percentiles = [10 25 50 75 90];

%% Python for DL inference
try
    pymod = py.importlib.import_module("onnx_inference_helper");
    py.importlib.reload(pymod);
catch ME
    error("Cannot import onnx_inference_helper.py: %s", ME.message);
end
np = py.importlib.import_module("numpy");

%% Storage
nmse_gsfdic = zeros(Nmc, 1);
nmse_dl = zeros(Nmc, 1);

% Per-tap MSE storage (8 = 4 real + 4 imag)
mse_per_tap_gsfdic = zeros(Nmc, 8);
mse_per_tap_dl = zeros(Nmc, 8);

% Store channel estimates for analysis
hcom_true_all = zeros(Nmc, 8);
hcom_gsfdic_all = zeros(Nmc, 8);
hcom_dl_all = zeros(Nmc, 8);

%% Monte Carlo
fprintf('\n========================================\n');
fprintf('Running %d Monte Carlo trials...\n', Nmc);
fprintf('Baseline: GSFDIC fb=%d (exact ISAC_anh_Phong.m logic)\n', no_of_fb);
fprintf('========================================\n\n');
tic;

for it = 1:Nmc
    if mod(it, 20) == 0 || it == 1
        fprintf("Trial %d / %d\n", it, Nmc);
    end

    %% ===== Signal Generation (same as ISAC_anh_Phong.m) =====
    
    % Radar modulation
    bitsPerSymbol = 1;
    qam = 2^bitsPerSymbol;
    data = randi([0 qam-1], Nc, Ns);
    Ftx_radar = qammod(data, qam, 'UnitAveragePower', true);
    
    % Environment Noise (random 0-20 dB)
    noise_power = randi([0 20], 1, 1);
    
    % SI: 16 objects (static + slow-moving)
    numSI = randi([5 16], 1, 1);
    si_power = -35 + 30*rand(1, numSI);
    si_velocity = [zeros(1, floor(numSI/2)), (2*rand(1, ceil(numSI/2))-1)*2];
    si_range = randi([1 100], 1, numSI) * delta_r;
    
    % Targets: 1-5
    k_target = randi([1 5], 1, 1);
    target_power = -25 + 20*rand(1, k_target);
    target_velocity = (2*rand(1, k_target)-1) * v_max;
    target_range = (5 + randi([0 floor(r_max/delta_r)-5], 1, k_target)) * delta_r;
    
    % Generate echoes
    Y_target = generate_target_echo(target_power, target_velocity, target_range, Ftx_radar);
    Y_si = generate_target_echo(si_power, si_velocity, si_range, Ftx_radar);
    R_noise = generate_noise(noise_power);
    
    % Communication signal (same structure as original)
    Ftx_com = zeros(Nc, Ns);
    Mat_x_com_bit = zeros(log2(M) * Nc, Ns);
    for col = 1:Ns
        x_com_bit = randi([0 1], 1, log2(M) * Nc);
        x_com = pskmodObj(x_com_bit');
        Ftx_com(:, col) = x_com;
        Mat_x_com_bit(:, col) = x_com_bit';
    end
    x_com_td = DFT_matrix' * Ftx_com;
    x_com_td_vec = reshape(x_com_td, [], 1);
    Ftx_com_vec = reshape(Ftx_com, [], 1);
    
    % Communication power
    com_power = randi([20 30], 1, 1);
    com_power_sqrt = sqrt(10^(com_power/10));
    
    % Communication channel (FIXED PHASE - EXACTLY as ISAC_anh_Phong.m line 106)
    % hcom = db2mag([0 -9.7 -19.2 -22.8]).*exp(1j*[0 -.8 1.6 -2.6]).*com_power_sqrt
    hcom_true = db2mag([0 -9.7 -19.2 -22.8]).' .* exp(1j*[0 -0.8 1.6 -2.6].') .* com_power_sqrt;
    
    % Apply channel
    y_com_td = filter(hcom_true, 1, x_com_td_vec);
    Y_com = reshape(y_com_td, Nc, Ns);
    
    % Received signal
    A = DFT_matrix * (Y_target + Y_si + Y_com) + R_noise;
    
    %% ===== METHOD A: GSFDIC (exact ISAC_anh_Phong.m logic) =====
    
    % Initialize (same as original lines 114-118)
    hcom_est = zeros(hcom_length, 1);
    epsilon = 0.0001;
    P1 = (epsilon^-1) * eye(hcom_length);
    Ftx_com_est = zeros(Nc, Ns);
    Y_target_rec = zeros(Nc, Ns);
    Y_com_rec = zeros(Nc, Ns);
    H_comp_mat = zeros(Nc, Ns);
    Mat_x_com_bit_est = zeros(log2(M) * Nc, Ns);
    
    for fb = 1:no_of_fb
        % Step 1: Target and Communication Removal (lines 121-131)
        Y_target_rec_prev = Y_target_rec;
        Y_com_rec_prev = Y_com_rec;
        A_tilde = A - Y_com_rec_prev - Y_target_rec_prev;
        
        if fb > 1
            H_comp = -1/(Ns-1) * ((A_tilde(:,1) ./ Ftx_radar(:,1)) - (A_tilde(:,end) ./ Ftx_radar(:,end)));
            H_comp_mat = expandVector(H_comp, Ns);
        else
            H_comp_mat = zeros(Nc, Ns);
        end
        
        % Step 2: SI Suppression with FDIC (line 134)
        E_si_suppress_tilde = fdic(A_tilde, Ftx_radar, H_comp_mat);
        
        % Step 3: Target Compensation and Estimation (lines 137-143)
        E_radar_comp_gifdic = E_si_suppress_tilde + Y_target_rec_prev;
        RDM = calc_periodogram(E_radar_comp_gifdic, Ftx_radar, 1/100, 1/100);
        targetout = cfar_ofdm_radar(RDM, k_target, 1, 3, 3, 3, 3);
        target_rec_range = targetout.ranges;
        target_rec_velocity = targetout.velocities;
        Y_target_rec = generate_target_echo(target_power, target_rec_velocity, target_rec_range, Ftx_radar);
        Y_target_rec = DFT_matrix * Y_target_rec;
        
        % E_tilde (line 146)
        E_tilde = E_si_suppress_tilde + Y_target_rec_prev - Y_target_rec + Y_com_rec_prev;
        
        % Step 4: Communication Estimation with RLS (lines 150-169)
        x_com_pilot = x_com_td_vec(1:N_pilot);
        E_td = DFT_matrix' * E_tilde;
        E_td_vec = reshape(E_td, [], 1);
        Ep_td = E_td_vec(1:N_pilot);
        
        for m = hcom_length:length(x_com_pilot)
            [hcom_est, P1] = RLS_function(hcom_est, Ep_td(m), x_com_pilot(m:-1:m-(hcom_length-1)), P1);
        end
        
        % Communication Signal Reconstruction (lines 171-190)
        Ec_td = reshape(DFT_matrix' * E_tilde, [], 1);
        Ec = filter(1, hcom_est', Ec_td);
        Ec = DFT_matrix * reshape(Ec, Nc, Ns);
        
        for kk = 1:Ns
            Mat_x_com_bit_est(:, kk) = pskdemodObj(Ec(:, kk));
            x_com_est_vec = pskmodObj(Mat_x_com_bit_est(:, kk));
            Ftx_com_est(:, kk) = x_com_est_vec;
        end
        Ftx_com_est_vec = reshape(Ftx_com_est, [], 1);
        Ftx_com_est_vec(1:N_pilot) = Ftx_com_vec(1:N_pilot);
        Ftx_com_est = reshape(Ftx_com_est_vec, Nc, Ns);
        
        x_com_est = reshape(DFT_matrix' * Ftx_com_est, [], 1);
        Y_com_rec = filter(hcom_est', 1, x_com_est);
        Y_com_rec = DFT_matrix * reshape(Y_com_rec, Nc, Ns);
    end
    
    % GSFDIC results
    hcom_true_8 = [real(hcom_true); imag(hcom_true)];
    hcom_gsfdic_8 = [real(hcom_est); imag(hcom_est)];
    
    nmse_gsfdic(it) = 10*log10(norm(hcom_est - hcom_true)^2 / (norm(hcom_true)^2 + eps));
    mse_per_tap_gsfdic(it, :) = (hcom_gsfdic_8 - hcom_true_8).^2;
    
    hcom_true_all(it, :) = hcom_true_8';
    hcom_gsfdic_all(it, :) = hcom_gsfdic_8';
    
    %% ===== METHOD B: 1-pass + Deep Learning =====
    
    % One-pass FDIC (no feedback)
    H_comp_mat_dl = zeros(Nc, Ns);
    E_dl = fdic(A, Ftx_radar, H_comp_mat_dl);
    
    % LS estimate from pilots
    e_td_dl = DFT_matrix' * E_dl;
    x_pilot_td = x_com_td(:, 1:8);  % 8 pilot symbols
    y_pilot_td = e_td_dl(:, 1:8);
    h_ls = ls_fir_from_pilot(reshape(x_pilot_td,[],1), reshape(y_pilot_td,[],1), hcom_length);
    
    % Feature extraction
    norm_factor = sqrt(mean(abs(E_dl(:)).^2)) + eps;
    E_n = E_dl ./ norm_factor;
    P_sc = mean(abs(E_n).^2, 2);
    logP = log10(P_sc + 1e-12);
    
    bandSize = Nc / B;
    band_mean = zeros(B,1);
    band_std = zeros(B,1);
    band_max = zeros(B,1);
    for b = 1:B
        s = (b-1)*bandSize + 1;
        t = b*bandSize;
        v = logP(s:t);
        band_mean(b) = mean(v);
        band_std(b) = std(v);
        band_max(b) = max(v);
    end
    
    g_mean = mean(logP); g_std = std(logP);
    g_max = max(logP); g_min = min(logP);
    g_p = prctile(logP, percentiles);
    g = [g_mean; g_std; g_max; g_min; g_p(:)];
    
    feat201 = single([band_mean; band_std; band_max; g]);
    nf_feat = single(log10(norm_factor));
    hrls8 = single([real(h_ls(:)); imag(h_ls(:))]);
    x_vec = single([feat201; nf_feat; hrls8]);
    
    % ONNX inference
    x_py = np.array(x_vec.', pyargs('dtype', np.float32));
    y_list = pymod.predict_hcom(x_py, char(onnxFile));
    yhat8 = double(cell2mat(cell(y_list)));
    yhat8 = yhat8(:);
    
    hcom_dl = yhat8(1:4) + 1j*yhat8(5:8);
    nmse_dl(it) = 10*log10(norm(hcom_dl - hcom_true)^2 / (norm(hcom_true)^2 + eps));
    mse_per_tap_dl(it, :) = (yhat8 - hcom_true_8).^2;
    hcom_dl_all(it, :) = yhat8';
end

elapsed = toc;
fprintf('\nCompleted %d trials in %.1f seconds\n', Nmc, elapsed);

%% Results
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════════╗\n');
fprintf('║              ACCURATE COMPARISON (GSFDIC exact logic)            ║\n');
fprintf('╠══════════════════════════════════════════════════════════════════╣\n');
fprintf('║                        NMSE (dB) Statistics                       ║\n');
fprintf('╠═══════════════════╦══════════════════╦═══════════════════════════╣\n');
fprintf('║     Metric        ║  GSFDIC fb=4     ║   1-pass + Deep Learning  ║\n');
fprintf('╠═══════════════════╬══════════════════╬═══════════════════════════╣\n');
fprintf('║  Mean             ║    %+8.2f dB    ║       %+8.2f dB          ║\n', mean(nmse_gsfdic), mean(nmse_dl));
fprintf('║  Median           ║    %+8.2f dB    ║       %+8.2f dB          ║\n', median(nmse_gsfdic), median(nmse_dl));
fprintf('║  Std Dev          ║    %8.2f dB    ║       %8.2f dB          ║\n', std(nmse_gsfdic), std(nmse_dl));
fprintf('║  Min (Best)       ║    %+8.2f dB    ║       %+8.2f dB          ║\n', min(nmse_gsfdic), min(nmse_dl));
fprintf('║  Max (Worst)      ║    %+8.2f dB    ║       %+8.2f dB          ║\n', max(nmse_gsfdic), max(nmse_dl));
fprintf('╚═══════════════════╩══════════════════╩═══════════════════════════╝\n');

% Linear NMSE (for easier interpretation)
nmse_linear_gsfdic = 10.^(nmse_gsfdic/10);
nmse_linear_dl = 10.^(nmse_dl/10);

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════════╗\n');
fprintf('║                    LINEAR ACCURACY METRICS                        ║\n');
fprintf('╠══════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Mean Linear NMSE (GSFDIC):    %.6f (Error = %.4f%% of signal)  ║\n', mean(nmse_linear_gsfdic), mean(nmse_linear_gsfdic)*100);
fprintf('║  Mean Linear NMSE (DL):        %.6f (Error = %.4f%% of signal)  ║\n', mean(nmse_linear_dl), mean(nmse_linear_dl)*100);
fprintf('║  Accuracy Improvement:         %.2fx better                      ║\n', mean(nmse_linear_gsfdic)/mean(nmse_linear_dl));
fprintf('╚══════════════════════════════════════════════════════════════════╝\n');

% Per-tap accuracy
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════════╗\n');
fprintf('║                    PER-TAP MSE COMPARISON                         ║\n');
fprintf('╠════════════╦════════════════╦═════════════════╦═════════════════╣\n');
fprintf('║    Tap     ║   GSFDIC MSE   ║     DL MSE      ║   DL Better by  ║\n');
fprintf('╠════════════╬════════════════╬═════════════════╬═════════════════╣\n');
tap_labels = {'Re(h1)', 'Re(h2)', 'Re(h3)', 'Re(h4)', 'Im(h1)', 'Im(h2)', 'Im(h3)', 'Im(h4)'};
for i = 1:8
    mse_gsfdic_tap = mean(mse_per_tap_gsfdic(:, i));
    mse_dl_tap = mean(mse_per_tap_dl(:, i));
    improvement_ratio = mse_gsfdic_tap / (mse_dl_tap + eps);
    fprintf('║  %-8s ║   %10.4f   ║    %10.6f   ║   %8.1fx      ║\n', tap_labels{i}, mse_gsfdic_tap, mse_dl_tap, improvement_ratio);
end
fprintf('╚════════════╩════════════════╩═════════════════╩═════════════════╝\n');

% Total MSE
total_mse_gsfdic = mean(sum(mse_per_tap_gsfdic, 2));
total_mse_dl = mean(sum(mse_per_tap_dl, 2));
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════════╗\n');
fprintf('║                    TOTAL MSE SUMMARY                              ║\n');
fprintf('╠══════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Total MSE (GSFDIC):          %.4f                              ║\n', total_mse_gsfdic);
fprintf('║  Total MSE (DL):              %.6f                            ║\n', total_mse_dl);
fprintf('║  MSE Reduction:               %.1fx                              ║\n', total_mse_gsfdic/total_mse_dl);
fprintf('║  MSE Reduction (%%):            %.2f%%                             ║\n', (1 - total_mse_dl/total_mse_gsfdic)*100);
fprintf('╚══════════════════════════════════════════════════════════════════╝\n');

% Improvement
improvement = mean(nmse_gsfdic) - mean(nmse_dl);
win_rate = sum(nmse_dl < nmse_gsfdic) / Nmc * 100;

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════════╗\n');
fprintf('║                       IMPROVEMENT ANALYSIS                        ║\n');
fprintf('╠══════════════════════════════════════════════════════════════════╣\n');
fprintf('║  DL improves over GSFDIC by:      %+.2f dB (mean NMSE)           ║\n', improvement);
fprintf('║  DL wins in:                      %.1f%% of trials                ║\n', win_rate);
fprintf('╚══════════════════════════════════════════════════════════════════╝\n');

% Statistical test
[~, pval] = ttest(nmse_gsfdic, nmse_dl);
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════════╗\n');
fprintf('║                     STATISTICAL SIGNIFICANCE                      ║\n');
fprintf('╠══════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Paired t-test p-value:          %.2e                           ║\n', pval);
if pval < 0.05
    fprintf('║  Result: STATISTICALLY SIGNIFICANT (p < 0.05)                    ║\n');
else
    fprintf('║  Result: NOT statistically significant                           ║\n');
end
fprintf('╚══════════════════════════════════════════════════════════════════╝\n');

%% Create figures folder
figFolder = 'figures';
if ~isfolder(figFolder)
    mkdir(figFolder);
end
timestamp = datestr(now, 'yyyymmdd_HHMMSS');

%% Plot 1: CDF Comparison
fig1 = figure('Position', [100 100 800 500]);
hold on; grid on;
plot(sort(nmse_gsfdic), (1:Nmc)/Nmc, 'b-', 'LineWidth', 2.5);
plot(sort(nmse_dl), (1:Nmc)/Nmc, 'r--', 'LineWidth', 2.5);
legend({'GSFDIC fb=4 (original)', '1-pass + Deep Learning'}, 'Location', 'best', 'FontSize', 12);
xlabel('NMSE (dB)', 'FontSize', 14);
ylabel('CDF', 'FontSize', 14);
title('CDF of Channel Estimation NMSE', 'FontSize', 16);
set(gca, 'FontSize', 12);
saveas(fig1, fullfile(figFolder, ['cdf_comparison_' timestamp '.png']));
saveas(fig1, fullfile(figFolder, ['cdf_comparison_' timestamp '.fig']));

%% Plot 2: Box Plot
fig2 = figure('Position', [100 650 600 400]);
boxplot([nmse_gsfdic, nmse_dl], 'Labels', {'GSFDIC fb=4', '1-pass + DL'});
ylabel('NMSE (dB)', 'FontSize', 14);
title('Distribution Comparison', 'FontSize', 16);
grid on;
saveas(fig2, fullfile(figFolder, ['boxplot_comparison_' timestamp '.png']));
saveas(fig2, fullfile(figFolder, ['boxplot_comparison_' timestamp '.fig']));

%% Plot 3: Scatter Comparison
fig3 = figure('Position', [720 100 600 500]);
scatter(nmse_gsfdic, nmse_dl, 40, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
hold on; grid on;
minVal = min([nmse_gsfdic; nmse_dl]) - 2;
maxVal = max([nmse_gsfdic; nmse_dl]) + 2;
plot([minVal maxVal], [minVal maxVal], 'k--', 'LineWidth', 1.5);
xlabel('GSFDIC fb=4 NMSE (dB)', 'FontSize', 14);
ylabel('1-pass + DL NMSE (dB)', 'FontSize', 14);
title('DL vs GSFDIC Performance', 'FontSize', 16);
legend({'Trials', 'Equal performance'}, 'Location', 'best');
xlim([minVal maxVal]); ylim([minVal maxVal]);
set(gca, 'FontSize', 12);
saveas(fig3, fullfile(figFolder, ['scatter_comparison_' timestamp '.png']));
saveas(fig3, fullfile(figFolder, ['scatter_comparison_' timestamp '.fig']));

%% Plot 4: Per-trial comparison
fig4 = figure('Position', [720 650 700 400]);
plot(1:Nmc, nmse_gsfdic, 'bo-', 'LineWidth', 1, 'MarkerSize', 4);
hold on; grid on;
plot(1:Nmc, nmse_dl, 'r^-', 'LineWidth', 1, 'MarkerSize', 4);
legend({'GSFDIC fb=4', '1-pass + DL'}, 'Location', 'best', 'FontSize', 12);
xlabel('Trial Index', 'FontSize', 14);
ylabel('NMSE (dB)', 'FontSize', 14);
title('Per-Trial Channel Estimation Error', 'FontSize', 16);
set(gca, 'FontSize', 12);
saveas(fig4, fullfile(figFolder, ['pertrial_comparison_' timestamp '.png']));
saveas(fig4, fullfile(figFolder, ['pertrial_comparison_' timestamp '.fig']));

%% Plot 5: Histogram of improvement
fig5 = figure('Position', [100 200 600 400]);
improvement_per_trial = nmse_gsfdic - nmse_dl;
histogram(improvement_per_trial, 20, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'white');
hold on;
xline(0, 'r--', 'LineWidth', 2);
xline(mean(improvement_per_trial), 'g-', 'LineWidth', 2);
xlabel('Improvement (GSFDIC - DL) dB', 'FontSize', 14);
ylabel('Count', 'FontSize', 14);
title('Distribution of DL Improvement', 'FontSize', 16);
legend({'Trials', 'No improvement', sprintf('Mean: %.1f dB', mean(improvement_per_trial))}, 'Location', 'best');
grid on;
saveas(fig5, fullfile(figFolder, ['improvement_histogram_' timestamp '.png']));
saveas(fig5, fullfile(figFolder, ['improvement_histogram_' timestamp '.fig']));

fprintf('\nPlots generated.\n');

%% Helper function
function h_hat = ls_fir_from_pilot(x, y, L)
    x = x(:); y = y(:);
    N = length(x);
    if N <= L, error('Not enough samples'); end
    rows = N - L + 1;
    Xc = zeros(rows, L);
    for kk = 1:rows
        Xc(kk, :) = x(kk+L-1:-1:kk).';
    end
    yv = y(L:end);
    h_hat = (Xc' * Xc + 1e-8 * eye(L)) \ (Xc' * yv);
end
