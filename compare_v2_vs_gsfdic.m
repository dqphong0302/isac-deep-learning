% compare_v2_vs_gsfdic.m
% -------------------------------------------------------------------------
% V2: Semi-Blind with Random Phase Comparison
%
% Uses RANDOM PHASE channel + h_ls features (210 input)
% -------------------------------------------------------------------------

clc; clear; close all;

addpath('./useful_function/');

%% Python setup
pe = pyenv;
fprintf("MATLAB is using Python: %s\n", string(pe.Executable));

%% Settings
onnxFile = "v2.onnx";

if ~isfile(onnxFile)
    error('ONNX file not found: %s', onnxFile);
end

Nmc = 100;
no_of_fb = 4;

%% System params
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

%% Params
hcom_length = 4;
N_pilot = 2*Nc;
Np_pilot_dl = 8;

M = 2;
pskmodObj = comm.PSKModulator(M, 'BitInput', true);
pskdemodObj = comm.PSKDemodulator(M, 'BitOutput', true);

B = 64;
percentiles = [10 25 50 75 90];

hcom_magnitude = db2mag([0 -9.7 -19.2 -22.8]).';

% Fixed pilot
pilot_bits = randi([0 1], Nc, 1);
Fpilot = pskmodObj(pilot_bits);

%% Python
try
    pymod = py.importlib.import_module("onnx_inference_helper");
    py.importlib.reload(pymod);
catch ME
    error("Cannot import onnx_inference_helper.py: %s", ME.message);
end
np = py.importlib.import_module("numpy");

%% Storage
nmse_gsfdic = zeros(Nmc, 1);
nmse_v2 = zeros(Nmc, 1);

%% Monte Carlo
fprintf('\n========================================\n');
fprintf('V2: Semi-Blind with Random Phase Comparison\n');
fprintf('Running %d Monte Carlo trials...\n', Nmc);
fprintf('========================================\n\n');
tic;

for it = 1:Nmc
    if mod(it, 20) == 0 || it == 1
        fprintf("Trial %d / %d\n", it, Nmc);
    end

    %% Signal Generation
    data = randi([0 1], Nc, Ns);
    Ftx_radar = qammod(data, 2, 'UnitAveragePower', true);
    
    numTarget = randi([1 5], 1, 1);
    k_target = numTarget;
    target_power = -25 + 20*rand(1, numTarget);
    target_velocity = (2*rand(1, numTarget)-1) * v_max;
    target_range = (5 + randi([0 floor(r_max/delta_r)-5], 1, numTarget)) * delta_r;
    Y_target = generate_target_echo(target_power, target_velocity, target_range, Ftx_radar);

    numSI = randi([5 16], 1, 1);
    si_power = -35 + 30*rand(1, numSI);
    si_velocity = [zeros(1, floor(numSI/2)), (2*rand(1, ceil(numSI/2))-1)*2];
    si_range = randi([1 100], 1, numSI) * delta_r;
    Y_si = generate_target_echo(si_power, si_velocity, si_range, Ftx_radar);

    noise_power = randi([0 20], 1, 1);
    R_noise = generate_noise(noise_power);

    Ftx_com = zeros(Nc, Ns);
    Mat_x_com_bit = zeros(log2(M) * Nc, Ns);
    for col = 1:Ns
        if col <= Np_pilot_dl
            Ftx_com(:, col) = Fpilot;
        else
            x_com_bit = randi([0 1], 1, log2(M) * Nc);
            Ftx_com(:, col) = pskmodObj(x_com_bit');
        end
        Mat_x_com_bit(:, col) = randi([0 1], log2(M) * Nc, 1);
    end
    x_com_td = DFT_matrix' * Ftx_com;
    x_com_td_vec = reshape(x_com_td, [], 1);
    Ftx_com_vec = reshape(Ftx_com, [], 1);

    com_power = randi([20 30], 1, 1);
    com_power_sqrt = sqrt(10^(com_power/10));

    % RANDOM PHASE
    random_phase = 2 * pi * rand(4, 1);
    hcom_true = hcom_magnitude .* exp(1j * random_phase) .* com_power_sqrt;

    y_com_td = filter(hcom_true, 1, x_com_td_vec);
    Y_com = reshape(y_com_td, Nc, Ns);

    A = DFT_matrix * (Y_target + Y_si + Y_com) + R_noise;

    %% GSFDIC
    hcom_est = zeros(hcom_length, 1);
    epsilon = 0.0001;
    P1 = (epsilon^-1) * eye(hcom_length);
    Ftx_com_est = zeros(Nc, Ns);
    Y_target_rec = zeros(Nc, Ns);
    Y_com_rec = zeros(Nc, Ns);
    H_comp_mat = zeros(Nc, Ns);
    Mat_x_com_bit_est = zeros(log2(M) * Nc, Ns);

    for fb = 1:no_of_fb
        Y_target_rec_prev = Y_target_rec;
        Y_com_rec_prev = Y_com_rec;
        A_tilde = A - Y_com_rec_prev - Y_target_rec_prev;

        if fb > 1
            H_comp = -1/(Ns-1) * ((A_tilde(:,1) ./ Ftx_radar(:,1)) - (A_tilde(:,end) ./ Ftx_radar(:,end)));
            H_comp_mat = expandVector(H_comp, Ns);
        else
            H_comp_mat = zeros(Nc, Ns);
        end

        E_si_suppress_tilde = fdic(A_tilde, Ftx_radar, H_comp_mat);
        E_radar_comp_gifdic = E_si_suppress_tilde + Y_target_rec_prev;
        RDM = calc_periodogram(E_radar_comp_gifdic, Ftx_radar, 1/100, 1/100);
        targetout = cfar_ofdm_radar(RDM, k_target, 1, 3, 3, 3, 3);
        Y_target_rec = generate_target_echo(target_power, targetout.velocities, targetout.ranges, Ftx_radar);
        Y_target_rec = DFT_matrix * Y_target_rec;

        E_tilde = E_si_suppress_tilde + Y_target_rec_prev - Y_target_rec + Y_com_rec_prev;

        x_com_pilot = x_com_td_vec(1:N_pilot);
        E_td = DFT_matrix' * E_tilde;
        Ep_td = reshape(E_td, [], 1);
        Ep_td = Ep_td(1:N_pilot);

        for m = hcom_length:length(x_com_pilot)
            [hcom_est, P1] = RLS_function(hcom_est, Ep_td(m), x_com_pilot(m:-1:m-(hcom_length-1)), P1);
        end

        Ec_td = reshape(DFT_matrix' * E_tilde, [], 1);
        Ec = filter(1, hcom_est', Ec_td);
        Ec = DFT_matrix * reshape(Ec, Nc, Ns);

        for kk = 1:Ns
            Mat_x_com_bit_est(:, kk) = pskdemodObj(Ec(:, kk));
            Ftx_com_est(:, kk) = pskmodObj(Mat_x_com_bit_est(:, kk));
        end
        Ftx_com_est_vec = reshape(Ftx_com_est, [], 1);
        Ftx_com_est_vec(1:N_pilot) = Ftx_com_vec(1:N_pilot);
        Ftx_com_est = reshape(Ftx_com_est_vec, Nc, Ns);

        x_com_est = reshape(DFT_matrix' * Ftx_com_est, [], 1);
        Y_com_rec = filter(hcom_est', 1, x_com_est);
        Y_com_rec = DFT_matrix * reshape(Y_com_rec, Nc, Ns);
    end

    nmse_gsfdic(it) = 10*log10(norm(hcom_est - hcom_true)^2 / (norm(hcom_true)^2 + eps));

    %% V2 DL
    H_comp_mat_dl = zeros(Nc, Ns);
    E_dl = fdic(A, Ftx_radar, H_comp_mat_dl);

    % LS estimate
    e_td = DFT_matrix' * E_dl;
    x_pilot_td = x_com_td(:, 1:Np_pilot_dl);
    y_pilot_td = e_td(:, 1:Np_pilot_dl);
    h_ls = ls_fir_from_pilot(reshape(x_pilot_td,[],1), reshape(y_pilot_td,[],1), hcom_length);

    % Features
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

    nf_feat = single(log10(norm_factor));
    hrls8 = single([real(h_ls(:)); imag(h_ls(:))]);

    % 210 features
    x_vec = single([band_mean; band_std; band_max; g; nf_feat; hrls8]);

    % ONNX
    x_py = np.array(x_vec.', pyargs('dtype', np.float32));
    y_list = pymod.predict_hcom(x_py, char(onnxFile));
    yhat8 = double(cell2mat(cell(y_list)));
    yhat8 = yhat8(:);

    hcom_v2 = yhat8(1:4) + 1j*yhat8(5:8);
    nmse_v2(it) = 10*log10(norm(hcom_v2 - hcom_true)^2 / (norm(hcom_true)^2 + eps));
end

elapsed = toc;
fprintf('\nCompleted %d trials in %.1f seconds\n', Nmc, elapsed);

%% Results
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════════╗\n');
fprintf('║           V2: SEMI-BLIND + RANDOM PHASE COMPARISON               ║\n');
fprintf('╠══════════════════════════════════════════════════════════════════╣\n');
fprintf('║                        NMSE (dB) Statistics                       ║\n');
fprintf('╠═══════════════════╦══════════════════╦═══════════════════════════╣\n');
fprintf('║     Metric        ║  GSFDIC fb=4     ║   V2 DL (Random Phase)    ║\n');
fprintf('╠═══════════════════╬══════════════════╬═══════════════════════════╣\n');
fprintf('║  Mean             ║    %+8.2f dB    ║       %+8.2f dB          ║\n', mean(nmse_gsfdic), mean(nmse_v2));
fprintf('║  Median           ║    %+8.2f dB    ║       %+8.2f dB          ║\n', median(nmse_gsfdic), median(nmse_v2));
fprintf('║  Std Dev          ║    %8.2f dB    ║       %8.2f dB          ║\n', std(nmse_gsfdic), std(nmse_v2));
fprintf('║  Min (Best)       ║    %+8.2f dB    ║       %+8.2f dB          ║\n', min(nmse_gsfdic), min(nmse_v2));
fprintf('║  Max (Worst)      ║    %+8.2f dB    ║       %+8.2f dB          ║\n', max(nmse_gsfdic), max(nmse_v2));
fprintf('╚═══════════════════╩══════════════════╩═══════════════════════════╝\n');

improvement = mean(nmse_gsfdic) - mean(nmse_v2);
win_rate = sum(nmse_v2 < nmse_gsfdic) / Nmc * 100;

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════════╗\n');
fprintf('║                       IMPROVEMENT ANALYSIS                        ║\n');
fprintf('╠══════════════════════════════════════════════════════════════════╣\n');
fprintf('║  V2 DL improves over GSFDIC by:    %+.2f dB                      ║\n', improvement);
fprintf('║  V2 DL wins in:                    %.1f%% of trials               ║\n', win_rate);
fprintf('╚══════════════════════════════════════════════════════════════════╝\n');

[~, pval] = ttest(nmse_gsfdic, nmse_v2);
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════════╗\n');
fprintf('║  Paired t-test p-value:          %.2e                           ║\n', pval);
if pval < 0.05
    fprintf('║  Result: STATISTICALLY SIGNIFICANT                              ║\n');
end
fprintf('╚══════════════════════════════════════════════════════════════════╝\n');

%% Plots
figFolder = 'figures_v2';
if ~isfolder(figFolder); mkdir(figFolder); end
timestamp = datestr(now, 'yyyymmdd_HHMMSS');

fig1 = figure('Position', [100 100 800 500]);
hold on; grid on;
plot(sort(nmse_gsfdic), (1:Nmc)/Nmc, 'b-', 'LineWidth', 2.5);
plot(sort(nmse_v2), (1:Nmc)/Nmc, 'r--', 'LineWidth', 2.5);
legend({'GSFDIC fb=4', 'V2 DL (Random Phase)'}, 'Location', 'best');
xlabel('NMSE (dB)'); ylabel('CDF');
title('V2: CDF of Channel Estimation NMSE');
saveas(fig1, fullfile(figFolder, ['cdf_v2_' timestamp '.png']));

fig2 = figure('Position', [100 650 600 400]);
boxplot([nmse_gsfdic, nmse_v2], 'Labels', {'GSFDIC fb=4', 'V2 DL'});
ylabel('NMSE (dB)'); title('V2: Distribution Comparison'); grid on;
saveas(fig2, fullfile(figFolder, ['boxplot_v2_' timestamp '.png']));

fprintf('\nPlots saved to %s/\n', figFolder);

%% Helper
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
