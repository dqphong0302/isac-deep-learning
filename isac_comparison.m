% isac_comparison.m
% =========================================================================
% Comparison between Original GIFDIC (4 iterations) and DL V3.6 (2 phases)
% Signal generation EXACTLY matches goc/isac_anh_phong.m
% =========================================================================
clc; clear; close all;

if ~isdeployed
    addpath('./useful_function/');
end

%% Python + ONNX Setup for DL
pe = pyenv;
try
    pymod = py.importlib.import_module("onnx_inference_helper");
    py.importlib.reload(pymod);
catch ME
    warning("Python module helper issue: %s", ME.message);
end
np = py.importlib.import_module("numpy");
onnxFile = "v36_opt.onnx";
if ~isfile(onnxFile), error('ONNX file not found: %s', onnxFile); end

%% System Parameters (EXACTLY like original script)
global Nc Ns fC c Delta_f Ts Te;
c = 3e8; Delta_f = 78.125e3; Nc = 1024; Ns = 512; fC = 5e9;
delta_r = 1.1574; r_max = 2500; delta_v = 2.68; v_max = 75;
Te = 1/Delta_f; Ts = Te;
lambda_rf = c / fC;
expandVector = @(v, m) v(:, ones(1, m));

n = 0:Nc-1; k = n';
DFT_matrix = 1/sqrt(Nc) * exp(-2i * pi * k * n / Nc);

%% Parameters
no_of_fb = 4;       % GIFDIC uses 4 iterations
hcom_length = 4;
epsilon = 0.0001;
N_pilot = 2*Nc;     % 2048 pilots for RLS

Np_pilot_dl = 4;    % DL uses only 4 pilot symbols
Ridge_Lambda = 0.1;
B = 64;
percentiles = [10 25 50 75 90];

%% SI and Target Configuration
si_power = [100 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10];
si_velocity = [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
si_range = [1 10 10 10 10 10 11 11 12 12 13 11 11 12 12 13];
target_power = [-10 -10 -10];
target_velocity = [12 -17 -15];
target_range = [5 10 15];
k_target = length(target_power);

%% Modulation Setup
bitsPerSymbol = 1; qam = 2^(bitsPerSymbol);
data = randi([0 qam - 1], Nc, Ns);
Ftx_radar = qammod(data, qam, 'UnitAveragePower', true);

noise_power = 0; com_power = 30;
com_power_sqrt = sqrt(10^(com_power/10));

M = 2;
pskmodObj = comm.PSKModulator(M, 'BitInput', true);
pskdemodObj = comm.PSKDemodulator(M, 'BitOutput', true);

%% Signal Generation (EXACTLY like original)
Y_target = generate_target_echo(target_power, target_velocity, target_range, Ftx_radar);
Y_si = generate_target_echo(si_power, si_velocity, si_range, Ftx_radar);
R_noise = generate_noise(noise_power);

Ftx_com = zeros(Nc, Ns);
Mat_x_com_bit = zeros(log2(M) * Nc, Ns);
for col = 1:Ns
    x_com_bit = randi([0 1], 1, log2(M) * Nc);
    Ftx_com(:, col) = pskmodObj(x_com_bit');
    Mat_x_com_bit(:, col) = x_com_bit';
end
x_com_td = DFT_matrix' * Ftx_com;
x_com_td_vec = reshape(x_com_td, [], 1);
Ftx_com_vec = reshape(Ftx_com, [], 1);

hcom_true = db2mag([0 -9.7 -19.2 -22.8]).' .* exp(1j*[0 -.8 1.6 -2.6].') .* com_power_sqrt;
y_com_td = filter(hcom_true, 1, x_com_td_vec);
Y_com = reshape(y_com_td, Nc, Ns);
A = DFT_matrix * (Y_target + Y_si + Y_com) + R_noise;

fprintf('========================================\n');
fprintf('ISAC Comparison: GIFDIC vs DL V3.6\n');
fprintf('========================================\n\n');

%% ===================== METHOD 1: Original GIFDIC (4 iterations) =====================
fprintf('Running GIFDIC (Original, 4 iterations)...\n');
tic;

hcom_est_rls = zeros(hcom_length, 1);
P1 = (epsilon^-1) * eye(hcom_length);
Y_target_rec = zeros(Nc, Ns);
Y_com_rec = zeros(Nc, Ns);
H_comp_mat = zeros(Nc, Ns);

for fb = 1:no_of_fb
    Y_target_rec_prev = Y_target_rec;
    Y_com_rec_prev = Y_com_rec;
    A_tilde = A - Y_com_rec_prev - Y_target_rec_prev;
    
    if fb > 1
        H_comp = -1/(Ns-1) * ((A_tilde(:,1)./Ftx_radar(:,1)) - (A_tilde(:,end)./Ftx_radar(:,end)));
        H_comp_mat = expandVector(H_comp, Ns);
    else
        H_comp_mat = zeros(Nc, Ns);
    end
    
    E_si_suppress_tilde = fdic(A_tilde, Ftx_radar, H_comp_mat);
    E_radar_comp = E_si_suppress_tilde + Y_target_rec_prev;
    RDM = calc_periodogram(E_radar_comp, Ftx_radar, 1/100, 1/100);
    targetout = cfar_ofdm_radar(RDM, k_target, 1, 3, 3, 3, 3);
    Y_target_rec = DFT_matrix * generate_target_echo(target_power, targetout.velocities, targetout.ranges, Ftx_radar);
    
    E_tilde = E_si_suppress_tilde + Y_target_rec_prev - Y_target_rec + Y_com_rec_prev;
    
    x_com_pilot = x_com_td_vec(1:N_pilot);
    E_td = reshape(DFT_matrix' * E_tilde, [], 1);
    Ep_td = E_td(1:N_pilot);
    for m = hcom_length:length(x_com_pilot)
        [hcom_est_rls, P1] = RLS_function(hcom_est_rls, Ep_td(m), x_com_pilot(m:-1:m-(hcom_length-1)), P1);
    end
    
    Ec_td = reshape(DFT_matrix' * E_tilde, [], 1);
    Ec = DFT_matrix * reshape(filter(1, hcom_est_rls', Ec_td), Nc, Ns);
    
    Ftx_com_est = zeros(Nc, Ns);
    for kk = 1:Ns
        bits_est = pskdemodObj(Ec(:, kk));
        Ftx_com_est(:, kk) = pskmodObj(bits_est);
    end
    Ftx_com_est = reshape(Ftx_com_est, [], 1);
    Ftx_com_est(1:N_pilot) = Ftx_com_vec(1:N_pilot);
    Ftx_com_est = reshape(Ftx_com_est, Nc, Ns);
    
    x_com_est = reshape(DFT_matrix' * Ftx_com_est, [], 1);
    Y_com_rec = DFT_matrix * reshape(filter(hcom_est_rls', 1, x_com_est), Nc, Ns);
end

E_output_gifdic = E_radar_comp;
time_gifdic = toc;
nmse_gifdic = 10*log10(norm(hcom_est_rls - hcom_true)^2 / (norm(hcom_true)^2 + eps));
fprintf('GIFDIC: NMSE = %.2f dB, Time = %.2f s\n', nmse_gifdic, time_gifdic);

%% ===================== METHOD 2: DL V3.6 (2 Phases) =====================
fprintf('Running DL V3.6 (2-Phase Processing)...\n');
tic;

% DL Estimation (one-shot)
E_dl_raw = fdic(A, Ftx_radar, zeros(Nc, Ns));
e_td_raw = DFT_matrix' * E_dl_raw;
x_pilot_td = x_com_td(:, 1:Np_pilot_dl);
y_pilot_td = e_td_raw(:, 1:Np_pilot_dl);
h_ls = ls_fir_from_pilot_robust(reshape(x_pilot_td,[],1), reshape(y_pilot_td,[],1), hcom_length, Ridge_Lambda);

norm_factor = sqrt(mean(abs(E_dl_raw(:)).^2)) + eps;
E_n = E_dl_raw ./ norm_factor;
logP = log10(mean(abs(E_n).^2, 2) + 1e-12);

bandSize = Nc / B;
band_mean = arrayfun(@(b) mean(logP((b-1)*bandSize+1:b*bandSize)), 1:B)';
band_std = arrayfun(@(b) std(logP((b-1)*bandSize+1:b*bandSize)), 1:B)';
band_max = arrayfun(@(b) max(logP((b-1)*bandSize+1:b*bandSize)), 1:B)';
g_p = prctile(logP, percentiles);
g = [mean(logP); std(logP); max(logP); min(logP); g_p(:)];

x_vec = single([band_mean; band_std; band_max; g; log10(norm_factor); single([real(h_ls(:)); imag(h_ls(:))])]);
y_list = pymod.predict_hcom(np.array(x_vec.', pyargs('dtype', np.float32)), char(onnxFile));
yhat8 = double(cell2mat(cell(y_list))); yhat8 = yhat8(:);
hcom_est_dl = yhat8(1:4) + 1j*yhat8(5:8);

% ========== PHASE 1: Remove SI and Communication ==========
Y_target_rec = zeros(Nc, Ns);
Y_com_rec = zeros(Nc, Ns);
H_comp_mat = zeros(Nc, Ns);

A_tilde = A;
E_si_suppress_tilde = fdic(A_tilde, Ftx_radar, H_comp_mat);
E_radar_comp = E_si_suppress_tilde;
RDM = calc_periodogram(E_radar_comp, Ftx_radar, 1/100, 1/100);
targetout = cfar_ofdm_radar(RDM, k_target, 1, 3, 3, 3, 3);
Y_target_rec = DFT_matrix * generate_target_echo(target_power, targetout.velocities, targetout.ranges, Ftx_radar);

E_tilde = E_si_suppress_tilde - Y_target_rec;
Ec_td = reshape(DFT_matrix' * E_tilde, [], 1);
Ec = DFT_matrix * reshape(filter(1, hcom_est_dl', Ec_td), Nc, Ns);

Ftx_com_est = zeros(Nc, Ns);
for kk = 1:Ns
    bits_est = pskdemodObj(Ec(:, kk));
    Ftx_com_est(:, kk) = pskmodObj(bits_est);
end
Ftx_com_vec_local = reshape(Ftx_com, [], 1);
Ftx_com_est = reshape(Ftx_com_est, [], 1);
N_pilot_samples = Np_pilot_dl * Nc;
Ftx_com_est(1:N_pilot_samples) = Ftx_com_vec_local(1:N_pilot_samples);
Ftx_com_est = reshape(Ftx_com_est, Nc, Ns);

x_com_est = reshape(DFT_matrix' * Ftx_com_est, [], 1);
Y_com_rec = DFT_matrix * reshape(filter(hcom_est_dl', 1, x_com_est), Nc, Ns);

% ========== PHASE 2: Refine Target Detection ==========
Y_target_rec_prev = Y_target_rec;
Y_com_rec_prev = Y_com_rec;
A_tilde = A - Y_com_rec_prev - Y_target_rec_prev;

H_comp = -1/(Ns-1) * ((A_tilde(:,1)./Ftx_radar(:,1)) - (A_tilde(:,end)./Ftx_radar(:,end)));
H_comp_mat = expandVector(H_comp, Ns);

E_si_suppress_tilde = fdic(A_tilde, Ftx_radar, H_comp_mat);
E_radar_comp_dl = E_si_suppress_tilde + Y_target_rec_prev;

E_output_dl = E_radar_comp_dl;
time_dl = toc;
nmse_dl = 10*log10(norm(hcom_est_dl - hcom_true)^2 / (norm(hcom_true)^2 + eps));
fprintf('DL V3.6: NMSE = %.2f dB, Time = %.2f s\n\n', nmse_dl, time_dl);

%% ===================== COMPARISON FIGURES =====================
figFolder = 'figures_comparison';
if ~isfolder(figFolder); mkdir(figFolder); end
timestamp = datestr(now, 'yyyymmdd_HHMMSS');

fig1 = figure('Position', [100 100 700 600], 'Name', 'GIFDIC (4 iter)');
plot_periodogram(E_output_gifdic, Ftx_radar, 1/100, 1/60);
title(sprintf('GIFDIC (4 iterations)\nNMSE = %.2f dB, Time = %.2f s', nmse_gifdic, time_gifdic));
figName1 = fullfile(figFolder, ['GIFDIC_4iter_' timestamp '.png']);
saveas(fig1, figName1);
fprintf('Figure 1 saved: %s\n', figName1);

fig2 = figure('Position', [850 100 700 600], 'Name', 'DL V3.6 (2 phases)');
plot_periodogram(E_output_dl, Ftx_radar, 1/100, 1/60);
title(sprintf('DL V3.6 (2 Phases)\nNMSE = %.2f dB, Time = %.2f s', nmse_dl, time_dl));
figName2 = fullfile(figFolder, ['DL_V36_2phase_' timestamp '.png']);
saveas(fig2, figName2);
fprintf('Figure 2 saved: %s\n', figName2);

%% ===================== SUMMARY =====================
fprintf('\n========================================\n');
fprintf('SUMMARY: GIFDIC (4 iter) vs DL V3.6 (2 phases)\n');
fprintf('========================================\n');
fprintf('Method            | NMSE (dB) | Time (s) | Iterations\n');
fprintf('------------------|-----------|----------|------------\n');
fprintf('GIFDIC (Original) | %8.2f  | %7.2f  |     4\n', nmse_gifdic, time_gifdic);
fprintf('DL V3.6           | %8.2f  | %7.2f  |     2\n', nmse_dl, time_dl);
fprintf('------------------|-----------|----------|------------\n');
fprintf('Improvement       | %8.2f  | %7.2f  |    -2\n', nmse_gifdic - nmse_dl, time_gifdic - time_dl);
fprintf('========================================\n');

if nmse_dl < nmse_gifdic
    fprintf('✓ DL V3.6 achieves BETTER NMSE with FEWER iterations!\n');
    fprintf('  Speedup: %.1fx faster\n', time_gifdic / time_dl);
end

%% Helper Functions
function h_hat = ls_fir_from_pilot_robust(x, y, L, lambda)
    x = x(:); y = y(:);
    rows = length(x) - L + 1;
    Xc = zeros(rows, L);
    for kk = 1:rows, Xc(kk, :) = x(kk+L-1:-1:kk).'; end
    h_hat = (Xc' * Xc + lambda * eye(L)) \ (Xc' * y(L:end));
end

function [h_est, P] = RLS_function(h_old, e, u, P_old)
    lambda = 0.99;
    k_gain = (P_old * u) / (lambda + u' * P_old * u);
    h_est = h_old + k_gain * (e - u' * h_old);
    P = (P_old - k_gain * u' * P_old) / lambda;
end
