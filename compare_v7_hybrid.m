% compare_v7_hybrid.m
% -------------------------------------------------------------------------
% Compare V7 Hybrid (V3.6 MLP + CNN Refiner) vs GSFDIC
% Monte Carlo simulation - 100 trials
% -------------------------------------------------------------------------

clc; clear; close all;

if ~isdeployed
    addpath('./useful_function/');
end

%% Load Python ONNX helper
try
    pymod = py.importlib.import_module("onnx_inference_helper");
    py.importlib.reload(pymod);
catch
    error('Python ONNX helper not found. Run from directory with onnx_inference_helper.py');
end

%% System params
global Nc Ns fC c Delta_f Ts Te;

c        = 3e8;
Delta_f  = 78.125e3;
Nc       = 1024;
Ns       = 512;
fC       = 5e9;
Te       = 1/Delta_f;
Ts       = Te;

delta_r  = 1.1574;
r_max    = 2500;
v_max    = 75;

n = 0:Nc-1;
k = n';
DFT_matrix = 1/sqrt(Nc) * exp(-2i * pi * k * n / Nc);

%% Parameters
numTrials = 100;
Np_pilot = 4;
Ridge_Lambda = 0.1;
hcom_length = 4;

% Feature dimensions
B = 64;
bandSize = Nc / B;
percentiles = [10 25 50 75 90];
G = 4 + numel(percentiles);
spectralDim = 3*B + G + 1 + 2*hcom_length;  % 210

% CNN input dimensions
E_height = 64;
E_width = 64;

%% Modulators
M = 2;
pskmodObj = comm.PSKModulator(M, 'BitInput', true);
pskdemodObj = comm.PSKDemodulator(M, 'BitOutput', true);

pilot_bits = randi([0 1], Nc, 1);
Fpilot = pskmodObj(pilot_bits);

hcom_magnitude = db2mag([0 -9.7 -19.2 -22.8]).';

%% Numpy helper
np = py.importlib.import_module("numpy");

%% Results storage
nmse_gsfdic = zeros(numTrials, 1);
nmse_v36 = zeros(numTrials, 1);
nmse_v7_hybrid = zeros(numTrials, 1);
nmse_v7_mlp = zeros(numTrials, 1);

fprintf('=== V3.6 vs V7 Hybrid vs GSFDIC Comparison ===\n');
fprintf('Trials: %d, Pilots: %d\n\n', numTrials, Np_pilot);
tic;

for trial = 1:numTrials
    if mod(trial, 20) == 0 || trial == 1
        fprintf('Trial %d / %d\n', trial, numTrials);
    end

    % Radar Tx
    data = randi([0 1], Nc, Ns);
    Ftx_radar = qammod(data, 2, 'UnitAveragePower', true);

    % Random targets
    numTarget = randi([1 5], 1, 1);
    target_power = -25 + 20 * rand(1, numTarget);
    rangeBins = randi([5 floor(r_max/delta_r)], 1, numTarget);
    target_range = rangeBins * delta_r;
    target_velocity = (2*rand(1, numTarget) - 1) * v_max;
    Y_target = generate_target_echo(target_power, target_velocity, target_range, Ftx_radar);

    % Random SI
    numStatic = randi([0 5], 1, 1);
    numSlow = randi([0 15], 1, 1);
    numSI = numStatic + numSlow;
    if numSI > 0
        si_power = -35 + 30 * rand(1, numSI);
        si_rangeBins = randi([1 floor(100/delta_r)], 1, numSI);
        si_range = si_rangeBins * delta_r;
        si_velocity = [zeros(1, numStatic), (2*rand(1, numSlow)-1)*2];
        permIdx = randperm(numSI);
        Y_si = generate_target_echo(si_power(permIdx), si_velocity(permIdx), si_range(permIdx), Ftx_radar);
    else
        Y_si = zeros(Nc, Ns);
    end

    % Noise
    noise_power = randi([0 20], 1, 1);
    R_noise = generate_noise(noise_power);

    % Communication
    Ftx_com = zeros(Nc, Ns);
    for col = 1:Np_pilot
        Ftx_com(:, col) = Fpilot;
    end
    for col = (Np_pilot+1):Ns
        x_com_bit = randi([0 1], 1, log2(M) * Nc);
        Ftx_com(:, col) = pskmodObj(x_com_bit');
    end
    x_com_td = DFT_matrix' * Ftx_com;

    com_power = randi([20 30], 1, 1);
    com_power_sqrt = sqrt(10^(com_power / 10));

    random_phase = 2 * pi * rand(4, 1);
    hcom_true = hcom_magnitude .* exp(1j * random_phase) .* com_power_sqrt;

    x_com_td_vec = reshape(x_com_td, [], 1);
    y_com_td_vec = filter(hcom_true, 1, x_com_td_vec);
    Y_com_td = reshape(y_com_td_vec, Nc, Ns);
    Y_com = DFT_matrix * Y_com_td;

    A = DFT_matrix * (Y_target + Y_si) + Y_com + R_noise;

    % === GSFDIC (4 iterations) ===
    H_comp_mat = zeros(Nc, Ns);
    E_gsfdic = A;
    for iter = 1:4
        E_gsfdic = fdic(A, Ftx_radar, H_comp_mat);
        e_td_gs = DFT_matrix' * E_gsfdic;
        x_pilot_td = x_com_td(:, 1:Np_pilot);
        y_pilot_td = e_td_gs(:, 1:Np_pilot);
        h_gs = ls_fir_from_pilot_robust(reshape(x_pilot_td,[],1), reshape(y_pilot_td,[],1), hcom_length, Ridge_Lambda);
        
        % Reconstruct and compensate
        H_comp_mat_new = zeros(Nc, Ns);
        for col = 1:Ns
            decoded = pskdemodObj(E_gsfdic(:, col));
            remod = pskmodObj(decoded);
            y_comp_td = filter(h_gs, 1, DFT_matrix' * remod);
            H_comp_mat_new(:, col) = DFT_matrix * y_comp_td;
        end
        H_comp_mat = H_comp_mat_new;
    end
    hcom_gsfdic = h_gs;

    % === V7 Hybrid DL ===
    E = fdic(A, Ftx_radar, zeros(Nc, Ns));
    
    % Spectral features
    norm_factor = sqrt(mean(abs(E(:)).^2)) + eps;
    E_n = E ./ norm_factor;
    P_sc = mean(E_n.*conj(E_n), 2);
    logP = log10(P_sc + 1e-12);

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

    e_td = DFT_matrix' * E;
    x_pilot_td = x_com_td(:, 1:Np_pilot);
    y_pilot_td = e_td(:, 1:Np_pilot);
    h_ls = ls_fir_from_pilot_robust(reshape(x_pilot_td,[],1), reshape(y_pilot_td,[],1), hcom_length, Ridge_Lambda);

    nf_feat = single(log10(norm_factor));
    hrls8 = single([real(h_ls(:)); imag(h_ls(:))]);
    x_spectral = single([band_mean; band_std; band_max; g; nf_feat; hrls8]);

    % 2D E matrix for CNN
    ds_r = Nc / E_height;
    ds_c = Ns / E_width;
    E_ds = zeros(E_height, E_width);
    for r = 1:E_height
        for c = 1:E_width
            r_start = (r-1)*ds_r + 1;
            r_end = r*ds_r;
            c_start = (c-1)*ds_c + 1;
            c_end = c*ds_c;
            E_ds(r, c) = mean(mean(E_n(r_start:r_end, c_start:c_end)));
        end
    end
    x_2d = single(cat(3, real(E_ds), imag(E_ds)));  % (64, 64, 2)

    % Prepare for ONNX (batch dim, channel first)
    x_spec_py = np.array(x_spectral.', pyargs('dtype', 'float32'));  % (1, 210)
    x_2d_py = np.array(permute(x_2d, [3, 1, 2]), pyargs('dtype', 'float32'));  % (2, 64, 64)
    x_2d_py = np.expand_dims(x_2d_py, int32(0));  % (1, 2, 64, 64)
    h_rls_py = np.array(hrls8.', pyargs('dtype', 'float32'));  % (1, 8)

    % Run ONNX inference - V7 Hybrid
    yhat = pymod.predict_v7_hybrid(x_spec_py, x_2d_py, h_rls_py, "v7_hybrid.onnx");
    yhat = double(yhat);
    hcom_v7 = yhat(1:4).' + 1j*yhat(5:8).';

    % Run ONNX inference - V3.6
    yhat_v36 = pymod.predict_hcom(x_spec_py, "v36_opt.onnx");
    yhat_v36 = double(yhat_v36);
    hcom_v36 = yhat_v36(1:4).' + 1j*yhat_v36(5:8).';

    % Also get MLP-only result (use h_ls as baseline)
    hcom_v7_mlp = h_ls;

    % === NMSE ===
    nmse_gsfdic(trial) = norm(hcom_gsfdic - hcom_true)^2 / norm(hcom_true)^2;
    nmse_v36(trial) = norm(hcom_v36 - hcom_true)^2 / norm(hcom_true)^2;
    nmse_v7_hybrid(trial) = norm(hcom_v7 - hcom_true)^2 / norm(hcom_true)^2;
    nmse_v7_mlp(trial) = norm(hcom_v7_mlp - hcom_true)^2 / norm(hcom_true)^2;
end

elapsed = toc;
fprintf('\n=== Results (%d trials) ===\n', numTrials);
fprintf('GSFDIC mean:      %.2f dB\n', 10*log10(mean(nmse_gsfdic)));
fprintf('h_ls (Ridge):     %.2f dB\n', 10*log10(mean(nmse_v7_mlp)));
fprintf('V3.6 Opt mean:    %.2f dB\n', 10*log10(mean(nmse_v36)));
fprintf('V7 Hybrid mean:   %.2f dB\n', 10*log10(mean(nmse_v7_hybrid)));
fprintf('\n');
fprintf('V3.6 vs GSFDIC:   +%.2f dB improvement\n', 10*log10(mean(nmse_gsfdic)) - 10*log10(mean(nmse_v36)));
fprintf('V7 vs V3.6:       %.2f dB\n', 10*log10(mean(nmse_v7_hybrid)) - 10*log10(mean(nmse_v36)));
fprintf('Time: %.1f s\n', elapsed);

function h_hat = ls_fir_from_pilot_robust(x, y, L, lambda)
    x = x(:); y = y(:);
    N = length(x);
    rows = N - L + 1;
    Xc = zeros(rows, L);
    for kk = 1:rows
        Xc(kk, :) = x(kk+L-1:-1:kk).';
    end
    yv = y(L:end);
    h_hat = (Xc' * Xc + lambda * eye(L)) \ (Xc' * yv);
end
