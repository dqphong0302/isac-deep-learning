% generate_dataset_v7_hybrid.m
% -------------------------------------------------------------------------
% V7: Hybrid Dataset for V3.6 + CNN Residual Refinement
%
% Output:
%   X_spectral: 210-dim spectral features (same as V3.6)
%   X_2d: Downsampled E matrix for CNN (64x64x2 real/imag)
%   Y: hcom (8 real values)
% -------------------------------------------------------------------------

clc; clear; close all;

if ~isdeployed
    addpath('./useful_function/');
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

%% Dataset config
numSamples  = 5000;
hcom_length = 4;

% Spectral features (same as V3.6)
B = 64;
bandSize = Nc / B;
percentiles = [10 25 50 75 90];
G = 4 + numel(percentiles);
spectralDim = 3*B + G + 1 + 2*hcom_length;  % 210

% 2D E matrix dimensions (downsampled)
E_height = 64;   % Downsample from 1024
E_width  = 64;   % Downsample from 512

Np_pilot = 4;
Ridge_Lambda = 0.1;

outFile = 'dataset_v7_hybrid.h5';
if exist(outFile,'file'); delete(outFile); end

chunkRows = 64;

% X_spectral: (numSamples, 210)
h5create(outFile, '/X_spectral', [numSamples spectralDim], ...
    'Datatype','single', 'ChunkSize',[min(chunkRows,numSamples) spectralDim], 'Deflate', 4);

% X_2d: (numSamples, E_height, E_width, 2) - Real and Imag
h5create(outFile, '/X_2d', [numSamples E_height E_width 2], ...
    'Datatype','single', 'ChunkSize',[1 E_height E_width 2], 'Deflate', 4);

% Y: (numSamples, 8)
h5create(outFile, '/Y', [numSamples 8], ...
    'Datatype','single', 'ChunkSize',[min(chunkRows,numSamples) 8], 'Deflate', 4);

h5create(outFile, '/meta/version', 1, 'Datatype','int32'); 
h5write(outFile, '/meta/version', int32(7));

%% Modulators
M = 2;
pskmodObj = comm.PSKModulator(M, 'BitInput', true);

pilot_bits = randi([0 1], Nc, 1);
Fpilot = pskmodObj(pilot_bits);

hcom_magnitude = db2mag([0 -9.7 -19.2 -22.8]).';

%% Generate samples
fprintf('=== V7: Hybrid Dataset (Spectral + 2D) ===\n');
fprintf('Spectral features: %d\n', spectralDim);
fprintf('2D E matrix: %dx%dx2\n', E_height, E_width);
fprintf('Samples: %d\n\n', numSamples);
tic;

for idx = 1:numSamples
    if mod(idx, 500) == 0 || idx == 1
        fprintf('Sample %d / %d (%.1f%%)\n', idx, numSamples, 100*idx/numSamples);
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
    hcom = hcom_magnitude .* exp(1j * random_phase) .* com_power_sqrt;

    x_com_td_vec = reshape(x_com_td, [], 1);
    y_com_td_vec = filter(hcom, 1, x_com_td_vec);
    Y_com_td = reshape(y_com_td_vec, Nc, Ns);
    Y_com = DFT_matrix * Y_com_td;

    A = DFT_matrix * (Y_target + Y_si) + Y_com + R_noise;

    % FDIC
    H_comp_mat = zeros(Nc, Ns);
    E = fdic(A, Ftx_radar, H_comp_mat);

    % ========== SPECTRAL FEATURES (V3.6) ==========
    norm_factor = sqrt(mean(abs(E(:)).^2)) + eps;
    E_n = E ./ norm_factor;
    P_sc = mean(abs(E_n).^2, 2);
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

    % h_ls with Ridge
    e_td = DFT_matrix' * E;
    x_pilot_td = x_com_td(:, 1:Np_pilot);
    y_pilot_td = e_td(:, 1:Np_pilot);
    h_ls = ls_fir_from_pilot_robust(reshape(x_pilot_td,[],1), reshape(y_pilot_td,[],1), hcom_length, Ridge_Lambda);

    nf_feat = single(log10(norm_factor));
    hrls8 = single([real(h_ls(:)); imag(h_ls(:))]);

    x_spectral = single([band_mean; band_std; band_max; g; nf_feat; hrls8]);

    % ========== 2D E MATRIX (CNN) ==========
    % Downsample E from (1024, 512) to (64, 64)
    ds_r = Nc / E_height;  % 16
    ds_c = Ns / E_width;   % 8
    
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
    
    x_2d = single(cat(3, real(E_ds), imag(E_ds)));

    % ========== LABELS ==========
    y_out = single([real(hcom); imag(hcom)]);

    % Write to HDF5
    h5write(outFile, '/X_spectral', x_spectral.', [idx 1], [1 spectralDim]);
    h5write(outFile, '/X_2d', reshape(x_2d, [1 E_height E_width 2]), [idx 1 1 1], [1 E_height E_width 2]);
    h5write(outFile, '/Y', y_out.', [idx 1], [1 8]);
end

elapsed = toc;
fprintf('\n=== Done! ===\n');
fprintf('Generated %d samples in %.1f seconds\n', numSamples, elapsed);
fprintf('File: %s\n', outFile);

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
