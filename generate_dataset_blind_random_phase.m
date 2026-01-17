% generate_dataset_blind_random_phase.m
% -------------------------------------------------------------------------
% V3: TRULY BLIND Dataset Generation
%
% Key differences from V1:
%   1. RANDOM PHASE: exp(1j * 2*pi*rand(4,1)) - different each sample
%   2. NO h_ls feature: Model must learn from spectral features only
%   3. Feature dim: 202 (not 210)
%
% Output HDF5:
%   /X : (N, 202) - spectral features only
%   /Y : (N, 8)   - [Re(hcom); Im(hcom)]
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
numSamples  = 5000;  % More samples for harder task
hcom_length = 4;

% Feature config (NO h_ls!)
B = 64;
bandSize = Nc / B;
percentiles = [10 25 50 75 90];
G = 4 + numel(percentiles);  % global stats
xDim = 3*B + G + 1;          % 192 + 9 + 1 = 202 (NO h_ls 8 features)
yDim = 2*hcom_length;        % 8

% Pilot config (for GSFDIC comparison only, not used in features)
Np_pilot = 8;

outFile = 'dataset_blind_random_phase.h5';
if exist(outFile,'file'); delete(outFile); end

chunkRows = 64;

h5create(outFile, '/X', [numSamples xDim], ...
    'Datatype','single', 'ChunkSize',[min(chunkRows,numSamples) xDim], 'Deflate', 4, 'Shuffle', true);
h5create(outFile, '/Y', [numSamples yDim], ...
    'Datatype','single', 'ChunkSize',[min(chunkRows,numSamples) yDim], 'Deflate', 4, 'Shuffle', true);
h5create(outFile, '/norm_factor', [numSamples 1], ...
    'Datatype','single', 'ChunkSize',[min(numSamples,1024) 1], 'Deflate', 4, 'Shuffle', true);

% Meta
h5create(outFile, '/meta/Nc', 1, 'Datatype','int32'); h5write(outFile,'/meta/Nc', int32(Nc));
h5create(outFile, '/meta/Ns', 1, 'Datatype','int32'); h5write(outFile,'/meta/Ns', int32(Ns));
h5create(outFile, '/meta/hcom_length', 1, 'Datatype','int32'); h5write(outFile,'/meta/hcom_length', int32(hcom_length));
h5create(outFile, '/meta/xDim', 1, 'Datatype','int32'); h5write(outFile,'/meta/xDim', int32(xDim));
h5create(outFile, '/meta/blind', 1, 'Datatype','int32'); h5write(outFile,'/meta/blind', int32(1));
h5create(outFile, '/meta/random_phase', 1, 'Datatype','int32'); h5write(outFile,'/meta/random_phase', int32(1));

%% Modulators
M = 2;
pskmodObj = comm.PSKModulator(M, 'BitInput', true);

% Fixed pilot (not used in features, but needed for signal generation)
pilot_bits = randi([0 1], Nc, 1);
Fpilot = pskmodObj(pilot_bits);

%% PDP for channel magnitude
hcom_magnitude = db2mag([0 -9.7 -19.2 -22.8]).';

%% Generate samples
fprintf('=== V3: TRULY BLIND Dataset Generation ===\n');
fprintf('Features: 202 (NO h_ls from pilot)\n');
fprintf('Phase: RANDOM for each sample\n');
fprintf('Samples: %d\n\n', numSamples);
tic;

for idx = 1:numSamples
    if mod(idx, 500) == 0 || idx == 1
        fprintf('Sample %d / %d (%.1f%%)\n', idx, numSamples, 100*idx/numSamples);
    end

    % Radar Tx
    data = randi([0 1], Nc, Ns);
    Ftx_radar = qammod(data, 2, 'UnitAveragePower', true);

    % Random targets (1-5)
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
        si_power = si_power(permIdx);
        si_range = si_range(permIdx);
        si_velocity = si_velocity(permIdx);

        Y_si = generate_target_echo(si_power, si_velocity, si_range, Ftx_radar);
    else
        Y_si = zeros(Nc, Ns);
    end

    % Noise
    noise_power = randi([0 20], 1, 1);
    R_noise = generate_noise(noise_power);

    % Communication signal
    Ftx_com = zeros(Nc, Ns);
    for col = 1:Np_pilot
        Ftx_com(:, col) = Fpilot;
    end
    for col = (Np_pilot+1):Ns
        x_com_bit = randi([0 1], 1, log2(M) * Nc);
        Ftx_com(:, col) = pskmodObj(x_com_bit');
    end

    x_com_td = DFT_matrix' * Ftx_com;

    % Communication power
    com_power = randi([20 30], 1, 1);
    com_power_sqrt = sqrt(10^(com_power / 10));

    % ========== RANDOM PHASE CHANNEL (KEY DIFFERENCE) ==========
    random_phase = 2 * pi * rand(4, 1);  % Random phase [0, 2π]
    hcom = hcom_magnitude .* exp(1j * random_phase) .* com_power_sqrt;
    % ============================================================

    % Apply channel
    x_com_td_vec = reshape(x_com_td, [], 1);
    y_com_td_vec = filter(hcom, 1, x_com_td_vec);
    Y_com_td = reshape(y_com_td_vec, Nc, Ns);
    Y_com = DFT_matrix * Y_com_td;

    % Received signal
    A = DFT_matrix * (Y_target + Y_si) + Y_com + R_noise;

    % One-pass FDIC
    H_comp_mat = zeros(Nc, Ns);
    E = fdic(A, Ftx_radar, H_comp_mat);

    % ========== FEATURE EXTRACTION (NO h_ls!) ==========
    norm_factor = sqrt(mean(abs(E(:)).^2)) + eps;
    E_n = E ./ norm_factor;

    P_sc = mean(abs(E_n).^2, 2);
    logP = log10(P_sc + 1e-12);

    % Band-aggregated statistics
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

    % Global statistics
    g_mean = mean(logP); g_std = std(logP);
    g_max = max(logP); g_min = min(logP);
    g_p = prctile(logP, percentiles);
    g = [g_mean; g_std; g_max; g_min; g_p(:)];

    % Normalization factor as feature
    nf_feat = single(log10(norm_factor));

    % Feature vector: 192 + 9 + 1 = 202 (NO h_ls!)
    x_in = single([band_mean; band_std; band_max; g; nf_feat]);

    % Labels
    y_out = single([real(hcom); imag(hcom)]);

    % Write
    h5write(outFile, '/X', x_in.', [idx 1], [1 xDim]);
    h5write(outFile, '/Y', y_out.', [idx 1], [1 yDim]);
    h5write(outFile, '/norm_factor', single(norm_factor), [idx 1], [1 1]);
end

elapsed = toc;
fprintf('\n=== Done! ===\n');
fprintf('Generated %d samples in %.1f seconds (%.1f samples/sec)\n', numSamples, elapsed, numSamples/elapsed);
fprintf('Feature dim: %d (truly blind, no h_ls)\n', xDim);
fprintf('Saved to: %s\n', outFile);
