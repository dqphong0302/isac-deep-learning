% generate_dataset_v4_raw_iq.m
% -------------------------------------------------------------------------
% V4: Raw IQ CNN - Truly Blind Channel Estimation
%
% Key differences:
%   - NO feature extraction (no band_mean, no h_ls)
%   - Saves RAW E_dl matrix as input (1024 x 512 x 2)
%   - CNN will learn features directly from signal
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

% V4: Save downsampled signal (reduce memory)
% Downsample factors
Nc_ds = 256;   % Downsample from 1024 to 256
Ns_ds = 128;   % Downsample from 512 to 128

outFile = 'dataset_v4_raw_iq.h5';
if exist(outFile,'file'); delete(outFile); end

% X: (numSamples, Nc_ds, Ns_ds, 2) - Real and Imag
h5create(outFile, '/X', [numSamples Nc_ds Ns_ds 2], ...
    'Datatype','single', 'ChunkSize',[1 Nc_ds Ns_ds 2], 'Deflate', 4);
% Y: (numSamples, 8) - h_com real/imag
h5create(outFile, '/Y', [numSamples 8], ...
    'Datatype','single', 'ChunkSize',[64 8], 'Deflate', 4);
% Norm factor for denormalization
h5create(outFile, '/norm_factor', [numSamples 1], ...
    'Datatype','single', 'ChunkSize',[1024 1], 'Deflate', 4);

% Meta
h5create(outFile, '/meta/Nc_ds', 1, 'Datatype','int32'); h5write(outFile,'/meta/Nc_ds', int32(Nc_ds));
h5create(outFile, '/meta/Ns_ds', 1, 'Datatype','int32'); h5write(outFile,'/meta/Ns_ds', int32(Ns_ds));
h5create(outFile, '/meta/version', 1, 'Datatype','int32'); h5write(outFile,'/meta/version', int32(4));

%% Modulators
M = 2;
pskmodObj = comm.PSKModulator(M, 'BitInput', true);

hcom_magnitude = db2mag([0 -9.7 -19.2 -22.8]).';

%% Generate samples
fprintf('=== V4: Raw IQ CNN Dataset ===\n');
fprintf('Output size: (%d, %d, 2) per sample\n', Nc_ds, Ns_ds);
fprintf('Total: %d samples\n\n', numSamples);
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

    % Communication (NO pilots - truly blind)
    Ftx_com = zeros(Nc, Ns);
    for col = 1:Ns
        x_com_bit = randi([0 1], 1, log2(M) * Nc);
        Ftx_com(:, col) = pskmodObj(x_com_bit');
    end
    x_com_td = DFT_matrix' * Ftx_com;

    % Communication power
    com_power = randi([20 30], 1, 1);
    com_power_sqrt = sqrt(10^(com_power / 10));

    % RANDOM PHASE CHANNEL
    random_phase = 2 * pi * rand(4, 1);
    hcom = hcom_magnitude .* exp(1j * random_phase) .* com_power_sqrt;

    % Apply channel
    x_com_td_vec = reshape(x_com_td, [], 1);
    y_com_td_vec = filter(hcom, 1, x_com_td_vec);
    Y_com_td = reshape(y_com_td_vec, Nc, Ns);
    Y_com = DFT_matrix * Y_com_td;

    % Received signal
    A = DFT_matrix * (Y_target + Y_si) + Y_com + R_noise;

    % FDIC (1-pass, no compensation)
    H_comp_mat = zeros(Nc, Ns);
    E = fdic(A, Ftx_radar, H_comp_mat);

    % Normalize
    norm_factor = sqrt(mean(abs(E(:)).^2)) + eps;
    E_n = E ./ norm_factor;

    % Downsample (average pooling)
    E_ds = zeros(Nc_ds, Ns_ds);
    sc_factor = Nc / Nc_ds;
    sym_factor = Ns / Ns_ds;
    for i = 1:Nc_ds
        for j = 1:Ns_ds
            i_start = (i-1)*sc_factor + 1;
            i_end = i*sc_factor;
            j_start = (j-1)*sym_factor + 1;
            j_end = j*sym_factor;
            E_ds(i,j) = mean(E_n(i_start:i_end, j_start:j_end), 'all');
        end
    end

    % Convert to real tensor (Nc_ds, Ns_ds, 2)
    X_iq = single(cat(3, real(E_ds), imag(E_ds)));
    
    % Reshape for h5write: (1, Nc_ds, Ns_ds, 2)
    X_iq_batch = reshape(X_iq, [1, Nc_ds, Ns_ds, 2]);

    % Labels
    y_out = single([real(hcom); imag(hcom)]);

    % Write
    h5write(outFile, '/X', X_iq_batch, [idx 1 1 1], [1 Nc_ds Ns_ds 2]);
    h5write(outFile, '/Y', y_out.', [idx 1], [1 8]);
    h5write(outFile, '/norm_factor', single(norm_factor), [idx 1], [1 1]);
end

elapsed = toc;
fprintf('\n=== Done! ===\n');
fprintf('Generated %d samples in %.1f seconds\n', numSamples, elapsed);
fprintf('Saved to: %s\n', outFile);
fprintf('X shape: (%d, %d, %d, 2)\n', numSamples, Nc_ds, Ns_ds);
