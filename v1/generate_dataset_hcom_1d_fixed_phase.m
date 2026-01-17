% generate_dataset_hcom_1d_fixed_phase.m
% -------------------------------------------------------------------------
% Dataset generation MATCHING EXACTLY with ISAC_anh_Phong.m channel model
%
% Key difference from previous version:
%   - Channel uses FIXED PHASE: exp(1j*[0 -0.8 1.6 -2.6])
%   - Same as original: hcom = db2mag([0 -9.7 -19.2 -22.8]).*exp(1j*[...]).*power
%
% HDF5 output:
%   /X          : (N, xDim) float32 - features
%   /Y          : (N, 8)    float32 - [Re(hcom); Im(hcom)]
%   /Hrls       : (N, 8)    float32 - LS estimate
%   /norm_factor: (N, 1)    float32
% -------------------------------------------------------------------------

clc; clear; close all;

if ~isdeployed
    addpath('./useful_function/');
end

%% System params (SAME AS ISAC_anh_Phong.m)
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
numSamples  = 2000;
hcom_length = 4;

% Feature config
B = 64;
bandSize = Nc / B;
use_percentiles = true;
percentiles = [10 25 50 75 90];
G = 4 + numel(percentiles);
xDim = 3*B + G;
yDim = 2*hcom_length;

% Pilot config
Np_pilot = 8;  % 8 OFDM symbols as pilot

outFile = 'dataset_hcom_1d_fixed_phase.h5';
if exist(outFile,'file'); delete(outFile); end

chunkRows = 64;

h5create(outFile, '/X', [numSamples xDim], ...
    'Datatype','single', 'ChunkSize',[min(chunkRows,numSamples) xDim], 'Deflate', 4, 'Shuffle', true);
h5create(outFile, '/Y', [numSamples yDim], ...
    'Datatype','single', 'ChunkSize',[min(chunkRows,numSamples) yDim], 'Deflate', 4, 'Shuffle', true);
h5create(outFile, '/Hrls', [numSamples yDim], ...
    'Datatype','single', 'ChunkSize',[min(chunkRows,numSamples) yDim], 'Deflate', 4, 'Shuffle', true);
h5create(outFile, '/norm_factor', [numSamples 1], ...
    'Datatype','single', 'ChunkSize',[min(numSamples,1024) 1], 'Deflate', 4, 'Shuffle', true);

% Meta
h5create(outFile, '/meta/Nc', 1, 'Datatype','int32'); h5write(outFile,'/meta/Nc', int32(Nc));
h5create(outFile, '/meta/Ns', 1, 'Datatype','int32'); h5write(outFile,'/meta/Ns', int32(Ns));
h5create(outFile, '/meta/hcom_length', 1, 'Datatype','int32'); h5write(outFile,'/meta/hcom_length', int32(hcom_length));
h5create(outFile, '/meta/xDim', 1, 'Datatype','int32'); h5write(outFile,'/meta/xDim', int32(xDim));
h5create(outFile, '/meta/fixed_phase', 1, 'Datatype','int32'); h5write(outFile,'/meta/fixed_phase', int32(1));

%% Modulators
M = 2;
pskmodObj = comm.PSKModulator(M, 'BitInput', true);

% Fixed pilot
pilot_bits = randi([0 1], Nc, 1);
Fpilot = pskmodObj(pilot_bits);

%% FIXED CHANNEL PARAMETERS (SAME AS ISAC_anh_Phong.m line 106)
% hcom = db2mag([0 -9.7 -19.2 -22.8]).*exp(1j*[0 -.8 1.6 -2.6]).*com_power_sqrt
hcom_magnitude = db2mag([0 -9.7 -19.2 -22.8]).';  % Fixed magnitude from PDP
hcom_phase = [0 -0.8 1.6 -2.6].';                  % Fixed phase (EXACTLY as original)

%% Generate samples
fprintf('Generating %d samples with FIXED PHASE channel...\n', numSamples);
fprintf('Channel: hcom = db2mag([0 -9.7 -19.2 -22.8]).*exp(1j*[0 -0.8 1.6 -2.6]).*power\n\n');
tic;

for idx = 1:numSamples
    if mod(idx, 100) == 0 || idx == 1
        fprintf('Sample %d / %d\n', idx, numSamples);
    end

    % Radar Tx
    bitsPerSymbol = 1;
    qam = 2^bitsPerSymbol;
    data = randi([0 qam-1], Nc, Ns);
    Ftx_radar = qammod(data, qam, 'UnitAveragePower', true);

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

    % Communication power (random 20-30 dB)
    com_power = randi([20 30], 1, 1);
    com_power_sqrt = sqrt(10^(com_power / 10));

    % ========== FIXED PHASE CHANNEL (EXACTLY AS ORIGINAL) ==========
    % Line 106 of ISAC_anh_Phong.m:
    % hcom = db2mag([0 -9.7 -19.2 -22.8]).*exp(1j*[0 -.8 1.6 -2.6]).*com_power_sqrt
    hcom = hcom_magnitude .* exp(1j * hcom_phase) .* com_power_sqrt;
    % ================================================================

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

    % LS estimate from pilots
    e_td = DFT_matrix' * E;
    x_pilot_td = x_com_td(:, 1:Np_pilot);
    y_pilot_td = e_td(:, 1:Np_pilot);

    h_ls = ls_fir_from_pilot(reshape(x_pilot_td,[],1), reshape(y_pilot_td,[],1), hcom_length);

    % Feature extraction
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

    x_in = single([band_mean; band_std; band_max; g]);

    % Labels
    y_out = single([real(hcom); imag(hcom)]);
    hrls_out = single([real(h_ls); imag(h_ls)]);

    % Write
    h5write(outFile, '/X', x_in.', [idx 1], [1 xDim]);
    h5write(outFile, '/Y', y_out.', [idx 1], [1 yDim]);
    h5write(outFile, '/Hrls', hrls_out.', [idx 1], [1 yDim]);
    h5write(outFile, '/norm_factor', single(norm_factor), [idx 1], [1 1]);
end

elapsed = toc;
fprintf('\nDone! Generated %d samples in %.1f seconds\n', numSamples, elapsed);
fprintf('Saved to: %s\n', outFile);
fprintf('\nChannel model: FIXED PHASE (matching ISAC_anh_Phong.m exactly)\n');

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
