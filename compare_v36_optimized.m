% compare_v36_optimized.m
% -------------------------------------------------------------------------
% V3.6 OPTIMIZED: 4 Pilots - Optimized Comparison
%
% Optimization:
%   - Robust LS with Ridge Regularization (0.1)
%   - Fair GSFDIC (using all 4 pilots)
% -------------------------------------------------------------------------

clc; clear; close all;

addpath('./useful_function/');

%% Python setup
pe = pyenv;
try
    pymod = py.importlib.import_module("onnx_inference_helper");
    py.importlib.reload(pymod);
catch ME
    warning("Python module helper issue: %s", ME.message);
end
np = py.importlib.import_module("numpy");

%% Settings
onnxFile = "v36_opt.onnx";
if ~isfile(onnxFile), error('ONNX file not found: %s', onnxFile); end

Nmc = 100;
no_of_fb = 4;

% === KEY PARAMETERS FOR V3.6 ===
Np_pilot_dl = 4;        % V3.6 uses 4 pilots
N_pilot = Np_pilot_dl * 1024; % GSFDIC uses all 4 pilots (FAIR)
Ridge_Lambda = 0.1;     % Robust LS Regularization
% ===============================

%% System params
global Nc Ns fC c Delta_f Ts Te;
c = 3e8; Delta_f = 78.125e3; Nc = 1024; Ns = 512; fC = 5e9;
Te = 1/Delta_f; Ts = Te; delta_r = 1.1574; r_max = 2500; v_max = 75;

expandVector = @(v, m) v(:, ones(1, m));
n = 0:Nc-1; k = n';
DFT_matrix = 1/sqrt(Nc) * exp(-2i*pi*k*n/Nc);

hcom_length = 4;
M = 2;
pskmodObj = comm.PSKModulator(M, 'BitInput', true);
pskdemodObj = comm.PSKDemodulator(M, 'BitOutput', true);

B = 64; percentiles = [10 25 50 75 90];
hcom_magnitude = db2mag([0 -9.7 -19.2 -22.8]).';
pilot_bits = randi([0 1], Nc, 1);
Fpilot = pskmodObj(pilot_bits);

nmse_gsfdic = zeros(Nmc, 1);
nmse_v36 = zeros(Nmc, 1);

fprintf('\n========================================\n');
fprintf('V3.6 OPTIMIZED: 4 Pilots Comparison\n');
fprintf('Regularization: 0.1 (Robust)\n');
fprintf('Running %d Monte Carlo trials...\n', Nmc);
fprintf('========================================\n\n');
tic;

for it = 1:Nmc
    if mod(it, 20)==0, fprintf("Trial %d / %d\n", it, Nmc); end

    % Signal Gen
    data = randi([0 1], Nc, Ns);
    Ftx_radar = qammod(data, 2, 'UnitAveragePower', true);
    
    numTarget = randi([1 5], 1, 1); k_target=numTarget;
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
    for col = 1:Ns
        if col <= Np_pilot_dl
            Ftx_com(:, col) = Fpilot;
        else
            x_com_bit = randi([0 1], 1, log2(M) * Nc);
            Ftx_com(:, col) = pskmodObj(x_com_bit');
        end
    end
    x_com_td = DFT_matrix' * Ftx_com;
    
    com_power = randi([20 30], 1, 1);
    com_power_sqrt = sqrt(10^(com_power/10));
    random_phase = 2 * pi * rand(4, 1);
    hcom_true = hcom_magnitude .* exp(1j * random_phase) .* com_power_sqrt;

    x_com_td_vec = reshape(x_com_td, [], 1);
    y_com_td_vec = filter(hcom_true, 1, x_com_td_vec);
    Y_com = DFT_matrix * reshape(y_com_td_vec, Nc, Ns);

    A = DFT_matrix * (Y_target + Y_si + Y_com) + R_noise;

    % --- GSFDIC (Fair: Uses 4 pilots) ---
    hcom_est = zeros(hcom_length, 1);
    epsilon = 0.0001; P1 = (epsilon^-1) * eye(hcom_length);
    Y_target_rec = zeros(Nc, Ns); Y_com_rec = zeros(Nc, Ns); H_comp_mat = zeros(Nc, Ns);

    for fb = 1:no_of_fb
        Y_target_rec_prev = Y_target_rec; Y_com_rec_prev = Y_com_rec;
        A_tilde = A - Y_com_rec_prev - Y_target_rec_prev;
        if fb > 1
             H_comp = -1/(Ns-1) * ((A_tilde(:,1)./Ftx_radar(:,1)) - (A_tilde(:,end)./Ftx_radar(:,end)));
             H_comp_mat = expandVector(H_comp, Ns);
        else, H_comp_mat = zeros(Nc, Ns); end

        E_si_suppress_tilde = fdic(A_tilde, Ftx_radar, H_comp_mat);
        RDM = calc_periodogram(E_si_suppress_tilde + Y_target_rec_prev, Ftx_radar, 1/100, 1/100);
        targetout = cfar_ofdm_radar(RDM, k_target, 1, 3, 3, 3, 3);
        Y_target_rec = DFT_matrix * generate_target_echo(target_power, targetout.velocities, targetout.ranges, Ftx_radar);

        E_tilde = E_si_suppress_tilde + Y_target_rec_prev - Y_target_rec + Y_com_rec_prev;
        
        x_com_pilot = x_com_td_vec(1:N_pilot); % USES ALL 4 PILOTS
        E_td = DFT_matrix' * E_tilde;
        Ep_td = reshape(E_td, [], 1); Ep_td = Ep_td(1:N_pilot);

        for m = hcom_length:length(x_com_pilot)
            [hcom_est, P1] = RLS_function(hcom_est, Ep_td(m), x_com_pilot(m:-1:m-(hcom_length-1)), P1);
        end
        % Reconstruct Com... (omitted detailed reconstruction logic for brevity if not needed for h_est, but necessary for next iter)
        Ec_td = reshape(DFT_matrix' * E_tilde, [], 1);
        Ec = DFT_matrix * reshape(filter(1, hcom_est', Ec_td), Nc, Ns);
        
        % Demod/Mod column by column (PSK objects require column vectors)
        Ftx_com_est = zeros(Nc, Ns);
        for kk = 1:Ns
            bits_est = pskdemodObj(Ec(:, kk));
            Ftx_com_est(:, kk) = pskmodObj(bits_est);
        end
        % Keep pilots unchanged
        for kk = 1:Np_pilot_dl
            Ftx_com_est(:, kk) = Fpilot;
        end
        
        x_com_est = reshape(DFT_matrix' * Ftx_com_est, [], 1);
        Y_com_rec = DFT_matrix * reshape(filter(hcom_est', 1, x_com_est), Nc, Ns);
    end
    nmse_gsfdic(it) = 10*log10(norm(hcom_est - hcom_true)^2 / (norm(hcom_true)^2 + eps));

    % --- V3.6 DL (Robust LS) ---
    E_dl = fdic(A, Ftx_radar, zeros(Nc, Ns)); 
    e_td = DFT_matrix' * E_dl;
    x_pilot_td = x_com_td(:, 1:Np_pilot_dl);
    y_pilot_td = e_td(:, 1:Np_pilot_dl);
    
    % ROBUST LS (Lambda = 0.1)
    h_ls = ls_fir_from_pilot_robust(reshape(x_pilot_td,[],1), reshape(y_pilot_td,[],1), hcom_length, Ridge_Lambda);

    norm_factor = sqrt(mean(abs(E_dl(:)).^2)) + eps;
    E_n = E_dl ./ norm_factor;
    logP = log10(mean(abs(E_n).^2, 2) + 1e-12);
    
    bandSize = Nc/B;
    band_mean = arrayfun(@(b) mean(logP((b-1)*bandSize+1:b*bandSize)), 1:B)';
    band_std = arrayfun(@(b) std(logP((b-1)*bandSize+1:b*bandSize)), 1:B)';
    band_max = arrayfun(@(b) max(logP((b-1)*bandSize+1:b*bandSize)), 1:B)';
    g_p = prctile(logP, percentiles);
    g = [mean(logP); std(logP); max(logP); min(logP); g_p(:)];
    
    x_vec = single([band_mean; band_std; band_max; g; log10(norm_factor); single([real(h_ls(:)); imag(h_ls(:))])]);
    
    y_list = pymod.predict_hcom(np.array(x_vec.', pyargs('dtype', np.float32)), char(onnxFile));
    yhat8 = double(cell2mat(cell(y_list))); yhat8 = yhat8(:);
    hcom_v36 = yhat8(1:4) + 1j*yhat8(5:8);
    
    nmse_v36(it) = 10*log10(norm(hcom_v36 - hcom_true)^2 / (norm(hcom_true)^2 + eps));
end

elapsed = toc;
fprintf('\nCompleted %d trials in %.1f s\n', Nmc, elapsed);
fprintf('Mean NMSE GSFDIC (4 pilots): %.2f dB\n', mean(nmse_gsfdic));
fprintf('Mean NMSE V3.6 Opt (4 pilots): %.2f dB\n', mean(nmse_v36));
fprintf('Impovement: %.2f dB\n', mean(nmse_gsfdic) - mean(nmse_v36));

figFolder = 'figures_v36';
if ~isfolder(figFolder); mkdir(figFolder); end
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
fig1 = figure('Visible','off'); 
plot(sort(nmse_gsfdic), (1:Nmc)/Nmc, 'b-', 'LineWidth', 2); hold on;
plot(sort(nmse_v36), (1:Nmc)/Nmc, 'r--', 'LineWidth', 2);
legend('GSFDIC (4 pilots)', 'V3.6 Opt (4 pilots)'); grid on; xlabel('NMSE (dB)'); ylabel('CDF');
saveas(fig1, fullfile(figFolder, ['cdf_v36_' timestamp '.png']));

function h_hat = ls_fir_from_pilot_robust(x, y, L, lambda)
    x = x(:); y = y(:);
    N = length(x);
    rows = N - L + 1;
    Xc = zeros(rows, L);
    for kk = 1:rows, Xc(kk, :) = x(kk+L-1:-1:kk).'; end
    yv = y(L:end);
    h_hat = (Xc' * Xc + lambda * eye(L)) \ (Xc' * yv);
end

function [h_est, P] = RLS_function(h_old, e, u, P_old)
    lambda = 0.99;
    k = (P_old * u) / (lambda + u' * P_old * u);
    xi = e - u' * h_old;
    h_est = h_old + k * xi;
    P = (P_old - k * u' * P_old) / lambda;
end
