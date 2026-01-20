#!/usr/bin/env python3
"""
evaluate_v7_test.py
Quick test evaluation for V7 Hybrid model
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class V36MLP(nn.Module):
    def __init__(self, in_dim=210, width=512, depth=4, dropout=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, width))
            layers.append(nn.LayerNorm(width))
            layers.append(nn.SiLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = width
        layers.append(nn.Linear(d, 8))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, h_rls):
        delta = self.net(x)
        return h_rls + delta


class CNNRefiner(nn.Module):
    def __init__(self, in_channels=2, base_filters=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_filters, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.conv2 = nn.Conv2d(base_filters, base_filters*2, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters*2)
        self.conv3 = nn.Conv2d(base_filters*2, base_filters*4, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(base_filters*4)
        self.conv4 = nn.Conv2d(base_filters*4, base_filters*8, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(base_filters*8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_filters*8, 8)
    
    def forward(self, x):
        x = F.silu(self.bn1(self.conv1(x)))
        x = F.silu(self.bn2(self.conv2(x)))
        x = F.silu(self.bn3(self.conv3(x)))
        x = F.silu(self.bn4(self.conv4(x)))
        x = self.gap(x).view(x.size(0), -1)
        return self.fc(x)


class HybridModel(nn.Module):
    def __init__(self, spectral_dim=210):
        super().__init__()
        self.mlp = V36MLP(in_dim=spectral_dim)
        self.cnn = CNNRefiner()
        self.alpha = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x_spectral, x_2d, h_rls):
        h_mlp = self.mlp(x_spectral, h_rls)
        delta_cnn = self.cnn(x_2d)
        h_final = h_mlp + self.alpha * delta_cnn
        return h_final, h_mlp, delta_cnn


def nmse(yhat, y):
    err = np.sum((yhat - y) ** 2, axis=1)
    den = np.sum(y ** 2, axis=1) + 1e-12
    return np.mean(err / den)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    ckpt = torch.load("best_v7_hybrid.pt", map_location=device)
    model = HybridModel(spectral_dim=ckpt['spectral_dim']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded model, alpha={model.alpha.item():.4f}")
    
    # Load test data (use validation split as test)
    with h5py.File("dataset_v7_hybrid.h5", 'r') as f:
        X_spectral = f['/X_spectral'][:]
        X_2d = f['/X_2d'][:]
        Y = f['/Y'][:]
    
    # Handle dimensions
    if X_spectral.shape[0] < X_spectral.shape[1]:
        X_spectral = X_spectral.T
    if Y.shape[0] < Y.shape[1]:
        Y = Y.T
    
    # X_2d transpose
    if X_2d.shape[0] == 2:
        X_2d = np.transpose(X_2d, (3, 0, 2, 1))
    
    # Use last 20% as test
    n = X_spectral.shape[0]
    n_test = int(n * 0.2)
    
    X_spec_test = X_spectral[-n_test:].astype(np.float32)
    X_2d_test = X_2d[-n_test:].astype(np.float32)
    Y_test = Y[-n_test:].astype(np.float32)
    H_rls_test = X_spec_test[:, -8:]
    
    print(f"Test samples: {n_test}")
    
    # Inference
    with torch.no_grad():
        x_spec = torch.from_numpy(X_spec_test).to(device)
        x_2d = torch.from_numpy(X_2d_test).to(device)
        h_rls = torch.from_numpy(H_rls_test).to(device)
        
        h_final, h_mlp, _ = model(x_spec, x_2d, h_rls)
        
        h_final = h_final.cpu().numpy()
        h_mlp = h_mlp.cpu().numpy()
    
    # Calculate NMSE
    nmse_final = nmse(h_final, Y_test)
    nmse_mlp = nmse(h_mlp, Y_test)
    nmse_rls = nmse(H_rls_test, Y_test)
    
    print("\n" + "="*50)
    print("TEST NMSE Results:")
    print("="*50)
    print(f"  h_rls (Ridge LS):     {10*np.log10(nmse_rls):.2f} dB")
    print(f"  MLP only:             {10*np.log10(nmse_mlp):.2f} dB")
    print(f"  Hybrid (MLP + CNN):   {10*np.log10(nmse_final):.2f} dB")
    print("="*50)


if __name__ == "__main__":
    main()
