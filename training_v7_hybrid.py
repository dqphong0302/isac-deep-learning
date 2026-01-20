#!/usr/bin/env python3
"""
training_v7_hybrid.py
V7: Hybrid Model - V3.6 MLP + CNN Residual Refinement

Architecture:
    1. V3.6 MLP: spectral features (210) → h_mlp (8)
    2. CNN Refiner: 2D E matrix (64x64x2) → delta_h (8)
    3. Final: h_pred = h_mlp + alpha * delta_h (alpha learnable)
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


class V36MLP(nn.Module):
    """V3.6 style MLP for spectral features"""
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
    """CNN to extract spatial patterns from 2D E matrix"""
    def __init__(self, in_channels=2, base_filters=32):
        super().__init__()
        # Input: (batch, 2, 64, 64)
        self.conv1 = nn.Conv2d(in_channels, base_filters, 5, stride=2, padding=2)  # -> 32x32
        self.bn1 = nn.BatchNorm2d(base_filters)
        
        self.conv2 = nn.Conv2d(base_filters, base_filters*2, 3, stride=2, padding=1)  # -> 16x16
        self.bn2 = nn.BatchNorm2d(base_filters*2)
        
        self.conv3 = nn.Conv2d(base_filters*2, base_filters*4, 3, stride=2, padding=1)  # -> 8x8
        self.bn3 = nn.BatchNorm2d(base_filters*4)
        
        self.conv4 = nn.Conv2d(base_filters*4, base_filters*8, 3, stride=2, padding=1)  # -> 4x4
        self.bn4 = nn.BatchNorm2d(base_filters*8)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_filters*8, 8)
    
    def forward(self, x):
        # x: (batch, 2, 64, 64)
        x = F.silu(self.bn1(self.conv1(x)))
        x = F.silu(self.bn2(self.conv2(x)))
        x = F.silu(self.bn3(self.conv3(x)))
        x = F.silu(self.bn4(self.conv4(x)))
        x = self.gap(x).view(x.size(0), -1)
        return self.fc(x)


class HybridModel(nn.Module):
    """V3.6 MLP + CNN Refiner with learnable alpha"""
    def __init__(self, spectral_dim=210):
        super().__init__()
        self.mlp = V36MLP(in_dim=spectral_dim)
        self.cnn = CNNRefiner()
        # Learnable fusion weight (initialized to 0.1)
        self.alpha = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x_spectral, x_2d, h_rls):
        h_mlp = self.mlp(x_spectral, h_rls)
        delta_cnn = self.cnn(x_2d)
        h_final = h_mlp + self.alpha * delta_cnn
        return h_final, h_mlp, delta_cnn


class HybridDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            X_spectral = f['/X_spectral'][:]
            X_2d = f['/X_2d'][:]
            Y = f['/Y'][:]
            
            # Handle MATLAB dimension order for spectral
            if X_spectral.shape[0] < X_spectral.shape[1]:
                X_spectral = X_spectral.T
            if Y.shape[0] < Y.shape[1]:
                Y = Y.T
            
            # Handle MATLAB dimension order for X_2d
            # MATLAB writes as (samples, H, W, 2) but in column-major
            # Need to check actual shape and transpose appropriately
            print(f"Raw X_2d shape: {X_2d.shape}")
            
            # If channels are last and samples first: (N, H, W, 2)
            if X_2d.ndim == 4:
                if X_2d.shape[-1] == 2:
                    # (N, H, W, 2) -> (N, 2, H, W)
                    X_2d = np.transpose(X_2d, (0, 3, 1, 2))
                elif X_2d.shape[1] == 2:
                    # Already (N, 2, H, W)
                    pass
                else:
                    # MATLAB column-major: need to figure out correct transpose
                    # Try: (2, W, H, N) -> (N, 2, H, W)
                    if X_2d.shape[0] == 2:
                        X_2d = np.transpose(X_2d, (3, 0, 2, 1))
                    else:
                        # (H, W, 2, N) -> (N, 2, H, W)
                        X_2d = np.transpose(X_2d, (3, 2, 0, 1))
            
            print(f"Transposed X_2d shape: {X_2d.shape}")
            
            self.X_spectral = X_spectral.astype(np.float32)
            self.X_2d = X_2d.astype(np.float32)
            self.Y = Y.astype(np.float32)
            # h_ls is last 8 dims of spectral features
            self.Hrls = self.X_spectral[:, -8:]
            
            self.n_samples = self.X_spectral.shape[0]
            self.spectral_dim = self.X_spectral.shape[1]
            print(f"Dataset: {self.n_samples} samples")
            print(f"  Spectral: {self.spectral_dim}")
            print(f"  2D shape per sample: {self.X_2d.shape[1:]}")

    def __len__(self): 
        return self.n_samples
    
    def __getitem__(self, i):
        return (torch.from_numpy(self.X_spectral[i]), 
                torch.from_numpy(self.X_2d[i]),
                torch.from_numpy(self.Y[i]), 
                torch.from_numpy(self.Hrls[i]))


def nmse_loss(yhat, y, eps=1e-12):
    err = torch.sum((yhat - y) ** 2, dim=1)
    den = torch.sum(y ** 2, dim=1) + eps
    return torch.mean(err / den)


def train_epoch(model, loader, optimizer, device, scaler):
    model.train()
    total_nmse = 0.0
    n = 0
    for x_spec, x_2d, y, h_rls in loader:
        x_spec = x_spec.to(device)
        x_2d = x_2d.to(device)
        y = y.to(device)
        h_rls = h_rls.to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            h_final, h_mlp, _ = model(x_spec, x_2d, h_rls)
            # Main loss on final output
            loss_final = nmse_loss(h_final, y)
            # Auxiliary loss on MLP output (to keep it stable)
            loss_mlp = nmse_loss(h_mlp, y)
            loss = loss_final + 0.3 * loss_mlp
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_nmse += loss_final.item()
        n += 1
    return total_nmse / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_final = 0.0
    total_mlp = 0.0
    n = 0
    for x_spec, x_2d, y, h_rls in loader:
        x_spec = x_spec.to(device)
        x_2d = x_2d.to(device)
        y = y.to(device)
        h_rls = h_rls.to(device)
        
        h_final, h_mlp, _ = model(x_spec, x_2d, h_rls)
        total_final += nmse_loss(h_final, y).item()
        total_mlp += nmse_loss(h_mlp, y).item()
        n += 1
    return total_final / n, total_mlp / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="dataset_v7_hybrid.h5")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ckpt", default="best_v7_hybrid.pt")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if not os.path.exists(args.h5):
        print(f"ERROR: {args.h5} not found")
        sys.exit(1)
    
    ds = HybridDataset(args.h5)
    spectral_dim = ds.spectral_dim
    
    n_val = int(len(ds) * 0.2)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    
    print(f"Train: {n_train}, Val: {n_val}")
    
    model = HybridModel(spectral_dim=spectral_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: HybridModel (MLP + CNN), {n_params:,} params")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    
    best_nmse = float('inf')
    
    print("\n" + "="*60)
    print("Training V7: Hybrid Model (V3.6 + CNN Refiner)")
    print("="*60 + "\n")
    
    for ep in range(1, args.epochs+1):
        train_nmse = train_epoch(model, train_loader, optimizer, device, scaler)
        val_nmse_final, val_nmse_mlp = evaluate(model, val_loader, device)
        scheduler.step()
        
        train_db = 10*np.log10(train_nmse + 1e-12)
        val_final_db = 10*np.log10(val_nmse_final + 1e-12)
        val_mlp_db = 10*np.log10(val_nmse_mlp + 1e-12)
        alpha_val = model.alpha.item()
        
        saved = ""
        if val_nmse_final < best_nmse:
            best_nmse = val_nmse_final
            torch.save({
                'model_state_dict': model.state_dict(),
                'spectral_dim': spectral_dim,
                'val_nmse': val_nmse_final,
                'alpha': alpha_val
            }, args.ckpt)
            saved = " *"
        
        print(f"[ep {ep:03d}] train: {train_db:.2f} dB | val: {val_final_db:.2f} dB (MLP: {val_mlp_db:.2f}) | α={alpha_val:.3f}{saved}")
    
    print(f"\nBest Val NMSE: {10*np.log10(best_nmse):.2f} dB")
    print(f"Final alpha: {model.alpha.item():.4f}")
    print(f"Checkpoint: {args.ckpt}")


if __name__ == "__main__":
    main()
