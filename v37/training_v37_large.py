#!/usr/bin/env python3
"""
training_v37_large.py
V3.7: Larger Model + More Data

Changes from V3.6:
- Width: 512 -> 1024
- Depth: 4 -> 6
- LR: 1e-3 -> 5e-4
- Epochs: 100 -> 200
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


class ResidualMLPLarge(nn.Module):
    """Larger ResidualMLP for V3.7"""
    
    def __init__(self, in_dim=210, width=1024, depth=6, dropout=0.2):
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


class V37Dataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            X = f['/X'][:]
            Y = f['/Y'][:]
            if X.shape[0] < X.shape[1]: X = X.T
            if Y.shape[0] < Y.shape[1]: Y = Y.T
            
            self.X = X.astype(np.float32)
            self.Y = Y.astype(np.float32)
            self.Hrls = self.X[:, -8:]
            self.n_samples = self.X.shape[0]
            print(f"Dataset: {self.n_samples} samples, {self.X.shape[1]} features")

    def __len__(self): return self.n_samples
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i]), torch.from_numpy(self.Hrls[i])


def nmse_loss(yhat, y, eps=1e-12):
    err = torch.sum((yhat - y) ** 2, dim=1)
    den = torch.sum(y ** 2, dim=1) + eps
    return torch.mean(err / den)


def aux_mag_loss(yhat, y):
    h_real, h_imag = yhat[:, :4], yhat[:, 4:]
    mag_hat = torch.sqrt(h_real**2 + h_imag**2 + 1e-8)
    y_real, y_imag = y[:, :4], y[:, 4:]
    mag_true = torch.sqrt(y_real**2 + y_imag**2 + 1e-8)
    return F.mse_loss(mag_hat, mag_true)


def train_epoch(model, loader, optimizer, device, scaler, alpha=0.5):
    model.train()
    total_nmse = 0.0
    n = 0
    for x, y, h_rls in loader:
        x, y, h_rls = x.to(device), y.to(device), h_rls.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            yhat = model(x, h_rls)
            nmse = nmse_loss(yhat, y)
            mag = aux_mag_loss(yhat, y)
            loss = nmse + alpha * mag
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_nmse += nmse.item()
        n += 1
    return total_nmse / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    for x, y, h_rls in loader:
        x, y, h_rls = x.to(device), y.to(device), h_rls.to(device)
        yhat = model(x, h_rls)
        total += nmse_loss(yhat, y).item()
        n += 1
    return total / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="dataset_v37_20k.h5")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ckpt", default="best_v37.pt")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if not os.path.exists(args.h5):
        print(f"ERROR: {args.h5} not found")
        sys.exit(1)
    
    ds = V37Dataset(args.h5)
    n_val = int(len(ds) * 0.2)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    
    print(f"Train: {n_train}, Val: {n_val}")
    
    model = ResidualMLPLarge(width=args.width, depth=args.depth, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: ResidualMLPLarge, {n_params:,} params")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    
    best_nmse = float('inf')
    
    print("\n" + "="*60)
    print("Training V3.7: Larger Model + More Data")
    print("="*60 + "\n")
    
    for ep in range(1, args.epochs+1):
        train_nmse = train_epoch(model, train_loader, optimizer, device, scaler)
        val_nmse = evaluate(model, val_loader, device)
        scheduler.step()
        
        train_db = 10*np.log10(train_nmse + 1e-12)
        val_db = 10*np.log10(val_nmse + 1e-12)
        
        saved = ""
        if val_nmse < best_nmse:
            best_nmse = val_nmse
            torch.save({
                'model_state_dict': model.state_dict(),
                'width': args.width, 'depth': args.depth,
                'val_nmse': val_nmse
            }, args.ckpt)
            saved = " *"
        
        if ep % 10 == 0 or ep == 1 or saved:
            print(f"[ep {ep:03d}] train: {train_db:.2f} dB | val: {val_db:.2f} dB{saved}")
    
    print(f"\nBest Val NMSE: {10*np.log10(best_nmse):.2f} dB")
    print(f"Checkpoint: {args.ckpt}")


if __name__ == "__main__":
    main()
