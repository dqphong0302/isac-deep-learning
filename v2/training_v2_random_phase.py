#!/usr/bin/env python3
"""
training_v2_random_phase.py
V2: Semi-Blind with Random Phase

Uses h_ls features (210 input) but with random phase channel.
Based on V1 training script but expects random phase data.
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


class ResidualMLP(nn.Module):
    """Same as V1 - residual learning with h_ls"""
    def __init__(self, in_dim: int = 210, width: int = 256, depth: int = 3, dropout: float = 0.0):
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
        layers.append(nn.Linear(d, 8))  # delta_h
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, h_rls: torch.Tensor) -> torch.Tensor:
        delta = self.net(x)
        return h_rls + delta


class V2Dataset(Dataset):
    """Dataset with h_ls features and random phase channel"""
    def __init__(self, h5_path: str, cache_in_mem: bool = True):
        self.h5_path = h5_path
        
        with h5py.File(h5_path, 'r') as f:
            X_raw = f['/X'][:]
            Y_raw = f['/Y'][:]
            Hrls_raw = f['/Hrls'][:] if '/Hrls' in f else None
            
            # Handle MATLAB column-major format
            if X_raw.shape[0] < X_raw.shape[1]:
                X_raw = X_raw.T
            if Y_raw.shape[0] < Y_raw.shape[1]:
                Y_raw = Y_raw.T
            if Hrls_raw is not None and Hrls_raw.shape[0] < Hrls_raw.shape[1]:
                Hrls_raw = Hrls_raw.T
            
            self.n_samples = X_raw.shape[0]
            self.x_dim = X_raw.shape[1]
            
            print(f"Dataset loaded: {self.n_samples} samples, {self.x_dim} features")
            
            self.X = X_raw.astype(np.float32)
            self.Y = Y_raw.astype(np.float32)
            # h_rls is the last 8 features of X
            self.Hrls = self.X[:, -8:] if Hrls_raw is None else Hrls_raw.astype(np.float32)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i]), torch.from_numpy(self.Hrls[i])


def nmse_loss(yhat, y, eps=1e-12):
    err = torch.sum((yhat - y) ** 2, dim=1)
    den = torch.sum(y ** 2, dim=1) + eps
    return torch.mean(err / den)


def train_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for x, y, h_rls in loader:
        x, y, h_rls = x.to(device), y.to(device), h_rls.to(device)
        optimizer.zero_grad()
        
        if scaler:
            with torch.amp.autocast('cuda'):
                yhat = model(x, h_rls)
                loss = nmse_loss(yhat, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            yhat = model(x, h_rls)
            loss = nmse_loss(yhat, y)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_nmse = 0.0
    n_batches = 0
    
    for x, y, h_rls in loader:
        x, y, h_rls = x.to(device), y.to(device), h_rls.to(device)
        yhat = model(x, h_rls)
        nmse = nmse_loss(yhat, y)
        total_nmse += nmse.item()
        n_batches += 1
    
    return total_nmse / n_batches


def main():
    parser = argparse.ArgumentParser(description="V2: Semi-Blind Training (Random Phase)")
    parser.add_argument("--h5", type=str, default="dataset_v2_random_phase.h5", help="HDF5 dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--width", type=int, default=256, help="Hidden layer width")
    parser.add_argument("--depth", type=int, default=3, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--ckpt", type=str, default="best_v2.pt", help="Checkpoint path")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if not os.path.exists(args.h5):
        print(f"ERROR: Dataset not found: {args.h5}")
        sys.exit(1)
    
    full_dataset = V2Dataset(args.h5)
    
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    
    print(f"Train: {n_train}, Val: {n_val}")
    
    in_dim = full_dataset.x_dim
    model = ResidualMLP(in_dim=in_dim, width=args.width, depth=args.depth, dropout=args.dropout)
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: ResidualMLP, {n_params:,} parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    best_val_nmse = float('inf')
    
    print("\n" + "="*60)
    print("Training V2: Semi-Blind with Random Phase")
    print("="*60 + "\n")
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        val_nmse = evaluate(model, val_loader, device)
        
        scheduler.step()
        
        val_db = 10 * np.log10(val_nmse + 1e-12)
        train_db = 10 * np.log10(train_loss + 1e-12)
        
        saved = ""
        if val_nmse < best_val_nmse:
            best_val_nmse = val_nmse
            torch.save({
                'model_state_dict': model.state_dict(),
                'in_dim': in_dim,
                'width': args.width,
                'depth': args.depth,
                'val_nmse': val_nmse,
                'epoch': epoch,
            }, args.ckpt)
            saved = "  *"
        
        print(f"[ep {epoch:03d}] train: {train_db:.2f} dB | val: {val_db:.2f} dB{saved}")
    
    best_db = 10 * np.log10(best_val_nmse + 1e-12)
    print("\n" + "="*60)
    print(f"Training complete. Best val NMSE = {best_db:.2f} dB")
    print(f"Checkpoint saved to: {args.ckpt}")
    print("="*60)


if __name__ == "__main__":
    main()
