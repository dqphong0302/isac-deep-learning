#!/usr/bin/env python3
"""
training_blind_random_phase.py
V3: Truly Blind Channel Estimation Training

Key differences from V1:
  - Input: 202 features (no h_ls from pilot)
  - No residual learning (direct prediction)
  - Larger model for harder task
  - Random phase channel in dataset
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

# ============== Model ==============

class BlindMLP(nn.Module):
    """
    Truly blind channel estimator.
    No residual learning - predicts h_com directly from spectral features.
    """
    def __init__(self, in_dim: int = 202, width: int = 512, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        assert depth >= 1
        
        layers = []
        d = in_dim
        for i in range(depth):
            layers.append(nn.Linear(d, width))
            layers.append(nn.LayerNorm(width))
            layers.append(nn.SiLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = width
        
        # Output: 8 values (4 real + 4 imag)
        layers.append(nn.Linear(d, 8))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize output layer with small weights
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_dim) - spectral features only
        returns: (B, 8) - predicted channel [Re(h1-4), Im(h1-4)]
        """
        return self.net(x)


# ============== Dataset ==============

class BlindDataset(Dataset):
    """
    Dataset for blind channel estimation.
    Returns (x, y) where:
      x: (202,) float32 - spectral features (no h_ls)
      y: (8,) float32 - ground truth channel
    """
    def __init__(self, h5_path: str, cache_in_mem: bool = True):
        self.h5_path = h5_path
        self.cache_in_mem = cache_in_mem
        
        with h5py.File(h5_path, 'r') as f:
            X_raw = f['/X'][:]
            Y_raw = f['/Y'][:]
            
            # Handle MATLAB column-major format: (features, samples) -> (samples, features)
            if X_raw.shape[0] < X_raw.shape[1]:
                X_raw = X_raw.T
            if Y_raw.shape[0] < Y_raw.shape[1]:
                Y_raw = Y_raw.T
            
            self.n_samples = X_raw.shape[0]
            self.x_dim = X_raw.shape[1]
            
            print(f"Dataset loaded: {self.n_samples} samples, {self.x_dim} features")
            
            if cache_in_mem:
                self.X = X_raw.astype(np.float32)
                self.Y = Y_raw.astype(np.float32)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, i):
        if self.cache_in_mem:
            x = self.X[i]
            y = self.Y[i]
        else:
            with h5py.File(self.h5_path, 'r') as f:
                x = f['/X'][i].astype(np.float32)
                y = f['/Y'][i].astype(np.float32)
        
        return torch.from_numpy(x), torch.from_numpy(y)


# ============== Loss ==============

def nmse_loss(yhat: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalized MSE loss"""
    err = torch.sum((yhat - y) ** 2, dim=1)
    den = torch.sum(y ** 2, dim=1) + eps
    return torch.mean(err / den)


def nmse_db(yhat: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """NMSE in dB (for logging)"""
    nmse = nmse_loss(yhat, y, eps)
    return 10 * torch.log10(nmse + eps).item()


# ============== Training ==============

def train_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                yhat = model(x)
                loss = nmse_loss(yhat, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            yhat = model(x)
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
    total_mse = 0.0
    n_batches = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yhat = model(x)
        
        nmse = nmse_loss(yhat, y)
        mse = torch.mean((yhat - y) ** 2)
        
        total_nmse += nmse.item()
        total_mse += mse.item()
        n_batches += 1
    
    return total_nmse / n_batches, total_mse / n_batches


def main():
    parser = argparse.ArgumentParser(description="V3: Truly Blind Channel Estimation Training")
    parser.add_argument("--h5", type=str, default="dataset_blind_random_phase.h5", help="HDF5 dataset")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--width", type=int, default=512, help="Hidden layer width")
    parser.add_argument("--depth", type=int, default=4, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--ckpt", type=str, default="best_blind.pt", help="Checkpoint path")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Dataset
    if not os.path.exists(args.h5):
        print(f"ERROR: Dataset not found: {args.h5}")
        sys.exit(1)
    
    full_dataset = BlindDataset(args.h5, cache_in_mem=True)
    print(f"Dataset: {len(full_dataset)} samples, {full_dataset.x_dim} features")
    
    # Train/Val split
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], 
                                     generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, 
                            num_workers=0, pin_memory=True)
    
    print(f"Train: {n_train}, Val: {n_val}")
    
    # Model
    in_dim = full_dataset.x_dim
    model = BlindMLP(in_dim=in_dim, width=args.width, depth=args.depth, dropout=args.dropout)
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: BlindMLP, {n_params:,} parameters")
    print(f"  in_dim={in_dim}, width={args.width}, depth={args.depth}, dropout={args.dropout}")
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # AMP
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    # Training loop
    best_val_nmse = float('inf')
    
    print("\n" + "="*60)
    print("Training V3: Truly Blind Channel Estimation")
    print("="*60 + "\n")
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        val_nmse, val_mse = evaluate(model, val_loader, device)
        
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        val_nmse_db = 10 * np.log10(val_nmse + 1e-12)
        train_nmse_db = 10 * np.log10(train_loss + 1e-12)
        
        saved = ""
        if val_nmse < best_val_nmse:
            best_val_nmse = val_nmse
            torch.save({
                'model_state_dict': model.state_dict(),
                'in_dim': in_dim,
                'width': args.width,
                'depth': args.depth,
                'dropout': args.dropout,
                'val_nmse': val_nmse,
                'epoch': epoch,
            }, args.ckpt)
            saved = "  *"
        
        print(f"[ep {epoch:03d}] lr={lr:.2e} | train: {train_nmse_db:.2f} dB | val: {val_nmse_db:.2f} dB{saved}")
    
    best_nmse_db = 10 * np.log10(best_val_nmse + 1e-12)
    print("\n" + "="*60)
    print(f"Training complete. Best val NMSE = {best_nmse_db:.2f} dB")
    print(f"Checkpoint saved to: {args.ckpt}")
    print("="*60)


if __name__ == "__main__":
    main()
