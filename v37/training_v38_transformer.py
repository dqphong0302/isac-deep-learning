#!/usr/bin/env python3
"""
training_v38_transformer.py
V3.8: Transformer-based Channel Estimation

Uses self-attention to learn complex relationships between features.
"""

import argparse
import os
import sys
import math

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerChannelEstimator(nn.Module):
    """Transformer for channel estimation"""
    
    def __init__(self, in_dim=210, d_model=128, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(in_dim, d_model)
        
        # Positional encoding (treat features as sequence)
        self.pos_enc = PositionalEncoding(d_model)
        
        # Reshape input to sequence: (batch, 210) -> (batch, seq_len, d_model)
        self.seq_len = 14  # 210 / 15 = 14 groups of 15 features
        self.feat_proj = nn.Linear(15, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model * self.seq_len, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 8)
    
    def forward(self, x, h_rls):
        # x: (batch, 210)
        batch_size = x.size(0)
        
        # Reshape to sequence: (batch, 14, 15)
        x = x.view(batch_size, self.seq_len, -1)
        
        # Project to d_model
        x = self.feat_proj(x)  # (batch, 14, d_model)
        
        # Add positional encoding
        x = self.pos_enc(x)
        
        # Transformer
        x = self.transformer(x)  # (batch, 14, d_model)
        
        # Flatten
        x = x.view(batch_size, -1)  # (batch, 14*d_model)
        
        # Output
        x = F.silu(self.fc1(x))
        x = self.dropout(x)
        x = F.silu(self.fc2(x))
        delta = self.fc3(x)
        
        # Residual learning
        return h_rls + delta


class TransformerDataset(Dataset):
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
            print(f"Dataset: {self.n_samples} samples")

    def __len__(self): return self.n_samples
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i]), torch.from_numpy(self.Hrls[i])


def nmse_loss(yhat, y, eps=1e-12):
    err = torch.sum((yhat - y) ** 2, dim=1)
    den = torch.sum(y ** 2, dim=1) + eps
    return torch.mean(err / den)


def train_epoch(model, loader, optimizer, device, scaler):
    model.train()
    total = 0.0
    n = 0
    for x, y, h_rls in loader:
        x, y, h_rls = x.to(device), y.to(device), h_rls.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            yhat = model(x, h_rls)
            loss = nmse_loss(yhat, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        n += 1
    return total / n


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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ckpt", default="best_v38_transformer.pt")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if not os.path.exists(args.h5):
        print(f"ERROR: {args.h5} not found")
        sys.exit(1)
    
    ds = TransformerDataset(args.h5)
    n_val = int(len(ds) * 0.2)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    
    print(f"Train: {n_train}, Val: {n_val}")
    
    model = TransformerChannelEstimator(
        d_model=args.d_model, nhead=args.nhead, num_layers=args.layers
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Transformer, {n_params:,} params")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    
    best_nmse = float('inf')
    
    print("\n" + "="*60)
    print("Training V3.8: Transformer Channel Estimator")
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
                'd_model': args.d_model, 'nhead': args.nhead, 'layers': args.layers,
                'val_nmse': val_nmse
            }, args.ckpt)
            saved = " *"
        
        if ep % 10 == 0 or ep == 1 or saved:
            print(f"[ep {ep:03d}] train: {train_db:.2f} dB | val: {val_db:.2f} dB{saved}")
    
    print(f"\nBest Val NMSE: {10*np.log10(best_nmse):.2f} dB")
    print(f"Checkpoint: {args.ckpt}")


if __name__ == "__main__":
    main()
