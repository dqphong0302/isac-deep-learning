#!/usr/bin/env python3
"""
training_v4_raw_iq_cnn.py
V4: Truly Blind Channel Estimation using Raw IQ + CNN

Architecture:
- Input: (batch, 2, 256, 128) - Real/Imag channels
- Conv2D layers with BatchNorm and ReLU
- Global Average Pooling
- Dense layers to predict h_com (8 outputs)
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


class ChannelEstimationCNN(nn.Module):
    """CNN for blind channel estimation from raw IQ data"""
    
    def __init__(self, in_channels=2, base_filters=32):
        super().__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv2d(in_channels, base_filters, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(base_filters)
        
        self.conv2 = nn.Conv2d(base_filters, base_filters*2, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(base_filters*2)
        
        self.conv3 = nn.Conv2d(base_filters*2, base_filters*4, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(base_filters*4)
        
        self.conv4 = nn.Conv2d(base_filters*4, base_filters*8, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(base_filters*8)
        
        self.conv5 = nn.Conv2d(base_filters*8, base_filters*16, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(base_filters*16)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Regression head
        self.fc1 = nn.Linear(base_filters*16, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 8)  # 4 complex = 8 real
        
    def forward(self, x):
        # x: (batch, 2, H, W)
        x = F.silu(self.bn1(self.conv1(x)))
        x = F.silu(self.bn2(self.conv2(x)))
        x = F.silu(self.bn3(self.conv3(x)))
        x = F.silu(self.bn4(self.conv4(x)))
        x = F.silu(self.bn5(self.conv5(x)))
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        x = F.silu(self.fc1(x))
        x = self.dropout(x)
        x = F.silu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class RawIQDataset(Dataset):
    """Dataset for raw IQ CNN training"""
    
    def __init__(self, h5_path: str):
        with h5py.File(h5_path, 'r') as f:
            X = f['/X'][:]  # MATLAB saves as (2, Ns_ds, Nc_ds, samples) due to column-major
            Y = f['/Y'][:]  # (8, samples)
            
            print(f"Raw X shape from file: {X.shape}")
            print(f"Raw Y shape from file: {Y.shape}")
            
            # Handle MATLAB column-major: transpose to (samples, 2, Nc_ds, Ns_ds)
            # MATLAB saves (2, Ns_ds, Nc_ds, samples) -> need (samples, 2, Nc_ds, Ns_ds)
            if X.shape[0] == 2:  # First dim is channels (2) -> need to transpose
                X = np.transpose(X, (3, 0, 2, 1))  # (samples, 2, Nc_ds, Ns_ds)
            elif X.shape[-1] == 2:  # Last dim is channels
                X = np.transpose(X, (0, 3, 1, 2))  # (samples, 2, H, W)
            
            # Handle Y
            if Y.shape[0] == 8:  # (8, samples) -> (samples, 8)
                Y = Y.T
            
            self.X = X.astype(np.float32)
            self.Y = Y.astype(np.float32)
            self.n_samples = self.X.shape[0]
            
            print(f"Dataset: {self.n_samples} samples, X shape: {self.X.shape}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])


def nmse_loss(yhat, y, eps=1e-12):
    err = torch.sum((yhat - y) ** 2, dim=1)
    den = torch.sum(y ** 2, dim=1) + eps
    return torch.mean(err / den)


def train_epoch(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            yhat = model(x)
            loss = nmse_loss(yhat, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_nmse = 0.0
    n_batches = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yhat = model(x)
        total_nmse += nmse_loss(yhat, y).item()
        n_batches += 1
    
    return total_nmse / n_batches


def main():
    parser = argparse.ArgumentParser(description="V4: Raw IQ CNN Training")
    parser.add_argument("--h5", default="dataset_v4_raw_iq.h5")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--filters", type=int, default=32, help="Base filter count")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ckpt", type=str, default="best_v4_cnn.pt")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if not os.path.exists(args.h5):
        print(f"ERROR: Dataset not found: {args.h5}")
        sys.exit(1)
    
    dataset = RawIQDataset(args.h5)
    
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    
    print(f"Train: {n_train}, Val: {n_val}")
    
    model = ChannelEstimationCNN(in_channels=2, base_filters=args.filters).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: ChannelEstimationCNN, {n_params:,} parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_nmse = float('inf')
    
    print("\n" + "="*60)
    print("Training V4: Raw IQ CNN (Truly Blind)")
    print("="*60 + "\n")
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
        val_nmse = evaluate(model, val_loader, device)
        scheduler.step()
        
        train_db = 10 * np.log10(train_loss + 1e-12)
        val_db = 10 * np.log10(val_nmse + 1e-12)
        
        saved = ""
        if val_nmse < best_val_nmse:
            best_val_nmse = val_nmse
            torch.save({
                'model_state_dict': model.state_dict(),
                'base_filters': args.filters,
                'val_nmse': val_nmse,
                'epoch': epoch,
            }, args.ckpt)
            saved = " *"
        
        print(f"[ep {epoch:03d}] train: {train_db:.2f} dB | val: {val_db:.2f} dB{saved}")
    
    best_db = 10 * np.log10(best_val_nmse + 1e-12)
    print("\n" + "="*60)
    print(f"Training complete. Best val NMSE = {best_db:.2f} dB")
    print(f"Checkpoint: {args.ckpt}")
    print("="*60)


if __name__ == "__main__":
    main()
