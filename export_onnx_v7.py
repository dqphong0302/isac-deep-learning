#!/usr/bin/env python3
"""
export_onnx_v7.py
Export V7 Hybrid model to ONNX for MATLAB inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


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
        return h_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="best_v7_hybrid.pt")
    parser.add_argument("--out", default="v7_hybrid.onnx")
    args = parser.parse_args()
    
    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    spectral_dim = ckpt.get('spectral_dim', 210)
    
    model = HybridModel(spectral_dim=spectral_dim)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"Loaded model, alpha={model.alpha.item():.4f}")
    
    # Dummy inputs
    x_spectral = torch.randn(1, spectral_dim)
    x_2d = torch.randn(1, 2, 64, 64)
    h_rls = torch.randn(1, 8)
    
    # Export
    torch.onnx.export(
        model,
        (x_spectral, x_2d, h_rls),
        args.out,
        input_names=['x_spectral', 'x_2d', 'h_rls'],
        output_names=['h_pred'],
        opset_version=14,
        dynamic_axes={
            'x_spectral': {0: 'batch'},
            'x_2d': {0: 'batch'},
            'h_rls': {0: 'batch'},
            'h_pred': {0: 'batch'}
        }
    )
    
    print(f"Exported to {args.out}")
    

if __name__ == "__main__":
    main()
