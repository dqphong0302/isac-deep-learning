#!/usr/bin/env python3
"""Export V2/V3.6 model to ONNX"""

import argparse
import torch
import torch.nn as nn

class ResidualMLP(nn.Module):
    def __init__(self, in_dim=210, width=256, depth=3, dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, width))
            layers.append(nn.LayerNorm(width))
            layers.append(nn.SiLU(inplace=True))
            # Include dropout to match training architecture for state_dict loading
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = width
        layers.append(nn.Linear(d, 8))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        h_rls = x[:, -8:]
        delta = self.net(x)
        return h_rls + delta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="best_v2.pt")
    parser.add_argument("--out", default="v2.onnx")
    args = parser.parse_args()
    
    ckpt = torch.load(args.ckpt, map_location='cpu')
    
    in_dim = ckpt.get('in_dim', 210)
    width = ckpt.get('width', 256)
    depth = ckpt.get('depth', 3)
    
    # Try to detect if dropout was used (not explicitly saved in V2 checkpoint, but usually 0.0)
    # V3.6 Optimized saves architecture? 
    # The training script saves: 'in_dim', 'width', 'depth', 'val_nmse'.
    # It does NOT save 'dropout'.
    # However, V3.6 uses dropout=0.2.
    # If we don't pass dropout=0.2 here, the layer indices wont match.
    # Heuristic: Try to load with dropout=0.0 (V2). If fails, try dropout=0.2 (V3.6).
    
    print(f"Loading {args.ckpt}...")
    state_dict = ckpt['model_state_dict']
    
    # Attempt 1: Standard (V2)
    try:
        model = ResidualMLP(in_dim, width, depth, dropout=0.0)
        model.load_state_dict(state_dict, strict=True)
        print("Loaded architecture: Dropout=0.0")
    except RuntimeError:
        print("Mismatch with Dropout=0.0, trying Dropout=0.2 (V3.6 Opt)...")
        # Attempt 2: Optimized (V3.6)
        model = ResidualMLP(in_dim, width, depth, dropout=0.2)
        model.load_state_dict(state_dict, strict=True)
        print("Loaded architecture: Dropout=0.2")

    model.eval()
    dummy = torch.randn(1, in_dim)
    
    torch.onnx.export(
        model, dummy, args.out,
        input_names=['x'],
        output_names=['h8_pred'],
        dynamic_axes={'x': {0: 'batch'}, 'h8_pred': {0: 'batch'}},
        opset_version=17,
    )
    
    print(f"Exported: {args.out}")
    print(f"  Config: in={in_dim}, w={width}, d={depth}")

if __name__ == "__main__":
    main()
