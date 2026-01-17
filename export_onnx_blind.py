#!/usr/bin/env python3
"""
export_onnx_blind.py
Export V3 Truly Blind model to ONNX format
"""

import argparse
import torch
import torch.nn as nn


class BlindMLP(nn.Module):
    """Truly blind channel estimator (same as training)"""
    def __init__(self, in_dim: int = 202, width: int = 512, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, width))
            layers.append(nn.LayerNorm(width))
            layers.append(nn.SiLU(inplace=True))
            d = width
        layers.append(nn.Linear(d, 8))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main():
    parser = argparse.ArgumentParser(description="Export Blind model to ONNX")
    parser.add_argument("--ckpt", type=str, default="best_blind.pt", help="Checkpoint path")
    parser.add_argument("--out", type=str, default="blind.onnx", help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()
    
    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    
    in_dim = ckpt.get('in_dim', 202)
    width = ckpt.get('width', 512)
    depth = ckpt.get('depth', 4)
    
    # Create model
    model = BlindMLP(in_dim=in_dim, width=width, depth=depth, dropout=0.0)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Dummy input
    dummy = torch.randn(1, in_dim)
    
    # Export
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=['x'],
        output_names=['h8_pred'],
        dynamic_axes={'x': {0: 'batch'}, 'h8_pred': {0: 'batch'}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    
    print(f"Exported: {args.out}")
    print(f"  in_dim={in_dim}, width={width}, depth={depth}")
    print(f"  Input: x (B, {in_dim})")
    print(f"  Output: h8_pred (B, 8)")


if __name__ == "__main__":
    main()
