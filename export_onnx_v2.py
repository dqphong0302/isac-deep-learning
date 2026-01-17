#!/usr/bin/env python3
"""Export V2 model to ONNX"""

import argparse
import torch
import torch.nn as nn


class ResidualMLP(nn.Module):
    def __init__(self, in_dim=210, width=256, depth=3):
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
    
    def forward(self, x):
        # For ONNX: x contains [features, h_rls] concatenated
        # h_rls is last 8 elements
        h_rls = x[:, -8:]
        delta = self.net(x)
        return h_rls + delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="best_v2.pt")
    parser.add_argument("--out", default="v2.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    
    in_dim = ckpt.get('in_dim', 210)
    width = ckpt.get('width', 256)
    depth = ckpt.get('depth', 3)
    
    model = ResidualMLP(in_dim=in_dim, width=width, depth=depth)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    dummy = torch.randn(1, in_dim)
    
    torch.onnx.export(
        model, dummy, args.out,
        input_names=['x'],
        output_names=['h8_pred'],
        dynamic_axes={'x': {0: 'batch'}, 'h8_pred': {0: 'batch'}},
        opset_version=args.opset,
    )
    
    print(f"Exported: {args.out}")
    print(f"  in_dim={in_dim}, width={width}, depth={depth}")


if __name__ == "__main__":
    main()
