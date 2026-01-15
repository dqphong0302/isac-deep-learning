"""export_onnx_hcom1d_smallfeat.py

Export the trained 1D residual MLP model to ONNX.

Compatible with:
  - generate_dataset_hcom_1d_residual_smallfeat_to_h5.m
  - training_hcom1d_residual_clean_win_optimized.py

In training, each sample returns:
  x      : (in_dim,) float32   concat([features(D), nf_feat(1), h_rls(8)])
  y      : (8,)      float32   [Re(h_true(1:4)), Im(h_true(1:4))]
  h_rls  : (8,)      float32   [Re(h_ls(1:4)),  Im(h_ls(1:4))]

The network predicts delta(x) and outputs:
  h8_pred = h_rls + delta(x)

This exporter builds an ONNX model with ONE input:
  x : (B, in_dim) float32
and ONE output:
  h8_pred : (B, 8) float32

Usage:
  python export_onnx_hcom1d_smallfeat.py --ckpt best_hcom1d_small.pt --out hcom1d_small.onnx

If MATLAB complains about opset, try:
  ... --opset 13
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn


class ResidualMLPNet(nn.Module):
    """Rebuild the same net as in training (Linear + LayerNorm + SiLU blocks)."""

    def __init__(self, in_dim: int, width: int = 256, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        assert depth >= 1
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, width))
            layers.append(nn.LayerNorm(width))
            layers.append(nn.SiLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = width
        layers.append(nn.Linear(d, 8))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        delta = self.net(x)              # (B, 8)
        h_rls = x[:, -8:]                # last 8 entries are h_rls (Re/Im)
        return h_rls + delta


def load_ckpt(path: str) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError("Expected checkpoint dict. Got: %r" % type(ckpt))
    if "model_state" not in ckpt:
        raise ValueError("Checkpoint missing key 'model_state'.")
    if "in_dim" not in ckpt:
        raise ValueError("Checkpoint missing key 'in_dim'.")
    return ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Checkpoint produced by training script (e.g., best_hcom1d_small.pt)")
    ap.add_argument("--out", default="hcom1d_small.onnx", help="Output ONNX filename")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    ap.add_argument("--fp16", action="store_true", help="Export with fp16 weights (still float32 IO).")
    args = ap.parse_args()

    ckpt = load_ckpt(args.ckpt)
    sd = ckpt["model_state"]
    in_dim = int(ckpt["in_dim"])

    cfg = ckpt.get("cfg", {}) if isinstance(ckpt.get("cfg", {}), dict) else {}
    width = int(cfg.get("width", 256))
    depth = int(cfg.get("depth", 3))
    dropout = float(cfg.get("dropout", 0.0))

    model = ResidualMLPNet(in_dim=in_dim, width=width, depth=depth, dropout=0.0)
    model.load_state_dict(sd, strict=True)
    model.eval()

    if args.fp16:
        model = model.half()
        dummy = torch.randn(1, in_dim, dtype=torch.float16)
    else:
        dummy = torch.randn(1, in_dim, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["x"],
        output_names=["h8_pred"],
        opset_version=args.opset,
        dynamic_axes={"x": {0: "batch"}, "h8_pred": {0: "batch"}},
    )

    print("Exported:", args.out)
    print(f"in_dim={in_dim} width={width} depth={depth} (dropout ignored in export)")


if __name__ == "__main__":
    main()
