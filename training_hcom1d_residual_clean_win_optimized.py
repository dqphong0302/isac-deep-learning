#!/usr/bin/env python3
"""training_hcom1d_residual_clean.py

Train a lightweight 1D residual MLP to estimate h_com (4 complex taps -> 8 real).

Pipeline:
  - Input features: flattened /X (any length D), optional scalar log10(norm_factor),
    and a fast 1-pass estimate h_rls (8 real).
  - Target: /Y (8 real) = ground-truth h_com
  - Model predicts residual: delta_h (8 real)
  - Output: h_hat = h_rls + delta_h
  - Loss: NMSE(h_hat, Y)  (linear NMSE, not dB)

HDF5 layouts supported (like your v2 script):
  Layout A (column-major samples):
    /X: (D, N)    /Y: (8, N)    [/Hrls: (8, N)]
  Layout B (row-major samples):
    /X: (N, D)    /Y: (N, 8)    [/Hrls: (N, 8)]

Optional datasets:
  /norm_factor : (N,), (1,N), (N,1) -> we use log10(norm_factor) as 1 scalar feature
  /Hrls        : 1-pass estimate in 8 real numbers.
                 If missing, we default to zeros (model becomes non-residual).

Usage:
  python training_hcom1d_residual_clean.py --h5 dataset_hcom.h5 --device cuda \
      --epochs 80 --batch 256 --lr 2e-3 --width 256 --depth 3 --ckpt best_hcom1d.pt

Notes:
  - If you want true residual training, regenerate dataset with /Hrls.
  - AMP is enabled only for CUDA.
"""

import argparse
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ------------------------- utils -------------------------

def pick_device(force: Optional[str] = None) -> torch.device:
    if force:
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _read_scalar_int(ds) -> int:
    v = ds[()]
    if isinstance(v, np.ndarray):
        return int(v.reshape(-1)[0])
    return int(v)


def _read_norm_factor(f: h5py.File, idx: int) -> float:
    if "/norm_factor" not in f:
        return 0.0
    nf = f["/norm_factor"][()]
    if isinstance(nf, np.ndarray):
        if nf.ndim == 2:
            if nf.shape[0] == 1:
                val = float(nf[0, idx])
            elif nf.shape[1] == 1:
                val = float(nf[idx, 0])
            else:
                val = float(nf.reshape(-1)[idx])
        elif nf.ndim == 1:
            val = float(nf[idx])
        else:
            val = float(nf.reshape(-1)[0])
    else:
        val = float(nf)
    return float(np.log10(val + 1e-12))


@torch.no_grad()
def nmse_metric(yhat: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # yhat, y: (B, 8)
    err = torch.sum((yhat - y) ** 2, dim=1)
    den = torch.sum(y ** 2, dim=1) + eps
    return torch.mean(err / den)


def nmse_loss(yhat: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # same as metric, but used for backprop
    err = torch.sum((yhat - y) ** 2, dim=1)
    den = torch.sum(y ** 2, dim=1) + eps
    return torch.mean(err / den)


# ------------------------- dataset -------------------------

class Hcom1DResidualH5(Dataset):
    """Returns (x, y, h_rls) where:
       x:  (D + 1 + 8,) float32  -> [X_flat, log10(norm_factor), Hrls]
       y:  (8,) float32
       h_rls: (8,) float32

    If /Hrls missing, h_rls=zeros and residual still works but is effectively non-residual.
    """

    def __init__(self, h5_path: str, indices: np.ndarray, use_norm_factor: bool = True, require_hrls: bool = False,
                 cache: bool = True, cache_max_gb: float = 32.0):
        super().__init__()
        self.h5_path = h5_path
        self.indices = indices.astype(np.int64)
        self.use_norm_factor = use_norm_factor
        self.require_hrls = require_hrls
        self._f: Optional[h5py.File] = None
        self.cache = bool(cache)
        self.cache_max_gb = float(cache_max_gb)
        self.cache_active = False
        self._X_mem: Optional[np.ndarray] = None
        self._Y_mem: Optional[np.ndarray] = None
        self._H_mem: Optional[np.ndarray] = None
        self._NF_mem: Optional[np.ndarray] = None


        with h5py.File(self.h5_path, "r") as f:
            if "/X" not in f or "/Y" not in f:
                raise ValueError("HDF5 must contain /X and /Y")

            X = f["/X"]
            Y = f["/Y"]
            if len(X.shape) != 2 or len(Y.shape) != 2:
                raise ValueError(f"Expected /X and /Y to be 2D. Got X{X.shape}, Y{Y.shape}")

            # Determine layout
            # Layout A: X=(D,N), Y=(8,N)
            # Layout B: X=(N,D), Y=(N,8)
            if Y.shape[0] == 8 and X.shape[1] == Y.shape[1]:
                self.layout = "A"
                self.N = int(Y.shape[1])
                self.D = int(X.shape[0])
            elif Y.shape[1] == 8 and X.shape[0] == Y.shape[0]:
                self.layout = "B"
                self.N = int(Y.shape[0])
                self.D = int(X.shape[1])
            else:
                raise ValueError(
                    "Unrecognized shapes. Expected X=(D,N)&Y=(8,N) or X=(N,D)&Y=(N,8). "
                    f"Got X{X.shape}, Y{Y.shape}."
                )

            self.has_nf = "/norm_factor" in f
            self.has_hrls = "/Hrls" in f
            if self.require_hrls and (not self.has_hrls):
                raise ValueError("require_hrls=True but HDF5 has no /Hrls. Regenerate dataset with /Hrls.")

            # Optional sanity: if /Hrls exists, ensure dimension matches (8,*) or (*,8)
            if self.has_hrls:
                H = f["/Hrls"]
                if len(H.shape) != 2:
                    raise ValueError(f"/Hrls must be 2D. Got {H.shape}")
                ok = False
                if self.layout == "A" and H.shape[0] == 8 and H.shape[1] == self.N:
                    ok = True
                if self.layout == "B" and H.shape[1] == 8 and H.shape[0] == self.N:
                    ok = True
                if not ok:
                    raise ValueError(f"/Hrls shape mismatch for layout {self.layout}. Got {H.shape}, expected (8,N) or (N,8).")

            # Cache to RAM for speed / Windows stability (optional)
            if self.cache:
                # Estimate size assuming float32 storage
                bytes_x = int(np.prod(X.shape)) * 4
                bytes_y = int(np.prod(Y.shape)) * 4
                bytes_h = 0
                if self.has_hrls:
                    bytes_h = int(np.prod(f["/Hrls"].shape)) * 4
                bytes_nf = 0
                if self.has_nf:
                    bytes_nf = int(np.prod(f["/norm_factor"].shape)) * 4
                total_gb = (bytes_x + bytes_y + bytes_h + bytes_nf) / (1024**3)

                if total_gb <= self.cache_max_gb:
                    # Load full arrays once, then close file handle.
                    self._X_mem = np.asarray(X[()], dtype=np.float32)
                    self._Y_mem = np.asarray(Y[()], dtype=np.float32)
                    if self.has_hrls:
                        self._H_mem = np.asarray(f["/Hrls"][()], dtype=np.float32)
                    if self.has_nf:
                        self._NF_mem = np.asarray(f["/norm_factor"][()], dtype=np.float32)
                    self.cache_active = True
                else:
                    print(
                        f"[dataset] cache requested but dataset ~{total_gb:.2f} GB exceeds cache_max_gb={self.cache_max_gb:.2f}. "
                        "Falling back to on-demand HDF5 reads."
                    )


    def _ensure_open(self):
        if self.cache_active:
            return
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = int(self.indices[i])

        if self.cache_active:
            X = self._X_mem
            Y = self._Y_mem
        else:
            self._ensure_open()
            X = self._f["/X"]
            Y = self._f["/Y"]

        if self.layout == "A":
            xflat = X[:, idx]
            y = Y[:, idx]
        else:
            xflat = X[idx, :]
            y = Y[idx, :]

        xflat = np.asarray(xflat, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # norm factor feature
        nf_feat = 0.0
        if self.use_norm_factor and self.has_nf:
            if self.cache_active and (self._NF_mem is not None):
                nf = self._NF_mem
                val = float(nf.reshape(-1)[idx])
                nf_feat = float(np.log10(val + 1e-12))
            else:
                nf_feat = _read_norm_factor(self._f, idx)

        # h_rls
        if self.has_hrls:
            H = self._H_mem if (self.cache_active and (self._H_mem is not None)) else self._f["/Hrls"]
            if self.layout == "A":
                h_rls = np.asarray(H[:, idx], dtype=np.float32)
            else:
                h_rls = np.asarray(H[idx, :], dtype=np.float32)
        else:
            h_rls = np.zeros((8,), dtype=np.float32)

        # concat: [xflat, nf, h_rls]
        x = np.concatenate([xflat, np.asarray([nf_feat], dtype=np.float32), h_rls], axis=0)

        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(h_rls)


# ------------------------- model -------------------------

class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, width: int = 256, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        assert depth >= 1
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
        # x: (B, in_dim), h_rls: (B, 8)
        delta = self.net(x)
        return h_rls + delta


# ------------------------- train/eval -------------------------

@dataclass
class TrainCfg:
    h5: str
    device: str
    epochs: int
    batch: int
    lr: float
    wd: float
    width: int
    depth: int
    dropout: float
    val_frac: float
    seed: int
    workers: int
    ckpt: str
    require_hrls: bool
    no_norm_factor: bool
    cache: bool
    cache_max_gb: float
    amp: bool
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    nmse_sum = 0.0
    mse_sum = 0.0
    n = 0
    for x, y, h_rls in loader:
        x = x.to(device)
        y = y.to(device)
        h_rls = h_rls.to(device)
        yhat = model(x, h_rls)

        b = x.size(0)
        n += b
        nmse_sum += float(nmse_metric(yhat, y).item()) * b
        mse_sum += float(torch.mean((yhat - y) ** 2).item()) * b
    return nmse_sum / max(n, 1), mse_sum / max(n, 1)


def train(cfg: TrainCfg) -> None:
    device = pick_device(cfg.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    set_seed(cfg.seed)

    with h5py.File(cfg.h5, "r") as f:
        X = f["/X"]
        # determine N similarly as dataset
        if "/Y" not in f:
            raise ValueError("HDF5 missing /Y")
        Y = f["/Y"]
        if len(X.shape) != 2 or len(Y.shape) != 2:
            raise ValueError(f"Expected /X and /Y to be 2D. Got X{X.shape}, Y{Y.shape}")
        if Y.shape[0] == 8 and X.shape[1] == Y.shape[1]:
            N = int(Y.shape[1])
            D = int(X.shape[0])
        elif Y.shape[1] == 8 and X.shape[0] == Y.shape[0]:
            N = int(Y.shape[0])
            D = int(X.shape[1])
        else:
            raise ValueError(f"Unrecognized /X,/Y shapes: X{X.shape}, Y{Y.shape}")

    idx = np.arange(N)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(idx)

    n_val = max(1, int(cfg.val_frac * N))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    ds_tr = Hcom1DResidualH5(cfg.h5, tr_idx, use_norm_factor=(not cfg.no_norm_factor), require_hrls=cfg.require_hrls,
                           cache=cfg.cache, cache_max_gb=cfg.cache_max_gb)
    ds_va = Hcom1DResidualH5(cfg.h5, val_idx, use_norm_factor=(not cfg.no_norm_factor), require_hrls=cfg.require_hrls,
                           cache=cfg.cache, cache_max_gb=cfg.cache_max_gb)

    pin = bool(cfg.pin_memory) and (device.type == "cuda")
    nw = int(cfg.workers)
    pw = bool(cfg.persistent_workers) and (nw > 0)
    kwargs = dict(batch_size=cfg.batch, num_workers=nw, pin_memory=pin)

    # On Windows, num_workers>0 can trigger shared-memory errors (1450) with large tensors.
    # Default cfg.workers=0 is safest; if you increase workers, keep batch moderate.
    if nw > 0:
        kwargs["persistent_workers"] = pw
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)

    dl_tr = DataLoader(ds_tr, shuffle=True, **kwargs)
    dl_va = DataLoader(ds_va, shuffle=False, **kwargs)

    in_dim = D + 1 + 8  # xflat + nf + h_rls
    model = ResidualMLP(in_dim=in_dim, width=cfg.width, depth=cfg.depth, dropout=cfg.dropout).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))

    use_amp = (device.type == "cuda") and bool(cfg.amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_nmse = float("inf")

    print(f"Device: {device}")
    print(f"N={N}, D={D}, train={len(ds_tr)}, val={len(ds_va)}")
    print(f"Input dim = {in_dim} (X_flat {D} + nf 1 + Hrls 8)")

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tr_loss = 0.0
        n_seen = 0

        for x, y, h_rls in dl_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            h_rls = h_rls.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                yhat = model(x, h_rls)
                loss = nmse_loss(yhat, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            b = x.size(0)
            tr_loss += float(loss.item()) * b
            n_seen += b

        sched.step()

        tr_loss /= max(n_seen, 1)
        va_nmse, va_mse = evaluate(model, dl_va, device)

        improved = va_nmse < best_nmse
        if improved:
            best_nmse = va_nmse
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "in_dim": in_dim,
                    "best_val_nmse": best_nmse,
                },
                cfg.ckpt,
            )

        print(
            f"[ep {ep:03d}] train_nmse={tr_loss:.6g}  val_nmse={va_nmse:.6g}  val_mse={va_mse:.6g}"
            + ("  (saved)" if improved else "")
        )

    print(f"Done. Best val NMSE = {best_nmse:.6g}. Checkpoint: {cfg.ckpt}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True, help="Path to dataset .h5")
    p.add_argument("--device", default=None, help="cpu|cuda|mps or leave empty for auto")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--workers", type=int, default=0)

    p.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True,
                   help="Cache full H5 datasets into RAM for speed (recommended on Windows). Use --no-cache to disable.")
    p.add_argument("--cache-max-gb", type=float, default=40.0,
                   help="Max RAM (GB) to use for caching. If dataset is larger, falls back to on-demand reads.")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="Use mixed precision (AMP) on CUDA for speed. Use --no-amp to disable.")
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True,
                   help="DataLoader pin_memory when using CUDA. Usually helps throughput.")
    p.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False,
                   help="Keep DataLoader workers alive between epochs (only when workers>0).")
    p.add_argument("--prefetch-factor", type=int, default=2,
                   help="Batches prefetched per worker (only when workers>0).")
    p.add_argument("--ckpt", default="best_hcom1d_residual.pt")
    p.add_argument("--require-hrls", action="store_true", help="Fail if /Hrls is missing")
    p.add_argument("--no-norm-factor", action="store_true", help="Do not use /norm_factor even if present")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    cfg = TrainCfg(
        h5=args.h5,
        device=args.device,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        wd=args.wd,
        width=args.width,
        depth=args.depth,
        dropout=args.dropout,
        val_frac=args.val_frac,
        seed=args.seed,
        workers=args.workers,
        ckpt=args.ckpt,
        require_hrls=args.require_hrls,
        no_norm_factor=args.no_norm_factor,
        cache=args.cache,
        cache_max_gb=args.cache_max_gb,
        amp=args.amp,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    train(cfg)
