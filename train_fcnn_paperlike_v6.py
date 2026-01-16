#!/usr/bin/env python3
"""
Paper-like FCNN (Liu et al. JACS Au 2022) reproduction helper - v7.

Key paper setup:
- train sheet: 1085 rows
- val sheet:   117 rows
- each epoch: randomly use 80% of the 1085 train pool (no replacement)
- inputs: 2291 = 2048 ECFP + 208 RDKit + 35 (conditions + Ln descriptors)
- FCNN: 512, 128, 16 with PReLU; L1 loss; SGD; lr=1e-5; weight_decay=0.01; epochs=15000

IMPORTANT FIXES in v7:
- PReLU alpha initialized to 0.25 (Keras default), NOT zeros
- StandardScaler applied to non-binary features by default (critical for convergence)
- CSV file input support (--train-csv and --val-csv)
- Explicit condition/Ln column names for robustness
- weight decay applied to weights only (not biases / PReLU alphas)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# Explicit list of 35 condition + Ln descriptor columns (from paper supplementary)
COND_LN_COLUMNS = [
    "ligand c.c./mM",
    "volume ratio of solvent a",
    "volume ratio of solvent b",
    "Molar mass(a)",
    "density(a)",
    "boiling point(a)",
    "melting point(a)",
    "Dipole moment(a)",
    "Solubility in water(a)",
    "log P(a)",
    "Molar mass(b)",
    "density(b)",
    "boiling point(b)",
    "melting point(b)",
    "Dipole moment(b)",
    "Solubility in water(b)",
    "log P(b)",
    "aicd dipole",
    "acid c.c./M",
    "temperature",
    "Ln c.c./mM",
    "Atomic Number_metal",
    "Outer shell electrons_metal",
    "Melting Point_metal  (K)",
    "Boiling Point_metal  (K)",
    "Density_metal  (g/cm3)",
    "First IE_metal  (kJ/mol)",
    "Second IE_metal  (kJ/mol)",
    "Third IE_metal  (kJ/mol)",
    "Electron Affinity_metal  (kJ/mol)",
    "Atomic Radius_metal",
    "Covalent Radius_metal",
    "Pauling EN_metal",
    "Ionic Radius_metal",
    "Standard Entropy_metal (J/mol.K)",
]


def _canon(x: str) -> str:
    return str(x).strip().lower()


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true, y_pred) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def find_target_col(df: pd.DataFrame, target: str) -> str:
    t = _canon(target)
    for c in df.columns:
        if _canon(c) == t:
            return c
    raise KeyError(f"Target column '{target}' not found.")


def find_ecfp_cols(df: pd.DataFrame) -> List[str]:
    """Find ECFP-1 to ECFP-2048 columns."""
    rx = re.compile(r"^ecfp-(\d+)$", re.IGNORECASE)
    pairs: List[Tuple[int, str]] = []
    for c in df.columns:
        m = rx.match(str(c).strip())
        if m:
            pairs.append((int(m.group(1)), c))
    pairs.sort(key=lambda t: t[0])
    cols = [c for _, c in pairs]
    if len(cols) != 2048:
        raise SystemExit(f"Expected 2048 ecfp-* columns; found {len(cols)}")
    nums = [int(str(c).split("-")[1]) for c in cols]
    if nums[0] != 1 or nums[-1] != 2048:
        raise SystemExit(f"ECFP columns not exactly ecfp-1..ecfp-2048. Range={nums[0]}..{nums[-1]}")
    return cols


def find_rdkit_cols(df: pd.DataFrame) -> List[str]:
    """Find RDKit descriptor block: MaxEStateIndex to fr_urea (208 columns)."""
    cols = list(df.columns)
    try:
        i0 = cols.index("MaxEStateIndex")
        i1 = cols.index("fr_urea")
    except ValueError:
        raise SystemExit("Could not find RDKit block start/end: MaxEStateIndex .. fr_urea")
    block = cols[i0:i1 + 1]
    if len(block) != 208:
        raise SystemExit(f"RDKit block MaxEStateIndex..fr_urea length != 208 (got {len(block)})")
    return block


def find_cond_ln_cols(df: pd.DataFrame) -> List[str]:
    """
    Find the 35 condition + Ln descriptor columns by explicit names.
    More robust than positional approach.
    """
    found_cols = []
    df_cols_stripped = {c.strip(): c for c in df.columns}
    
    for cname in COND_LN_COLUMNS:
        # Try exact match first
        if cname in df.columns:
            found_cols.append(cname)
            continue
        # Try stripped match
        cname_stripped = cname.strip()
        if cname_stripped in df_cols_stripped:
            found_cols.append(df_cols_stripped[cname_stripped])
            continue
        # Try with trailing space (some CSV exports add spaces)
        for df_col in df.columns:
            if df_col.strip() == cname_stripped:
                found_cols.append(df_col)
                break
        else:
            raise SystemExit(f"Could not find condition/Ln column: '{cname}'")
    
    if len(found_cols) != 35:
        raise SystemExit(f"Expected 35 condition+Ln columns; found {len(found_cols)}")
    return found_cols


def detect_binary_columns(X: np.ndarray) -> np.ndarray:
    """
    Returns boolean mask: True if column values are all in {0,1} (within tolerance).
    """
    col_min = np.nanmin(X, axis=0)
    col_max = np.nanmax(X, axis=0)
    # Must be within [0,1] and close to endpoints
    within = (col_min >= -1e-6) & (col_max <= 1 + 1e-6)
    close0 = np.isclose(col_min, 0.0, atol=1e-6) | np.isclose(col_min, 1.0, atol=1e-6)
    close1 = np.isclose(col_max, 0.0, atol=1e-6) | np.isclose(col_max, 1.0, atol=1e-6)
    return within & close0 & close1


class FCNN(nn.Module):
    """
    Paper architecture: 2291 -> 512 -> 128 -> 16 -> 1
    Activation: PReLU (per-neuron, not shared)
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.a1 = nn.PReLU(num_parameters=512)   # per-neuron
        self.fc2 = nn.Linear(512, 128)
        self.a2 = nn.PReLU(num_parameters=128)   # per-neuron
        self.fc3 = nn.Linear(128, 16)
        self.a3 = nn.PReLU(num_parameters=16)    # per-neuron
        self.out = nn.Linear(16, 1)

        self._init_like_keras()

    def _init_like_keras(self):
        """
        Keras Dense default: Glorot/Xavier uniform for weights, zeros for biases.
        Keras PReLU default: alpha = 0.25 (NOT zeros!)
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.PReLU):
                # CRITICAL FIX: Keras PReLU default alpha is 0.25, not 0!
                nn.init.constant_(m.weight, 0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a1(self.fc1(x))
        x = self.a2(self.fc2(x))
        x = self.a3(self.fc3(x))
        return self.out(x).squeeze(-1)


def make_sgd_weightdecay_weights_only(model: nn.Module, lr: float, weight_decay: float):
    """
    Apply weight decay only to weight matrices (not biases or PReLU alphas).
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Apply wd only to weight matrices (ndim >= 2)
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return torch.optim.SGD(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        momentum=0.0,
        nesterov=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Train FCNN for lanthanide separation (paper replication)")
    
    # CSV input (new in v7)
    ap.add_argument("--train-csv", type=str, help="Path to training CSV file")
    ap.add_argument("--val-csv", type=str, help="Path to validation CSV file")
    
    # Excel input (legacy, for compatibility)
    ap.add_argument("--xlsx", type=str, help="Path to Excel file (legacy mode)")
    ap.add_argument("--train-sheet", type=str, help="Training sheet name (for xlsx)")
    ap.add_argument("--val-sheet", type=str, help="Validation sheet name (for xlsx)")
    
    ap.add_argument("--target", default="log_D", help="Target column name")
    ap.add_argument("--epochs", type=int, default=15000, help="Number of training epochs")
    ap.add_argument("--train-frac-each-epoch", type=float, default=0.8, 
                    help="Fraction of training data to use each epoch")
    ap.add_argument("--batch-size", type=int, default=128, help="Batch size")
    ap.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    ap.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay (L2 reg)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--scale", choices=["none", "standard_nonbinary", "standard_all"], 
                    default="standard_nonbinary",
                    help="Feature scaling strategy (default: standard_nonbinary)")
    ap.add_argument("--log-every", type=int, default=250, help="Log interval")
    ap.add_argument("--outdir", required=True, help="Output directory")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Validate input arguments
    if args.train_csv and args.val_csv:
        mode = "csv"
    elif args.xlsx and args.train_sheet and args.val_sheet:
        mode = "xlsx"
    else:
        ap.error("Must provide either (--train-csv and --val-csv) or (--xlsx, --train-sheet, --val-sheet)")

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    if mode == "csv":
        print(f"[DATA] Loading from CSV: train={args.train_csv}, val={args.val_csv}")
        train_df = pd.read_csv(args.train_csv)
        val_df = pd.read_csv(args.val_csv)
    else:
        print(f"[DATA] Loading from Excel: {args.xlsx}")
        train_df = pd.read_excel(args.xlsx, sheet_name=args.train_sheet)
        val_df = pd.read_excel(args.xlsx, sheet_name=args.val_sheet)

    # Find columns
    target_train = find_target_col(train_df, args.target)
    target_val = find_target_col(val_df, args.target)

    ecfp_cols = find_ecfp_cols(train_df)
    rdkit_cols = find_rdkit_cols(train_df)
    cond_ln_cols = find_cond_ln_cols(train_df)
    feat_cols = ecfp_cols + rdkit_cols + cond_ln_cols

    if len(feat_cols) != 2291:
        raise SystemExit(f"Expected 2291 features; got {len(feat_cols)}")

    # Verify validation set has the same columns
    _ = find_ecfp_cols(val_df)
    _ = find_rdkit_cols(val_df)
    _ = find_cond_ln_cols(val_df)

    # Numeric coercion + dropna
    for c in feat_cols + [target_train]:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
    for c in feat_cols + [target_val]:
        val_df[c] = pd.to_numeric(val_df[c], errors="coerce")

    train_df = train_df.dropna(subset=feat_cols + [target_train]).copy()
    val_df = val_df.dropna(subset=feat_cols + [target_val]).copy()

    print(f"[SPLIT] train_rows={len(train_df)} val_rows={len(val_df)}")
    print(f"[COLS] ECFP={len(ecfp_cols)} RDKit={len(rdkit_cols)} cond+Ln={len(cond_ln_cols)} total={len(feat_cols)}")

    X_tr = train_df[feat_cols].to_numpy(dtype=np.float32)
    y_tr = train_df[target_train].to_numpy(dtype=np.float32)
    X_va = val_df[feat_cols].to_numpy(dtype=np.float32)
    y_va = val_df[target_val].to_numpy(dtype=np.float32)

    # Feature scaling
    scaler = None
    bin_mask = None
    
    if args.scale == "standard_all":
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_va = scaler.transform(X_va).astype(np.float32)
        print(f"[SCALE] Applied StandardScaler to ALL {X_tr.shape[1]} features")

    elif args.scale == "standard_nonbinary":
        # Scale only columns that are NOT binary in train pool
        bin_mask = detect_binary_columns(X_tr)
        n_binary = bin_mask.sum()
        n_continuous = (~bin_mask).sum()
        print(f"[SCALE] Detected {n_binary} binary cols, {n_continuous} continuous cols")
        
        if n_continuous > 0:
            scaler = StandardScaler()
            X_tr_cont = X_tr[:, ~bin_mask]
            X_va_cont = X_va[:, ~bin_mask]
            X_tr[:, ~bin_mask] = scaler.fit_transform(X_tr_cont).astype(np.float32)
            X_va[:, ~bin_mask] = scaler.transform(X_va_cont).astype(np.float32)
            print(f"[SCALE] Applied StandardScaler to {n_continuous} non-binary features")
    else:
        print(f"[SCALE] No scaling applied (--scale={args.scale})")

    # Baseline: predict mean of training targets
    y_mean = float(np.mean(y_tr))
    base_pred = np.full_like(y_va, y_mean, dtype=np.float32)
    base_m = metrics(y_va, base_pred)
    print(f"[BASELINE mean] val RMSE={base_m['rmse']:.4f} MAE={base_m['mae']:.4f} R2={base_m['r2']:.4f}")
    
    n_each = int(round(args.train_frac_each_epoch * len(y_tr)))
    n_each = max(1, min(len(y_tr), n_each))
    print(f"[TRAIN] pool={len(y_tr)} samples, using {n_each} ({args.train_frac_each_epoch:.0%}) per epoch")

    # Convert to tensors
    X_tr_t = torch.tensor(X_tr, device=device)
    y_tr_t = torch.tensor(y_tr, device=device)
    X_va_t = torch.tensor(X_va, device=device)

    # Build model
    model = FCNN(in_dim=X_tr.shape[1]).to(device)
    loss_fn = nn.L1Loss()  # L1 loss as per paper
    opt = make_sgd_weightdecay_weights_only(model, lr=args.lr, weight_decay=args.weight_decay)

    print(f"[MODEL] FCNN: {X_tr.shape[1]} -> 512 -> 128 -> 16 -> 1")
    print(f"[OPTIM] SGD lr={args.lr}, weight_decay={args.weight_decay} (weights only)")
    print(f"[LOSS] L1Loss (MAE)")

    rng = np.random.RandomState(args.seed)
    n_pool = len(y_tr)

    best = {"epoch": None, "val_rmse": float("inf"), "val_mae": None, "val_r2": None}
    best_path = os.path.join(args.outdir, "best_model.pt")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Sample 80% of training data each epoch (no replacement)
        idx = rng.choice(n_pool, size=n_each, replace=False)

        ds = TensorDataset(X_tr_t[idx], y_tr_t[idx])
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            pred_tr = model(X_tr_t).detach().cpu().numpy()
            pred_va = model(X_va_t).detach().cpu().numpy()

        tr_m = metrics(y_tr, pred_tr)
        va_m = metrics(y_va, pred_va)
        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in tr_m.items()},
            **{f"val_{k}": v for k, v in va_m.items()}
        })

        # Track best model by val RMSE
        if va_m["rmse"] < best["val_rmse"]:
            best = {"epoch": epoch, "val_rmse": va_m["rmse"], "val_mae": va_m["mae"], "val_r2": va_m["r2"]}
            torch.save(model.state_dict(), best_path)

        if epoch == 1 or epoch % args.log_every == 0:
            print(f"[E{epoch:5d}] train RMSE={tr_m['rmse']:.4f} | val RMSE={va_m['rmse']:.4f} MAE={va_m['mae']:.4f} R2={va_m['r2']:.4f}")

    # Load best model and do final evaluation
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        pred_tr = model(X_tr_t).detach().cpu().numpy()
        pred_va = model(X_va_t).detach().cpu().numpy()

    tr_m = metrics(y_tr, pred_tr)
    va_m = metrics(y_va, pred_va)

    # Save outputs
    pd.DataFrame(history).to_csv(os.path.join(args.outdir, "history.csv"), index=False)
    pd.DataFrame({
        "set": ["train"] * len(y_tr) + ["val"] * len(y_va),
        "y_true": np.concatenate([y_tr, y_va]),
        "y_pred": np.concatenate([pred_tr, pred_va]),
    }).to_csv(os.path.join(args.outdir, "predictions.csv"), index=False)

    out = {
        "paper_hparams": {
            "activation": "PReLU (alpha=0.25)",
            "lr": 1e-5,
            "weight_decay": 0.01,
            "epochs": 15000,
            "hidden": [512, 128, 16],
            "loss": "L1",
            "optimizer": "SGD",
            "train_frac_each_epoch": 0.8,
        },
        "run_hparams": {
            "epochs": args.epochs,
            "train_frac_each_epoch": args.train_frac_each_epoch,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "scale": args.scale,
            "seed": args.seed,
            "device": str(device),
        },
        "features": {
            "n_total": len(feat_cols),
            "ecfp": len(ecfp_cols),
            "rdkit": len(rdkit_cols),
            "cond_ln": len(cond_ln_cols),
            "cond_ln_names": cond_ln_cols,
        },
        "baseline_mean": {"val": base_m, "mean_y_train": float(np.mean(y_tr))},
        "best_by_val_rmse": best,
        "final_train": tr_m,
        "final_val": va_m,
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 60)
    print(f"[BEST] epoch={best['epoch']} val RMSE={best['val_rmse']:.4f} MAE={best['val_mae']:.4f} R2={best['val_r2']:.4f}")
    print(f"[FINAL] val RMSE={va_m['rmse']:.4f} MAE={va_m['mae']:.4f} R2={va_m['r2']:.4f}")
    print("=" * 60)
    print(f"[DONE] Outputs saved to: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
