#!/usr/bin/env python3
"""Train one fold of cnn_2d_residual_images with loss-curve logging.

Produces a train-vs-val MSE curve to diagnose overfit (val diverges while
train drops) vs underfit (both plateau high) vs well-fit (val tracks train).
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import PipelineConfig
from src.data.loader import OPTIONAL_FIELDS, REQUIRED_PRICE_FIELDS, load_etf_csv, restrict_common_valid_sample
from src.models.cnn import build_cnn_model
from src.pipeline import build_samples, set_global_seed
from src.walkforward import _build_feature_sets, generate_walkforward_folds

ROOT = Path(__file__).parent
OUT_FIG = ROOT / "ode_inputs_cnn" / "figures" / "10_2d_cnn_loss_curves.png"


def _bundle_tensors(features: np.ndarray, targets: np.ndarray) -> TensorDataset:
    return TensorDataset(torch.from_numpy(features).float(), torch.from_numpy(targets.astype(np.float32)).float())


def train_with_curves(model, train_x, train_y, val_x, val_y, epochs=30, batch_size=128,
                      lr=1e-3, weight_decay=1e-4, device="cpu") -> dict:
    train_loader = DataLoader(_bundle_tensors(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_bundle_tensors(val_x, val_y), batch_size=batch_size, shuffle=False)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()
    history = {"train": [], "val": []}
    for _ in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for bx, by in train_loader:
            opt.zero_grad()
            pred = model(bx.to(device))
            loss = crit(pred, by.to(device))
            loss.backward()
            opt.step()
            running += loss.item() * len(bx)
            n += len(bx)
        train_loss = running / n
        model.eval()
        with torch.no_grad():
            running = 0.0
            n = 0
            for bx, by in val_loader:
                pred = model(bx.to(device))
                running += crit(pred, by.to(device)).item() * len(bx)
                n += len(bx)
            val_loss = running / n
        history["train"].append(train_loss)
        history["val"].append(val_loss)
    return history


def pick_fold(bundle, config, fold_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metadata = bundle.metadata
    all_dates = sorted(metadata["date"].unique().tolist())
    folds = generate_walkforward_folds(all_dates, config)
    print(f"  {len(folds)} folds; picking fold index {fold_idx}")
    fold = folds[fold_idx]
    dates = metadata["date"]
    return (
        dates.isin(fold["train_dates"]).to_numpy(),
        dates.isin(fold["val_dates"]).to_numpy(),
        dates.isin(fold["test_dates"]).to_numpy(),
    )


def main() -> None:
    set_global_seed(42)
    config = PipelineConfig(
        data_path="etfdata.csv",
        output_dir="outputs_diagnose",
        lookback=60,
        horizon=20,
        label_mode="regression",
        target_name="future_return",
        enabled_models=["cnn_2d_residual_images", "cnn_1d_dilated_image_scale"],
        wf_min_train_days=500,
        wf_val_days=60,
        wf_test_days=60,
        cnn_epochs=30,
        cnn_repeats=1,
        top_k=2,
        seed=42,
        sample_preview_count=0,
    )

    print("Loading data ...")
    _, long_panel, assets, fields, _ = load_etf_csv(config.data_path)
    required = REQUIRED_PRICE_FIELDS.copy()
    if "volume" in fields:
        required.extend(OPTIONAL_FIELDS)
    panel, _ = restrict_common_valid_sample(long_panel, selected_assets=assets, required_fields=required)

    print("Building samples ...")
    bundle, prep = build_samples(panel, config)
    print(f"  shape: {prep['sequence_shape']}, n_samples={prep['n_samples']}")

    # mid fold (~12) → decent training size
    tr, val, te = pick_fold(bundle, config, fold_idx=12)
    n_tr = int(tr.sum()); n_val = int(val.sum())
    print(f"  fold 12: train={n_tr}  val={n_val}")

    feats = _build_feature_sets(bundle, len(bundle.metadata), config.enabled_models)
    target = bundle.metadata["target"].to_numpy(dtype=np.float32)

    histories = {}
    for name in config.enabled_models:
        print(f"\nTraining {name} ...")
        x = feats[name]
        m = build_cnn_model(name, x.shape[1:])
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  params={n_params}")
        hist = train_with_curves(
            m, x[tr], target[tr], x[val], target[val],
            epochs=30, batch_size=128, lr=1e-3, weight_decay=1e-4, device="cpu",
        )
        histories[name] = hist
        print(f"  final train={hist['train'][-1]:.6f}  val={hist['val'][-1]:.6f}  gap={hist['val'][-1]-hist['train'][-1]:+.6f}")
        best_val = min(hist["val"])
        best_epoch = hist["val"].index(best_val) + 1
        print(f"  best val={best_val:.6f} at epoch {best_epoch}")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, name in zip(axes, config.enabled_models):
        h = histories[name]
        ax.plot(range(1, len(h["train"])+1), h["train"], label="train", linewidth=2)
        ax.plot(range(1, len(h["val"])+1), h["val"], label="val", linewidth=2)
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE loss")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("2D CNN vs 1D dilated — train/val loss curves (fold 12)", fontsize=14)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=160, bbox_inches="tight")
    print(f"\n  wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
