#!/usr/bin/env python3
"""Extended ensemble search including non-CNN baselines.

Motivation: current top-3 CNN ensemble is ceiling-bound by logistic_image_scale
(rank corr 0.0392). Because CNN-CNN raw-score correlations sit at 0.5~0.6, the
CNN-only ensemble has limited diversification. This script enumerates every
size-2/3/4 combination over 9 models (7 CNN + 2 logistic) and reports the best
by OOS rank correlation and top-k Sharpe.

Two aggregation modes:
  - raw:  mean of raw signal_value (same as current ensemble_top3)
  - rank: mean of per-(date) cross-sectional rank (scale-invariant)

Outputs:
  ode_inputs_cnn/ensemble_search.csv        — every combination's metrics
  ode_inputs_cnn/ensemble_search_top.md     — top-20 per aggregation mode
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "ode_inputs_cnn"

SOURCES = {
    "logistic_image_scale":         "outputs_walkforward_4model",
    "logistic_cumulative_scale":    "outputs_walkforward_4model",
    "cnn_1d_image_scale":           "outputs_walkforward_4model",
    "cnn_2d_rendered_images":       "outputs_walkforward_4model",
    "cnn_1d_attention_image_scale": "outputs_walkforward_1dcnn_extra",
    "cnn_1d_cumulative_scale":      "outputs_walkforward_1dcnn_extra",
    "cnn_1d_dilated_image_scale":   "outputs_walkforward_1dcnn_extra",
    "cnn_1d_multiscale_image_scale":"outputs_walkforward_1dcnn_extra",
    "cnn_2d_residual_images":       "outputs_walkforward_2d_residual",
}

HORIZON = 20
TOP_K = 2


def load_long() -> pd.DataFrame:
    frames = []
    for model, d in SOURCES.items():
        df = pd.read_csv(ROOT / d / "walkforward_predictions.csv")
        df = df[df["model_name"] == model][["date", "asset", "signal_value", "future_return"]].copy()
        df["model_name"] = model
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"])
    return long


def pivot_wide(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (raw_wide, rank_wide) both indexed by (date, asset)."""
    raw = long.pivot_table(index=["date", "asset"], columns="model_name", values="signal_value")
    # cross-sectional rank per date per model
    rank = (
        long.assign(rk=long.groupby(["date", "model_name"])["signal_value"].rank(pct=True))
        .pivot_table(index=["date", "asset"], columns="model_name", values="rk")
    )
    future = long.groupby(["date", "asset"])["future_return"].first()
    raw["future_return"] = future
    rank["future_return"] = future
    return raw.reset_index(), rank.reset_index()


def eval_combo(df: pd.DataFrame, models: list[str]) -> dict:
    """Average signals across selected models and compute rank corr + top-k Sharpe."""
    signal = df[models].mean(axis=1)
    tmp = df[["date", "asset", "future_return"]].copy()
    tmp["signal"] = signal.values

    # rank correlation — average spearman per date
    rc = (
        tmp.groupby("date")
        .apply(lambda g: g["signal"].rank().corr(g["future_return"].rank()))
        .mean()
    )

    # top-k portfolio Sharpe — rebal every HORIZON days
    dates = sorted(tmp["date"].unique())[::HORIZON]
    rets = []
    for d in dates:
        frame = tmp[tmp["date"] == d].sort_values("signal", ascending=False)
        if len(frame) < TOP_K:
            continue
        rets.append(float(frame.head(TOP_K)["future_return"].mean()))
    r = np.array(rets, dtype=float)
    if len(r) < 2:
        sharpe = np.nan
    else:
        sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(252.0 / HORIZON))
    cum = float(np.prod(1 + r) - 1) if len(r) else np.nan
    hit = float((r > 0).mean()) if len(r) else np.nan
    return {
        "rank_corr": float(rc),
        "top_k_sharpe": sharpe,
        "top_k_cum": cum,
        "top_k_hit": hit,
    }


def search(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    models = [m for m in SOURCES.keys()]
    rows = []
    for k in range(1, 5):
        for combo in itertools.combinations(models, k):
            res = eval_combo(df, list(combo))
            rows.append({
                "mode": mode,
                "k": k,
                "members": " + ".join(combo),
                **res,
            })
    return pd.DataFrame(rows)


def main() -> None:
    print("Loading predictions ...")
    long = load_long()
    raw_wide, rank_wide = pivot_wide(long)
    print(f"  raw_wide shape: {raw_wide.shape}, rank_wide shape: {rank_wide.shape}")

    print("Searching ensemble combinations (raw mean) ...")
    raw_res = search(raw_wide, "raw")
    print("Searching ensemble combinations (rank mean) ...")
    rank_res = search(rank_wide, "rank")

    full = pd.concat([raw_res, rank_res], ignore_index=True)
    full.to_csv(OUT / "ensemble_search.csv", index=False)
    print(f"  wrote {OUT/'ensemble_search.csv'}  ({len(full)} rows)")

    # Top-20 per mode by rank_corr
    lines = ["# Ensemble search — top-20 per aggregation mode\n"]
    for mode in ["raw", "rank"]:
        sub = full[full["mode"] == mode].sort_values("rank_corr", ascending=False).head(20)
        lines.append(f"## Mode: {mode} (sorted by OOS rank corr)\n")
        lines.append(sub[["k", "members", "rank_corr", "top_k_sharpe", "top_k_cum", "top_k_hit"]].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
        # Also sorted by Sharpe
        sub2 = full[full["mode"] == mode].sort_values("top_k_sharpe", ascending=False).head(20)
        lines.append(f"## Mode: {mode} (sorted by top-k Sharpe)\n")
        lines.append(sub2[["k", "members", "rank_corr", "top_k_sharpe", "top_k_cum", "top_k_hit"]].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
    (OUT / "ensemble_search_top.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {OUT/'ensemble_search_top.md'}")

    # Headline
    best_rc = full.sort_values("rank_corr", ascending=False).head(5)
    best_sr = full.sort_values("top_k_sharpe", ascending=False).head(5)
    print("\n=== TOP-5 by rank corr ===")
    print(best_rc[["mode", "k", "members", "rank_corr", "top_k_sharpe"]].to_string(index=False))
    print("\n=== TOP-5 by Sharpe ===")
    print(best_sr[["mode", "k", "members", "rank_corr", "top_k_sharpe"]].to_string(index=False))


if __name__ == "__main__":
    main()
