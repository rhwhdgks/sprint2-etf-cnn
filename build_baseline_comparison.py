#!/usr/bin/env python3
"""Merge CNN and non-CNN (logistic) baseline results into a unified comparison
and regenerate model-comparison figures with baselines included."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parent
OUT = ROOT / "ode_inputs_cnn"
FIG = OUT / "figures"

sns.set_theme(style="whitegrid", context="talk")

# walk-forward comparison files that contain logistic baselines
WF_DIRS = [
    "outputs_walkforward_4model",
    "outputs_walkforward_mu",
    "outputs_walkforward_1dcnn_extra",
    "outputs_walkforward_2d_residual",
]

KEY_COLS = [
    "model_name",
    "future_return_rank_correlation",
    "top_k_sharpe",
    "top_k_cumulative_return",
    "top_k_hit_rate",
    "turnover",
    "n_folds",
]


def compute_ensemble_portfolio_metrics(
    top_models: list[str] | None = None,
    horizon: int = 20,
    top_k: int = 2,
) -> dict:
    """Recompute top-k Sharpe / cumulative return for the ensemble_top3 signal."""
    import math
    import numpy as np

    if top_models is None:
        top_models = [
            "cnn_1d_dilated_image_scale",
            "cnn_1d_cumulative_scale",
            "cnn_1d_attention_image_scale",
        ]
    preds = pd.read_csv(ROOT / "outputs_walkforward_1dcnn_extra" / "walkforward_predictions.csv")
    sub = preds[preds["model_name"].isin(top_models)]
    ens = (
        sub.groupby(["date", "asset"], as_index=False)
        .agg(signal_value=("signal_value", "mean"), future_return=("future_return", "first"))
    )
    ens["date"] = pd.to_datetime(ens["date"])
    rebal_dates = sorted(ens["date"].drop_duplicates().tolist())[::horizon]
    rets = []
    for d in rebal_dates:
        frame = ens[ens["date"] == d].sort_values("signal_value", ascending=False)
        if len(frame) < top_k:
            continue
        rets.append(float(frame.head(top_k)["future_return"].mean()))
    r = np.array(rets, dtype=float)
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(252.0 / horizon))
    return {
        "top_k_sharpe": sharpe,
        "top_k_cumulative_return": float(np.prod(1 + r) - 1),
        "top_k_hit_rate": float(np.mean(r > 0)),
    }


def load_unified() -> pd.DataFrame:
    frames = []
    for d in WF_DIRS:
        p = ROOT / d / "walkforward_comparison.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)[KEY_COLS]
        frames.append(df)
    merged = pd.concat(frames).drop_duplicates(subset="model_name").reset_index(drop=True)

    # bring in ensemble from ode_inputs_cnn/comparison.csv + recompute portfolio metrics
    ens = pd.read_csv(OUT / "comparison.csv")
    ens_row = ens[ens["model_name"] == "ensemble_top3"].iloc[0]
    ens_portfolio = compute_ensemble_portfolio_metrics()
    ens_new = {
        "model_name": "ensemble_top3",
        "future_return_rank_correlation": float(ens_row["oos_rank_corr_mean"]),
        "top_k_sharpe": ens_portfolio["top_k_sharpe"],
        "top_k_cumulative_return": ens_portfolio["top_k_cumulative_return"],
        "top_k_hit_rate": ens_portfolio["top_k_hit_rate"],
        "turnover": None,
        "n_folds": int(ens_row["n_folds"]),
    }
    merged = pd.concat([merged, pd.DataFrame([ens_new])], ignore_index=True)

    merged["family"] = merged["model_name"].apply(classify_family)
    return merged.sort_values("future_return_rank_correlation", ascending=False).reset_index(drop=True)


def classify_family(name: str) -> str:
    if name.startswith("ensemble_"):
        return "Ensemble (CNN)"
    if name.startswith("logistic_") and "image" in name:
        return "Baseline (image+logistic)"
    if name.startswith("logistic_"):
        return "Baseline (no image)"
    if "_cumulative_" in name:
        return "CNN (no image)"
    if "_2d_" in name:
        return "CNN 2D image"
    return "CNN 1D image"


FAMILY_COLORS = {
    "Ensemble (CNN)": "#c0392b",
    "CNN 1D image": "#2980b9",
    "CNN 2D image": "#8e44ad",
    "CNN (no image)": "#16a085",
    "Baseline (image+logistic)": "#e67e22",
    "Baseline (no image)": "#7f8c8d",
}


def fig_model_comparison_with_baseline(df: pd.DataFrame) -> None:
    df = df.sort_values("future_return_rank_correlation", ascending=True)
    colors = [FAMILY_COLORS[f] for f in df["family"]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    axes[0].barh(df["model_name"], df["future_return_rank_correlation"],
                 color=colors, edgecolor="black", linewidth=0.5)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("OOS rank correlation")
    axes[0].set_title("Signal quality (rank corr)")

    sharpe = df.set_index("model_name")["top_k_sharpe"].reindex(df["model_name"]).fillna(0).values
    axes[1].barh(df["model_name"], sharpe, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Top-k portfolio Sharpe (annualized)")
    axes[1].set_title("Portfolio-level quality (top-k Sharpe)")

    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black") for c in FAMILY_COLORS.values()]
    fig.legend(handles, list(FAMILY_COLORS.keys()), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=11)
    fig.suptitle("All models — CNN vs non-CNN baselines", fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "01_model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG/'01_model_comparison.png'}")


def fig_ablation_image_vs_cnn(df: pd.DataFrame) -> None:
    """2x2 ablation: (image on/off) × (CNN on/off). Bar for each cell with rank corr."""
    cells = {
        ("No image", "Logistic"): df.loc[df["model_name"] == "logistic_cumulative_scale", "future_return_rank_correlation"].iloc[0],
        ("Image",    "Logistic"): df.loc[df["model_name"] == "logistic_image_scale", "future_return_rank_correlation"].iloc[0],
        ("No image", "CNN"):      df.loc[df["model_name"] == "cnn_1d_cumulative_scale", "future_return_rank_correlation"].iloc[0],
        ("Image",    "CNN"):      df.loc[df["model_name"] == "cnn_1d_dilated_image_scale", "future_return_rank_correlation"].iloc[0],
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    rows = ["Logistic", "CNN"]
    cols = ["No image", "Image"]
    matrix = [[cells[(c, r)] for c in cols] for r in rows]

    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-0.05, vmax=0.05, aspect="auto")
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            val = cells[(c, r)]
            ax.text(j, i, f"{val:+.4f}", ha="center", va="center", fontsize=18,
                    color="white" if abs(val) > 0.025 else "black", fontweight="bold")
    ax.set_xticks(range(len(cols)), cols, fontsize=14)
    ax.set_yticks(range(len(rows)), rows, fontsize=14)
    ax.set_xlabel("Input representation")
    ax.set_ylabel("Model family")
    ax.set_title("Ablation — image transform vs CNN contribution (OOS rank corr)")
    fig.colorbar(im, ax=ax, label="rank corr")
    fig.tight_layout()
    fig.savefig(FIG / "09_ablation_image_vs_cnn.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {FIG/'09_ablation_image_vs_cnn.png'}")


def main() -> None:
    print("Merging comparison data ...")
    df = load_unified()
    out_csv = OUT / "comparison_with_baselines.csv"
    df.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}")
    print(df[["model_name", "future_return_rank_correlation", "top_k_sharpe", "family"]].to_string(index=False))

    fig_model_comparison_with_baseline(df)
    fig_ablation_image_vs_cnn(df)


if __name__ == "__main__":
    main()
