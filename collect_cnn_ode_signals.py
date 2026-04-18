#!/usr/bin/env python3
"""Collect walk-forward CNN outputs and package each as an ODE-ready signal.

Usage (single predictions file):
  python collect_cnn_ode_signals.py \
    --pred-paths outputs_wf_cnn_all/walkforward_predictions.csv \
    --risk-path  outputs_walkforward_risk/walkforward_predictions.csv \
    --output-dir ode_inputs_cnn

Usage (merging multiple walk-forward runs):
  python collect_cnn_ode_signals.py \
    --pred-paths outputs_walkforward_4model/walkforward_predictions.csv \
                 outputs_walkforward_extra/walkforward_predictions.csv \
    --risk-path  outputs_walkforward_risk/walkforward_predictions.csv \
    --output-dir ode_inputs_cnn

Outputs (per CNN model):
  ode_inputs_cnn/{model_name}/mu_daily.csv
  ode_inputs_cnn/{model_name}/ode_bundle.csv
  ode_inputs_cnn/{model_name}/ode_config.json

  ode_inputs_cnn/comparison.csv      (all CNN models ranked by OOS quality)
  ode_inputs_cnn/comparison.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from make_ode_inputs import (
    calibrate_mu_expanding,
    compute_rolling_sigma,
    forward_fill_to_daily,
    process_risk_signal,
    build_ode_bundle,
)


CNN_MODELS = {
    "cnn_1d_cumulative_scale",
    "cnn_1d_image_scale",
    "cnn_1d_multiscale_image_scale",
    "cnn_1d_dilated_image_scale",
    "cnn_1d_attention_image_scale",
    "cnn_2d_rendered_images",
    "cnn_2d_residual_images",
}


def load_and_merge_predictions(paths: list[str]) -> pd.DataFrame:
    parts = []
    for p in paths:
        df = pd.read_csv(p, parse_dates=["date"])
        parts.append(df)
    merged = pd.concat(parts, ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "asset", "model_name"], keep="last")
    return merged.sort_values(["model_name", "date", "asset"]).reset_index(drop=True)


def oos_quality(df: pd.DataFrame) -> dict:
    """Compute OOS rank correlation and Sharpe from raw predictions."""
    per_date = (
        df.groupby("date")
        .apply(
            lambda g: g["signal_value"].corr(g["future_return"], method="spearman")
            if len(g) > 1
            else float("nan")
        )
        .dropna()
    )
    return {
        "oos_rank_corr_mean": float(per_date.mean()),
        "oos_rank_corr_std": float(per_date.std()),
        "oos_pct_positive": float((per_date > 0).mean()),
        "n_dates": int(len(per_date)),
        "n_folds": int(df["fold"].nunique()) if "fold" in df.columns else None,
    }


MU_CALIBRATION_DESC = {
    "method": "cross_sectional_zscore_x_expanding_return_std",
    "shrinkage": 0.4,
    "prior_std_before_14_dates": 0.05,
    "output_units": {
        "mu_hat_horizon": "expected return over horizon (daily_log_return scale summed)",
        "mu_hat_daily": "mu_hat_horizon / horizon (daily return scale)",
        "mu_raw_score": "raw CNN output prior to calibration (unitless score)",
    },
    "source_code": "make_ode_inputs.calibrate_mu_expanding",
}


def build_per_model_ode(
    model_df: pd.DataFrame,
    model_name: str,
    risk_df: pd.DataFrame | None,
    sigma_df: pd.DataFrame,
    horizon: int,
    sigma_window: int,
    output_dir: Path,
    risk_model_name: str | None = None,
) -> dict:
    assets = sorted(model_df["asset"].unique())
    out = output_dir / model_name
    out.mkdir(parents=True, exist_ok=True)

    mu_cal = calibrate_mu_expanding(model_df, horizon=horizon)
    mu_cal = mu_cal.rename(columns={"signal_value": "mu_raw_score"})

    mu_daily = forward_fill_to_daily(
        mu_cal,
        date_col="date",
        asset_col="asset",
        value_cols=["mu_hat_daily", "mu_hat_horizon", "mu_raw_score", "future_return"],
        horizon=horizon,
    )

    risk_daily = None
    if risk_df is not None:
        risk_cal = process_risk_signal(risk_df)
        risk_daily = forward_fill_to_daily(
            risk_cal,
            date_col="date",
            asset_col="asset",
            value_cols=["risk_score", "target"],
            horizon=horizon,
        )

    bundle = build_ode_bundle(mu_daily, risk_daily, sigma_df, assets)

    mu_export = mu_daily.copy()
    mu_export["date"] = mu_export["date"].dt.strftime("%Y-%m-%d")
    mu_export.to_csv(out / "mu_daily.csv", index=False)

    if risk_daily is not None:
        risk_export = risk_daily.copy()
        risk_export["date"] = risk_export["date"].dt.strftime("%Y-%m-%d")
        risk_export.to_csv(out / "risk_daily.csv", index=False)

    bundle_export = bundle.copy()
    bundle_export["date"] = pd.to_datetime(bundle_export["date"]).dt.strftime("%Y-%m-%d")
    bundle_export.to_csv(out / "ode_bundle.csv", index=False)

    quality = oos_quality(model_df)
    config = {
        "model_name": model_name,
        "mu_model": model_name,
        "risk_model": risk_model_name,
        "horizon": horizon,
        "sigma_window": sigma_window,
        "n_assets": len(assets),
        "assets": assets,
        "mu_date_range": [
            str(pd.to_datetime(mu_daily["date"]).min().date()),
            str(pd.to_datetime(mu_daily["date"]).max().date()),
        ],
        "sigma_date_range": [
            str(pd.to_datetime(sigma_df["date"]).min().date()),
            str(pd.to_datetime(sigma_df["date"]).max().date()),
        ],
        "n_dates_oos": quality["n_dates"],
        "oos_rank_corr_mean": quality["oos_rank_corr_mean"],
        "oos_pct_positive": quality["oos_pct_positive"],
        "mu_range_daily": [
            float(mu_cal["mu_hat_daily"].min()),
            float(mu_cal["mu_hat_daily"].max()),
        ],
        "mu_calibration": MU_CALIBRATION_DESC,
    }
    (out / "ode_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    return {**quality, "model_name": model_name}


def build_ensemble_predictions(all_preds: pd.DataFrame, top_models: list[str]) -> pd.DataFrame:
    """Average signal_value across top-k models per (date, asset).

    Keeps future_return, target, fold from the first model (they're identical across
    models since walk-forward uses the same target data)."""
    sub = all_preds[all_preds["model_name"].isin(top_models)].copy()
    grouped = (
        sub.groupby(["date", "asset"], as_index=False)
        .agg(
            signal_value=("signal_value", "mean"),
            future_return=("future_return", "first"),
            target=("target", "first"),
            fold=("fold", "first"),
        )
    )
    grouped["model_name"] = "ensemble_top3"
    grouped["selection_score"] = grouped["signal_value"]
    grouped["confidence"] = grouped["signal_value"].abs()
    return grouped.sort_values(["date", "asset"]).reset_index(drop=True)


def build_comparison(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.sort_values("oos_rank_corr_mean", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    cols = ["rank", "model_name", "oos_rank_corr_mean", "oos_rank_corr_std",
            "oos_pct_positive", "n_dates", "n_folds"]
    return df[[c for c in cols if c in df.columns]]


def build_comparison_md(df: pd.DataFrame, best_model: str) -> str:
    lines = [
        "# CNN Model Comparison — ODE Signal Quality",
        "",
        "Metric: mean OOS Spearman rank correlation (signal_value vs actual future_return).",
        "Higher = better mu(t) input for ODE.",
        "",
        "| rank | model | rank_corr | pct_positive_dates | n_folds |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {int(row['rank'])} | `{row['model_name']}` "
            f"| {row['oos_rank_corr_mean']:.4f} "
            f"| {row['oos_pct_positive']:.2%} "
            f"| {int(row['n_folds']) if pd.notna(row.get('n_folds')) else '-'} |"
        )
    lines += [
        "",
        f"**Recommended for ODE mu(t): `{best_model}`**",
        "",
        "## ODE connection",
        "- Each model's `ode_bundle.csv` contains `{asset}_mu`, `{asset}_sigma_ii`,",
        "  `{asset}_risk`, and all off-diagonal covariances aligned on a daily grid.",
        "- Use `{asset}_mu` → mu(t) vector.",
        "- Use `{asset}_sigma_ii` + `{a}_{b}_cov` → Sigma(t) matrix.",
        "- Use `{asset}_risk` → modulate gamma(t) or penalise mu.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-paths", nargs="+", required=True,
                   help="walk-forward prediction CSV files (can pass multiple to merge)")
    p.add_argument("--risk-path", default=None,
                   help="walk-forward predictions for downside_like target")
    p.add_argument("--risk-model", default="logistic_cumulative_scale")
    p.add_argument("--data-path", default="etfdata.csv")
    p.add_argument("--output-dir", default="ode_inputs_cnn")
    p.add_argument("--sigma-window", type=int, default=60)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--cnn-only", action="store_true", default=True,
                   help="only process CNN models (skip logistic)")
    p.add_argument("--ensemble-top-k", type=int, default=3,
                   help="build ensemble_top{k}/ bundle averaging signals from top-k CNN models (0 to disable)")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading predictions ...")
    all_preds = load_and_merge_predictions(args.pred_paths)
    found_models = sorted(all_preds["model_name"].unique())
    print(f"  models found: {found_models}")

    target_models = [m for m in found_models if m in CNN_MODELS] if args.cnn_only else found_models
    if not target_models:
        raise ValueError("No CNN models found in prediction files.")
    print(f"  CNN models to package: {target_models}")

    risk_df = None
    if args.risk_path:
        print(f"Loading risk signal from {args.risk_path} ...")
        risk_raw = pd.read_csv(args.risk_path, parse_dates=["date"])
        risk_df = risk_raw[risk_raw["model_name"] == args.risk_model].copy()
        print(f"  {len(risk_df)} rows for model '{args.risk_model}'")

    assets = sorted(all_preds["asset"].unique())
    print(f"Computing rolling Sigma (window={args.sigma_window}) ...")
    sigma_df = compute_rolling_sigma(args.data_path, assets, window=args.sigma_window)

    print("\nPackaging per-model ODE inputs ...")
    quality_rows = []
    for model_name in target_models:
        model_df = all_preds[all_preds["model_name"] == model_name].copy()
        print(f"  {model_name}: {len(model_df)} rows, {model_df['date'].nunique()} dates")
        quality = build_per_model_ode(
            model_df=model_df,
            model_name=model_name,
            risk_df=risk_df,
            sigma_df=sigma_df,
            horizon=args.horizon,
            sigma_window=args.sigma_window,
            output_dir=output_dir,
            risk_model_name=args.risk_model if risk_df is not None else None,
        )
        quality_rows.append(quality)

    comp_df = build_comparison(quality_rows)
    comp_df.to_csv(output_dir / "comparison.csv", index=False)

    if args.ensemble_top_k and args.ensemble_top_k >= 2:
        top_models = comp_df.head(args.ensemble_top_k)["model_name"].tolist()
        print(f"\nBuilding ensemble_top{args.ensemble_top_k} bundle from: {top_models}")
        ensemble_df = build_ensemble_predictions(all_preds, top_models)
        ensemble_quality = build_per_model_ode(
            model_df=ensemble_df,
            model_name=f"ensemble_top{args.ensemble_top_k}",
            risk_df=risk_df,
            sigma_df=sigma_df,
            horizon=args.horizon,
            sigma_window=args.sigma_window,
            output_dir=output_dir,
            risk_model_name=args.risk_model if risk_df is not None else None,
        )
        ensemble_config_path = output_dir / f"ensemble_top{args.ensemble_top_k}" / "ode_config.json"
        ensemble_config = json.loads(ensemble_config_path.read_text())
        ensemble_config["ensemble_members"] = top_models
        ensemble_config_path.write_text(json.dumps(ensemble_config, indent=2), encoding="utf-8")
        quality_rows.append(ensemble_quality)
        comp_df = build_comparison(quality_rows)
        comp_df.to_csv(output_dir / "comparison.csv", index=False)

    best = str(comp_df.iloc[0]["model_name"])
    md = build_comparison_md(comp_df, best)
    (output_dir / "comparison.md").write_text(md, encoding="utf-8")

    print("\n=== CNN ODE Signal Comparison ===")
    print(comp_df.to_string(index=False))
    print(f"\nBest model: {best}")
    print(f"Outputs: {output_dir}/")
    for m in target_models:
        files = list((output_dir / m).glob("*.csv"))
        print(f"  {m}/  ({len(files)} CSV files)")


if __name__ == "__main__":
    main()
