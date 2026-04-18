#!/usr/bin/env python3
"""Build ODE-ready inputs from walk-forward signal outputs.

Reads walk-forward predictions for:
  - future_return  → mu(t): expected return signal
  - downside_like  → risk(t): downside risk adjustment

Outputs to ode_inputs/:
  mu_daily.csv       date × asset daily expected return (return scale, forward-filled)
  risk_daily.csv     date × asset downside risk score (cross-sectional z-score)
  sigma_daily.csv    date × asset_pair rolling covariance (tidy format)
  ode_bundle.csv     wide format: all signals aligned on the same daily dates
  ode_summary.md     statistics and metadata
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_etf_csv, restrict_common_valid_sample, REQUIRED_PRICE_FIELDS, OPTIONAL_FIELDS


# ─── loaders ─────────────────────────────────────────────────────────────────

def load_wf_predictions(path: str, model_name: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    if model_name:
        df = df[df["model_name"] == model_name].copy()
    return df.sort_values(["date", "asset"]).reset_index(drop=True)


def auto_select_best_model(path: str, metric: str = "future_return_rank_correlation") -> str:
    """Pick the model with highest OOS metric from the comparison CSV."""
    comp_path = Path(path).parent / "walkforward_comparison.csv"
    if not comp_path.exists():
        raise FileNotFoundError(f"comparison file not found: {comp_path}")
    comp = pd.read_csv(comp_path)
    if metric not in comp.columns:
        metric = comp.columns[2]
    best = comp.sort_values(metric, ascending=False).iloc[0]["model_name"]
    print(f"  auto-selected model '{best}' by {metric}={comp.iloc[0][metric]:.4f}")
    return str(best)


# ─── mu calibration ───────────────────────────────────────────────────────────

def calibrate_mu_expanding(mu_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Calibrate signal_value → daily expected return using cross-sectional z-score.

    For each date t:
      1. Cross-sectional z-score: z(t,i) = (s(t,i) - mean_i s(t)) / std_i s(t)
      2. Scale by expanding historical return std (no lookahead).
      3. Divide by horizon to convert horizon-return → daily.

    The 0.4 shrinkage factor is conservative: assumes alpha ≈ 40% of total return vol.
    """
    mu_df = mu_df.sort_values("date").reset_index(drop=True)
    unique_dates = sorted(mu_df["date"].unique())

    expanding_returns: list[float] = []
    calibrated: list[pd.DataFrame] = []

    for date in unique_dates:
        mask = mu_df["date"] == date
        rows = mu_df[mask].copy()
        signals = rows["signal_value"].values.astype(float)
        n_assets = len(signals)

        if n_assets > 1 and signals.std() > 1e-10:
            z = (signals - signals.mean()) / signals.std()
        else:
            z = np.zeros(n_assets)

        if len(expanding_returns) >= 14:
            ret_std = float(np.std(expanding_returns))
        else:
            ret_std = 0.05  # prior: 5% per horizon

        mu_horizon = z * ret_std * 0.4
        rows["mu_hat_horizon"] = mu_horizon
        rows["mu_hat_daily"] = mu_horizon / horizon
        calibrated.append(rows)

        expanding_returns.extend(rows["future_return"].tolist())

    return pd.concat(calibrated, ignore_index=True)


# ─── risk signal ─────────────────────────────────────────────────────────────

def process_risk_signal(risk_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize downside predictions to cross-sectional z-scores.

    Higher risk_score = more expected downside = riskier.
    Use this to reduce allocation (penalize mu or raise gamma(t)).
    """
    risk_df = risk_df.sort_values("date").reset_index(drop=True)
    unique_dates = sorted(risk_df["date"].unique())

    processed: list[pd.DataFrame] = []
    for date in unique_dates:
        mask = risk_df["date"] == date
        rows = risk_df[mask].copy()
        signals = rows["signal_value"].values.astype(float)
        n = len(signals)
        if n > 1 and signals.std() > 1e-10:
            z = (signals - signals.mean()) / signals.std()
        else:
            z = np.zeros(n)
        rows["risk_score"] = z
        processed.append(rows)
    return pd.concat(processed, ignore_index=True)


# ─── forward-fill to daily ────────────────────────────────────────────────────

def forward_fill_to_daily(
    panel: pd.DataFrame,
    date_col: str,
    asset_col: str,
    value_cols: list[str],
    horizon: int,
) -> pd.DataFrame:
    """Forward-fill sparse signal dates to daily business-day grid.

    Each signal is valid for [signal_date, signal_date + horizon business days).
    """
    assets = sorted(panel[asset_col].unique())
    min_date = panel[date_col].min()
    max_date = panel[date_col].max() + pd.offsets.BDay(horizon + 5)
    daily_idx = pd.bdate_range(str(min_date.date()), str(max_date.date()))

    pieces: list[pd.DataFrame] = []
    for asset in assets:
        sub = panel[panel[asset_col] == asset].copy()
        sub = sub.sort_values(date_col).set_index(date_col)[value_cols]
        sub = sub[~sub.index.duplicated(keep="last")]
        daily = sub.reindex(daily_idx).ffill()
        daily.index.name = "date"
        daily["asset"] = asset
        pieces.append(daily.reset_index())

    return pd.concat(pieces, ignore_index=True)


# ─── rolling covariance ───────────────────────────────────────────────────────

def compute_rolling_sigma(
    data_path: str, assets: list[str], window: int = 60
) -> pd.DataFrame:
    """Compute daily rolling covariance matrices from log returns.

    Returns tidy DataFrame: date, asset_i, asset_j, cov_ij, corr_ij.
    Only the upper triangle + diagonal are stored (symmetric matrix).
    """
    _, long_panel, _, _, _ = load_etf_csv(data_path)

    common_panel, _ = restrict_common_valid_sample(
        long_panel, selected_assets=assets, required_fields=REQUIRED_PRICE_FIELDS
    )

    close_wide = (
        common_panel.pivot(index="date", columns="asset", values="close")
        .sort_index()[sorted(assets)]
    )
    log_ret = np.log(close_wide / close_wide.shift(1)).dropna()

    assets_sorted = sorted(assets)
    rows: list[dict] = []

    for i in range(len(log_ret)):
        if i < window - 1:
            continue
        date = log_ret.index[i]
        window_ret = log_ret.iloc[i - window + 1 : i + 1]
        cov = window_ret.cov().values
        std = window_ret.std().values
        for ai, a1 in enumerate(assets_sorted):
            for aj, a2 in enumerate(assets_sorted):
                if aj < ai:
                    continue
                c = cov[ai, aj]
                corr = c / (std[ai] * std[aj]) if std[ai] * std[aj] > 1e-12 else float("nan")
                rows.append(
                    {
                        "date": date,
                        "asset_i": a1,
                        "asset_j": a2,
                        "cov_daily": c,
                        "corr": corr,
                    }
                )

    return pd.DataFrame(rows)


# ─── bundle assembly ──────────────────────────────────────────────────────────

def build_ode_bundle(
    mu_daily: pd.DataFrame,
    risk_daily: pd.DataFrame | None,
    sigma_df: pd.DataFrame,
    assets: list[str],
) -> pd.DataFrame:
    """Assemble wide-format ODE bundle.

    Columns:
      date
      {asset}_mu          daily expected return estimate
      {asset}_sigma_ii    diagonal variance (daily)
      {asset}_risk        downside risk z-score (if available)
      {a1}_{a2}_cov       off-diagonal covariances

    Each row = one date. All assets and covariances aligned.
    """
    assets_s = sorted(assets)
    dates = sorted(mu_daily["date"].unique())

    mu_wide = mu_daily.pivot(index="date", columns="asset", values="mu_hat_daily")
    mu_wide.columns = [f"{a}_mu" for a in mu_wide.columns]

    sigma_diag = sigma_df[sigma_df["asset_i"] == sigma_df["asset_j"]].copy()
    sigma_diag_wide = sigma_diag.pivot(index="date", columns="asset_i", values="cov_daily")
    sigma_diag_wide.columns = [f"{a}_sigma_ii" for a in sigma_diag_wide.columns]

    bundle = mu_wide.join(sigma_diag_wide, how="outer")

    if "mu_raw_score" in mu_daily.columns:
        raw_wide = mu_daily.pivot(index="date", columns="asset", values="mu_raw_score")
        raw_wide.columns = [f"{a}_mu_raw" for a in raw_wide.columns]
        bundle = bundle.join(raw_wide, how="outer")

    if risk_daily is not None:
        risk_wide = risk_daily.pivot(index="date", columns="asset", values="risk_score")
        risk_wide.columns = [f"{a}_risk" for a in risk_wide.columns]
        bundle = bundle.join(risk_wide, how="outer")

    offdiag = sigma_df[sigma_df["asset_i"] != sigma_df["asset_j"]].copy()
    offdiag["pair"] = offdiag["asset_i"] + "_" + offdiag["asset_j"] + "_cov"
    offdiag_wide = offdiag.pivot_table(index="date", columns="pair", values="cov_daily")
    bundle = bundle.join(offdiag_wide, how="outer")

    bundle = bundle.sort_index()
    bundle.index.name = "date"
    return bundle.reset_index()


# ─── summary report ──────────────────────────────────────────────────────────

def build_summary(
    mu_daily: pd.DataFrame,
    risk_daily: pd.DataFrame | None,
    sigma_df: pd.DataFrame,
    mu_model: str,
    risk_model: str | None,
    config: dict,
) -> str:
    assets = sorted(mu_daily["asset"].unique())
    date_range = f"{mu_daily['date'].min().date()} ~ {mu_daily['date'].max().date()}"

    mu_stats = (
        mu_daily.groupby("asset")["mu_hat_daily"]
        .describe()[["mean", "std", "min", "max"]]
        .round(6)
    )

    lines = [
        "# ODE Input Bundle Summary",
        "",
        "## Configuration",
        f"- mu model: `{mu_model}`",
        f"- risk model: `{risk_model or 'none'}`",
        f"- sigma window: {config.get('sigma_window', 60)} days",
        f"- date range: {date_range}",
        f"- assets ({len(assets)}): {', '.join(assets)}",
        "",
        "## mu(t) statistics — daily expected return per asset",
        mu_stats.to_string(),
        "",
        "## Files produced",
        "| File | Description |",
        "| --- | --- |",
        "| `mu_daily.csv` | date × asset, mu_hat_daily (return scale) |",
        "| `risk_daily.csv` | date × asset, risk_score (z-score, higher=riskier) |",
        "| `sigma_daily.csv` | date × asset_i × asset_j, daily covariance |",
        "| `ode_bundle.csv` | wide format, all signals on same daily grid |",
        "",
        "## How to connect to ODE",
        "- `{asset}_mu` → mu(t) vector input to ODE expected-return term",
        "- `{asset}_sigma_ii` + `{a}_{b}_cov` → Sigma(t) matrix",
        "- `{asset}_risk` → scale gamma(t) = gamma_0 * exp(k * risk_score) or",
        "  subtract from mu: mu_adj(t,i) = mu(t,i) - delta * risk_score(t,i)",
        "- All signals are forward-filled to daily frequency.",
        "- Sigma is rolling window (no lookahead).",
        "- mu is cross-sectionally calibrated with expanding return std (no lookahead).",
        "",
        "## Quality check",
    ]

    if "future_return_rank_correlation" in mu_daily.columns:
        per_date_corr = (
            mu_daily.groupby("date")
            .apply(lambda g: g["mu_hat_daily"].corr(g["future_return"], method="spearman") if len(g) > 1 else float("nan"))
            .dropna()
        )
        lines.append(f"- mean OOS Spearman(mu_hat, future_return) = {per_date_corr.mean():.4f}")
        lines.append(f"- std = {per_date_corr.std():.4f}")
        lines.append(f"- % dates positive = {(per_date_corr > 0).mean():.2%}")

    if risk_daily is not None and "target" in risk_daily.columns:
        per_date_risk_corr = (
            risk_daily.groupby("date")
            .apply(lambda g: g["risk_score"].corr(g["target"], method="spearman") if len(g) > 1 else float("nan"))
            .dropna()
        )
        lines.append(f"- mean OOS Spearman(risk_score, actual_downside) = {per_date_risk_corr.mean():.4f}")

    return "\n".join(lines) + "\n"


# ─── main ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build ODE-ready inputs from walk-forward signals")
    p.add_argument("--mu-path", required=True, help="walk-forward predictions CSV for future_return target")
    p.add_argument("--mu-model", default=None, help="model name to use for mu; auto-selects best if omitted")
    p.add_argument("--risk-path", default=None, help="walk-forward predictions CSV for downside_like target (optional)")
    p.add_argument("--risk-model", default="logistic_cumulative_scale")
    p.add_argument("--data-path", default="etfdata.csv")
    p.add_argument("--output-dir", default="ode_inputs")
    p.add_argument("--sigma-window", type=int, default=60, help="rolling window (days) for covariance")
    p.add_argument("--horizon", type=int, default=20)
    return p


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── load mu signal ──
    print(f"Loading mu predictions from {args.mu_path} ...")
    mu_model = args.mu_model or auto_select_best_model(args.mu_path)
    mu_raw = load_wf_predictions(args.mu_path, model_name=mu_model)
    print(f"  {len(mu_raw)} rows, {mu_raw['date'].nunique()} dates, {mu_raw['asset'].nunique()} assets")

    # ── calibrate mu ──
    print("Calibrating mu signal (cross-sectional z-score × expanding return std) ...")
    mu_cal = calibrate_mu_expanding(mu_raw, horizon=args.horizon)

    # ── forward-fill mu to daily ──
    print(f"Forward-filling mu to daily grid (horizon={args.horizon}) ...")
    mu_daily = forward_fill_to_daily(
        mu_cal, date_col="date", asset_col="asset",
        value_cols=["mu_hat_daily", "mu_hat_horizon", "future_return"],
        horizon=args.horizon,
    )

    # ── load and process risk signal (optional) ──
    risk_daily: pd.DataFrame | None = None
    risk_model_used: str | None = None
    if args.risk_path:
        print(f"Loading risk predictions from {args.risk_path} ...")
        risk_model_used = args.risk_model
        risk_raw = load_wf_predictions(args.risk_path, model_name=risk_model_used)
        print(f"  {len(risk_raw)} rows")
        risk_cal = process_risk_signal(risk_raw)
        risk_daily = forward_fill_to_daily(
            risk_cal, date_col="date", asset_col="asset",
            value_cols=["risk_score", "target"],
            horizon=args.horizon,
        )

    # ── rolling covariance ──
    assets = sorted(mu_raw["asset"].unique())
    print(f"Computing rolling Sigma (window={args.sigma_window}) for {len(assets)} assets ...")
    sigma_df = compute_rolling_sigma(args.data_path, assets, window=args.sigma_window)
    print(f"  {sigma_df['date'].nunique()} dates of covariance matrices")

    # ── assemble bundle ──
    print("Assembling ODE bundle ...")
    bundle = build_ode_bundle(mu_daily, risk_daily, sigma_df, assets)

    # ── export ──
    mu_export = mu_daily.copy()
    mu_export["date"] = mu_export["date"].dt.strftime("%Y-%m-%d")
    mu_export.to_csv(output_dir / "mu_daily.csv", index=False)

    sigma_export = sigma_df.copy()
    sigma_export["date"] = sigma_export["date"].dt.strftime("%Y-%m-%d")
    sigma_export.to_csv(output_dir / "sigma_daily.csv", index=False)

    if risk_daily is not None:
        risk_export = risk_daily.copy()
        risk_export["date"] = risk_export["date"].dt.strftime("%Y-%m-%d")
        risk_export.to_csv(output_dir / "risk_daily.csv", index=False)

    bundle_export = bundle.copy()
    bundle_export["date"] = pd.to_datetime(bundle_export["date"]).dt.strftime("%Y-%m-%d")
    bundle_export.to_csv(output_dir / "ode_bundle.csv", index=False)

    config_meta = {
        "mu_model": mu_model,
        "risk_model": risk_model_used,
        "sigma_window": args.sigma_window,
        "horizon": args.horizon,
        "n_assets": len(assets),
        "assets": assets,
        "mu_date_range": [str(mu_daily["date"].min().date()), str(mu_daily["date"].max().date())],
        "sigma_date_range": [str(sigma_df["date"].min().date()), str(sigma_df["date"].max().date())],
    }
    (output_dir / "ode_config.json").write_text(
        json.dumps(config_meta, indent=2), encoding="utf-8"
    )

    summary = build_summary(mu_cal, risk_cal if args.risk_path else None, sigma_df, mu_model, risk_model_used, config_meta)
    (output_dir / "ode_summary.md").write_text(summary, encoding="utf-8")

    print("\n=== ODE Bundle Ready ===")
    mu_stats = mu_cal.groupby("asset")["mu_hat_daily"].agg(["mean", "std"]).round(6)
    print("\nmu_hat_daily per asset:")
    print(mu_stats.to_string())
    print(f"\n→ {output_dir}/")
    for f in sorted(output_dir.glob("*.csv")):
        n = len(pd.read_csv(f))
        print(f"   {f.name}  ({n:,} rows)")


if __name__ == "__main__":
    main()
