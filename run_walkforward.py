#!/usr/bin/env python3
"""Walk-forward OOS evaluation runner.

Recommended invocation:
  python run_walkforward.py --lookback 60 --horizon 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import PipelineConfig
from src.data.loader import (
    OPTIONAL_FIELDS,
    REQUIRED_PRICE_FIELDS,
    load_etf_csv,
    restrict_common_valid_sample,
)
from src.pipeline import _json_default, _markdown_table, _to_records, build_samples, set_global_seed
from src.walkforward import run_walkforward

_DEFAULT_MODELS = [
    "logistic_cumulative_scale",
    "logistic_image_scale",
    "cnn_1d_image_scale",
    "cnn_1d_attention_image_scale",
    "cnn_1d_dilated_image_scale",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Walk-forward OOS evaluation")
    p.add_argument("--data-path", default="etfdata.csv")
    p.add_argument("--output-dir", default="outputs_walkforward")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--label-mode", default="regression")
    p.add_argument("--target-name", default="future_return")
    p.add_argument("--models", nargs="*", default=None, help="subset of models; default uses 5 standard models")
    p.add_argument("--wf-min-train-days", type=int, default=500)
    p.add_argument("--wf-val-days", type=int, default=60)
    p.add_argument("--wf-test-days", type=int, default=60)
    p.add_argument("--cnn-epochs", type=int, default=8)
    p.add_argument("--cnn-repeats", type=int, default=1)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_parser().parse_args()
    enabled_models = args.models if args.models else _DEFAULT_MODELS

    config = PipelineConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        lookback=args.lookback,
        horizon=args.horizon,
        label_mode=args.label_mode,
        target_name=args.target_name,
        enabled_models=enabled_models,
        wf_min_train_days=args.wf_min_train_days,
        wf_val_days=args.wf_val_days,
        wf_test_days=args.wf_test_days,
        cnn_epochs=args.cnn_epochs,
        cnn_repeats=args.cnn_repeats,
        top_k=args.top_k,
        seed=args.seed,
        sample_preview_count=0,
    )

    set_global_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {config.data_path} ...")
    _, long_panel, assets, fields, _ = load_etf_csv(config.data_path)

    required_fields = REQUIRED_PRICE_FIELDS.copy()
    if config.include_volume:
        if "volume" in fields:
            required_fields.extend(OPTIONAL_FIELDS)
        else:
            config.include_volume = False

    common_panel, _ = restrict_common_valid_sample(
        long_panel,
        selected_assets=assets,
        required_fields=required_fields,
    )
    if common_panel.empty:
        raise ValueError("no common valid sample after filtering")

    print(f"Building samples  (lookback={config.lookback}, horizon={config.horizon}) ...")
    bundle, prep = build_samples(common_panel, config)
    print(f"  total samples: {prep['n_samples']}  shape: {prep['sequence_shape']}")

    print(f"\nRunning walk-forward OOS  (min_train={config.wf_min_train_days}, val={config.wf_val_days}, test={config.wf_test_days}) ...")
    oos_df, comparison_df = run_walkforward(bundle, config)

    oos_export = oos_df.copy()
    oos_export["date"] = oos_export["date"].dt.strftime("%Y-%m-%d")
    oos_export.to_csv(output_dir / "walkforward_predictions.csv", index=False)
    comparison_df.to_csv(output_dir / "walkforward_comparison.csv", index=False)

    report_lines = [
        "# Walk-Forward OOS Evaluation",
        "",
        "## Configuration",
        f"- lookback={config.lookback}, horizon={config.horizon}",
        f"- label_mode={config.label_mode}, target={config.target_name}",
        f"- wf_min_train_days={config.wf_min_train_days}  wf_val_days={config.wf_val_days}  wf_test_days={config.wf_test_days}",
        f"- models: {', '.join(enabled_models)}",
        f"- total OOS predictions: {len(oos_df)}",
        "",
        "## Model Comparison (Aggregated OOS)",
        _markdown_table(comparison_df),
        "",
        "## ODE Connection",
        "- Use the best model's `signal_value` as mu(t) proxy after calibration.",
        "- `future_return_rank_correlation` is the primary OOS quality indicator.",
    ]
    report_text = "\n".join(report_lines) + "\n"
    (output_dir / "walkforward_report.md").write_text(report_text, encoding="utf-8")

    payload = {
        "config": config.to_dict(),
        "prep_summary": {k: v for k, v in prep.items() if k != "sample_previews"},
        "comparison": _to_records(comparison_df),
    }
    (output_dir / "walkforward_results.json").write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )

    print("\n=== Walk-Forward Results ===")
    display_cols = [c for c in ["model_name", "n_folds", "top_k_sharpe", "future_return_rank_correlation",
                                 "target_rank_correlation", "rmse", "top_k_cumulative_return"] if c in comparison_df.columns]
    print(comparison_df[display_cols].to_string(index=False))
    print(f"\nOutputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
