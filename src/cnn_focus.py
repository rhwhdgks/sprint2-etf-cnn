from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PipelineConfig
from src.pipeline import run_pipeline
from src.reporting import markdown_table, sort_summary_frame, write_json

DEFAULT_CNN_MODELS = [
    "cnn_1d_cumulative_scale",
    "cnn_1d_image_scale",
    "cnn_1d_multiscale_image_scale",
    "cnn_1d_dilated_cumulative_scale",
    "cnn_1d_dilated_image_scale",
    "cnn_2d_rendered_images",
    "cnn_2d_residual_images",
]


def run_cnn_focus_experiment(base_config: PipelineConfig, windows: list[tuple[int, int]], models: list[str]) -> dict:
    suite_root = Path(base_config.output_dir)
    suite_root.mkdir(parents=True, exist_ok=True)

    best_rows = []
    all_rows = []
    per_window_payloads: dict[str, dict] = {}

    for lookback, horizon in windows:
        tag = f"lookback_{lookback}_horizon_{horizon}"
        config = PipelineConfig(**base_config.to_dict())
        config.lookback = lookback
        config.horizon = horizon
        config.enabled_models = models
        config.output_dir = str(suite_root / tag)

        payload = run_pipeline(config)
        per_window_payloads[tag] = payload
        comparison_df = pd.DataFrame(payload["comparison"])
        comparison_df["lookback"] = lookback
        comparison_df["horizon"] = horizon
        comparison_df["window_tag"] = tag
        all_rows.extend(comparison_df.to_dict(orient="records"))

        best = dict(comparison_df.iloc[0].to_dict())
        best["best_signal_file"] = payload["paths"]["best_signal_file"]
        best["report_file"] = payload["paths"]["report"]
        best_rows.append(best)

    best_df = sort_summary_frame(pd.DataFrame(best_rows))
    all_df = pd.DataFrame(all_rows).sort_values(
        ["lookback", "horizon", "top_k_sharpe", "future_return_rank_correlation"],
        ascending=[True, True, False, False],
        na_position="last",
    ).reset_index(drop=True)

    best_csv = suite_root / "cnn_focus_best_by_window.csv"
    all_csv = suite_root / "cnn_focus_all_results.csv"
    best_df.to_csv(best_csv, index=False)
    all_df.to_csv(all_csv, index=False)

    overall_best = best_df.iloc[0]
    lines = [
        "# CNN Focus Summary",
        "",
        "Windows: " + ", ".join(f"{lookback}/{horizon}" for lookback, horizon in windows),
        "Models: " + ", ".join(models),
        "",
        "## Best CNN by Window",
        markdown_table(best_df, float_precision=4),
        "",
        "## Overall Recommendation",
        f"- Best overall CNN: `{overall_best['model_name']}` at `lookback={int(overall_best['lookback'])}`, `horizon={int(overall_best['horizon'])}`.",
        f"- top-k Sharpe: `{overall_best['top_k_sharpe']:.4f}`",
        f"- top-k cumulative return: `{overall_best['top_k_cumulative_return']:.4f}`",
        f"- future-return rank correlation: `{overall_best['future_return_rank_correlation']:.4f}`",
    ]
    summary_md = suite_root / "cnn_focus_summary.md"
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "windows": [{"lookback": lookback, "horizon": horizon} for lookback, horizon in windows],
        "models": models,
        "best_by_window": best_df.to_dict(orient="records"),
        "paths": {
            "best_csv": str(best_csv),
            "all_csv": str(all_csv),
            "summary_md": str(summary_md),
        },
        "per_window_payloads": per_window_payloads,
    }
    summary_json = suite_root / "cnn_focus_summary.json"
    payload["paths"]["summary_json"] = str(summary_json)
    write_json(summary_json, payload)
    return payload
