from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PipelineConfig
from src.features.labels import get_target_metadata
from src.pipeline import run_pipeline
from src.reporting import markdown_table, sort_summary_frame, write_json

DEFAULT_REGRESSION_TARGETS = ["future_return", "downside_like", "skew_like"]


def run_regression_suite(base_config: PipelineConfig, targets: list[str]) -> dict:
    suite_root = Path(base_config.output_dir)
    suite_root.mkdir(parents=True, exist_ok=True)

    suite_rows: list[dict] = []
    per_target_payloads: dict[str, dict] = {}

    for target_name in targets:
        config = PipelineConfig(**base_config.to_dict())
        config.label_mode = "regression"
        config.target_name = target_name
        config.output_dir = str(suite_root / target_name)

        payload = run_pipeline(config)
        per_target_payloads[target_name] = payload
        target_meta = get_target_metadata("regression", target_name)
        best_row = dict(payload["comparison"][0])
        best_row["target_name"] = target_name
        best_row["canonical_target_name"] = target_meta["canonical_target_name"]
        best_row["signal_role"] = target_meta["signal_role"]
        best_row["best_model_name"] = payload["best_model_name"]
        best_row["target_output_dir"] = str(config.output_dir)
        best_row["best_signal_file"] = payload["paths"]["best_signal_file"]
        suite_rows.append(best_row)

    summary_df = pd.DataFrame(suite_rows)
    preferred_columns = [
        "target_name",
        "canonical_target_name",
        "signal_role",
        "best_model_name",
        "rmse",
        "mae",
        "target_rank_correlation",
        "future_return_rank_correlation",
        "top_k_cumulative_return",
        "top_k_sharpe",
        "top_k_hit_rate",
        "top_bottom_spread_mean",
        "turnover",
        "target_output_dir",
        "best_signal_file",
    ]
    summary_df = summary_df[[column for column in preferred_columns if column in summary_df.columns]]
    summary_df = sort_summary_frame(summary_df)

    summary_csv_path = suite_root / "regression_suite_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    report_lines = [
        "# Regression Target Suite",
        "",
        "Targets run: " + ", ".join(targets),
        "",
        markdown_table(summary_df, float_precision=4),
        "",
        "Per-target reports:",
    ]
    for target_name, payload in per_target_payloads.items():
        report_lines.append(f"- {target_name}: {payload['paths']['report']}")

    summary_md_path = suite_root / "regression_suite_summary.md"
    summary_md_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary_payload = {
        "targets": targets,
        "summary": summary_df.to_dict(orient="records"),
        "per_target_payloads": per_target_payloads,
        "paths": {
            "summary_csv": str(summary_csv_path),
            "summary_md": str(summary_md_path),
        },
    }
    summary_json_path = suite_root / "regression_suite_summary.json"
    summary_payload["paths"]["summary_json"] = str(summary_json_path)
    write_json(summary_json_path, summary_payload)
    return summary_payload
