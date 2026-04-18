from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PipelineConfig
from src.pipeline import run_pipeline
from src.reporting import markdown_table, sort_summary_frame, write_json

DEFAULT_ABLATION_VARIANTS = [
    ("ohlc_only", False, False),
    ("ohlc_ma", True, False),
    ("ohlc_volume", False, True),
    ("ohlc_ma_volume", True, True),
]
ABLATION_VARIANT_MAP = {name: (name, include_ma, include_volume) for name, include_ma, include_volume in DEFAULT_ABLATION_VARIANTS}


def resolve_ablation_variants(names: list[str] | None = None) -> list[tuple[str, bool, bool]]:
    if not names:
        return list(DEFAULT_ABLATION_VARIANTS)
    unknown = [name for name in names if name not in ABLATION_VARIANT_MAP]
    if unknown:
        raise ValueError(f"unknown ablation variants: {', '.join(unknown)}")
    return [ABLATION_VARIANT_MAP[name] for name in names]


def run_cnn_ablation(
    base_config: PipelineConfig,
    variants: list[tuple[str, bool, bool]],
) -> dict:
    root = Path(base_config.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    per_variant_payloads: dict[str, dict] = {}

    for variant_name, include_ma, include_volume in variants:
        config = PipelineConfig(**base_config.to_dict())
        config.output_dir = str(root / variant_name)
        config.include_moving_average = include_ma
        config.include_volume = include_volume
        payload = run_pipeline(config)
        per_variant_payloads[variant_name] = payload
        best_row = dict(payload["comparison"][0])
        best_row["variant_name"] = variant_name
        best_row["include_moving_average"] = include_ma
        best_row["include_volume"] = include_volume
        best_row["best_signal_file"] = payload["paths"]["best_signal_file"]
        best_row["report_file"] = payload["paths"]["report"]
        rows.append(best_row)

    summary_df = sort_summary_frame(pd.DataFrame(rows))

    summary_csv = root / "cnn_ablation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    summary_md = root / "cnn_ablation_summary.md"
    best_variant = summary_df.iloc[0]
    lines = [
        "# CNN Ablation Summary",
        "",
        f"- Best variant: `{best_variant['variant_name']}`",
        f"- Best model: `{best_variant['model_name']}`",
        f"- top-k Sharpe: `{best_variant['top_k_sharpe']:.4f}`",
        f"- future-return rank correlation: `{best_variant['future_return_rank_correlation']:.4f}`",
        "",
        markdown_table(summary_df, float_precision=4),
    ]
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "variants": [
            {
                "variant_name": variant_name,
                "include_moving_average": include_ma,
                "include_volume": include_volume,
            }
            for variant_name, include_ma, include_volume in variants
        ],
        "summary": summary_df.to_dict(orient="records"),
        "paths": {
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
        },
        "per_variant_payloads": per_variant_payloads,
    }
    summary_json = root / "cnn_ablation_summary.json"
    payload["paths"]["summary_json"] = str(summary_json)
    write_json(summary_json, payload)
    return payload
