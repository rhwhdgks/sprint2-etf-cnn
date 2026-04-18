from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SORT_PRIORITY = ("top_k_sharpe", "future_return_rank_correlation", "rmse")
DEFAULT_SORT_ASCENDING = (False, False, True)


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def markdown_table(df: pd.DataFrame, float_precision: int = 4) -> str:
    headers = list(df.columns)

    def format_cell(value) -> str:
        if isinstance(value, float):
            if np.isnan(value):
                return "nan"
            return f"{value:.{float_precision}f}"
        return str(value)

    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, series in df.iterrows():
        rows.append("| " + " | ".join(format_cell(series[column]) for column in headers) + " |")
    return "\n".join(rows)


def sort_summary_frame(
    df: pd.DataFrame,
    columns: tuple[str, ...] = DEFAULT_SORT_PRIORITY,
    ascending: tuple[bool, ...] = DEFAULT_SORT_ASCENDING,
) -> pd.DataFrame:
    available_columns = [column for column in columns if column in df.columns]
    available_ascending = [flag for column, flag in zip(columns, ascending) if column in df.columns]
    if not available_columns:
        return df.reset_index(drop=True)
    return df.sort_values(
        available_columns,
        ascending=available_ascending,
        na_position="last",
    ).reset_index(drop=True)


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
