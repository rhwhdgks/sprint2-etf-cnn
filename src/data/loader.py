from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_PRICE_FIELDS = ["open", "high", "low", "close"]
OPTIONAL_FIELDS = ["volume"]


def load_etf_csv(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], pd.DataFrame]:
    raw = pd.read_csv(csv_path, header=[0, 1])
    columns = list(raw.columns)
    columns[0] = ("date", "date")
    raw.columns = pd.MultiIndex.from_tuples(columns)

    raw = raw[raw[("date", "date")].ne("date")].copy()
    raw[("date", "date")] = pd.to_datetime(raw[("date", "date")], errors="coerce")
    raw = raw.dropna(subset=[("date", "date")]).copy()

    for column in raw.columns[1:]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    assets = sorted({column[1] for column in raw.columns if column[0] != "date"})
    fields = sorted({column[0] for column in raw.columns if column[0] != "date"})

    coverage_rows: list[dict] = []
    for asset in assets:
        present_fields = [field for field in REQUIRED_PRICE_FIELDS + OPTIONAL_FIELDS if (field, asset) in raw.columns]
        if not present_fields:
            continue
        valid = raw[[(field, asset) for field in present_fields]].notna().all(axis=1)
        coverage_rows.append(
            {
                "asset": asset,
                "first_valid_all_present_fields": raw.loc[valid, ("date", "date")].min(),
                "last_valid_all_present_fields": raw.loc[valid, ("date", "date")].max(),
                "n_valid_all_present_fields": int(valid.sum()),
                "available_fields": ",".join(present_fields),
            }
        )

    long_frames = []
    for asset in assets:
        frame = pd.DataFrame({"date": raw[("date", "date")], "asset": asset})
        for field in fields:
            if (field, asset) in raw.columns:
                frame[field] = raw[(field, asset)]
            else:
                frame[field] = pd.NA
        long_frames.append(frame)

    long_panel = pd.concat(long_frames, ignore_index=True)
    long_panel = long_panel.sort_values(["date", "asset"]).reset_index(drop=True)
    coverage_df = pd.DataFrame(coverage_rows)
    return raw, long_panel, assets, fields, coverage_df


def restrict_common_valid_sample(
    long_panel: pd.DataFrame,
    selected_assets: list[str],
    required_fields: list[str],
) -> tuple[pd.DataFrame, dict]:
    panel = long_panel[long_panel["asset"].isin(selected_assets)].copy()
    valid_rows = panel[required_fields].notna().all(axis=1)
    valid_counts = panel.loc[valid_rows].groupby("date")["asset"].nunique()
    common_dates = valid_counts[valid_counts == len(selected_assets)].index
    common_panel = panel[panel["date"].isin(common_dates)].copy()
    common_panel = common_panel.sort_values(["date", "asset"]).reset_index(drop=True)

    summary = {
        "selected_assets": selected_assets,
        "required_fields": required_fields,
        "n_common_dates": int(common_panel["date"].nunique()),
        "n_common_rows": int(len(common_panel)),
        "common_start": common_panel["date"].min().isoformat() if not common_panel.empty else None,
        "common_end": common_panel["date"].max().isoformat() if not common_panel.empty else None,
    }
    return common_panel, summary


def build_dataset_summary(
    csv_path: str,
    fields: list[str],
    assets: list[str],
    common_panel: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> dict:
    date_index = common_panel["date"].drop_duplicates().sort_values()
    median_step_days = None
    if len(date_index) > 1:
        diffs = date_index.diff().dropna().dt.total_seconds() / 86400.0
        median_step_days = float(diffs.median())

    asset_rows = []
    for asset, asset_frame in common_panel.groupby("asset"):
        asset_rows.append(
            {
                "asset": asset,
                "start": asset_frame["date"].min().isoformat(),
                "end": asset_frame["date"].max().isoformat(),
                "rows": int(len(asset_frame)),
            }
        )

    return {
        "data_file": str(Path(csv_path)),
        "available_assets": assets,
        "available_fields": fields,
        "common_sample_assets": sorted(common_panel["asset"].unique().tolist()),
        "common_sample_start": common_panel["date"].min().isoformat(),
        "common_sample_end": common_panel["date"].max().isoformat(),
        "common_sample_dates": int(common_panel["date"].nunique()),
        "common_sample_rows": int(len(common_panel)),
        "median_step_days": median_step_days,
        "asset_rows": asset_rows,
        "raw_asset_coverage": coverage_df.to_dict(orient="records"),
    }
