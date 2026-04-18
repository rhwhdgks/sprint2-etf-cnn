from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import PipelineConfig
from src.data.discovery import discover_local_files
from src.data.loader import (
    OPTIONAL_FIELDS,
    REQUIRED_PRICE_FIELDS,
    build_dataset_summary,
    load_etf_csv,
    restrict_common_valid_sample,
)
from src.eval.metrics import compute_prediction_metrics, mean_rank_correlation, top_k_backtest
from src.features.labels import compute_auxiliary_targets, get_target_metadata, resolve_target
from src.features.scaling import channel_names, cumulative_return_scale_window, image_scale_window
from src.images.chart_renderer import render_jiang_chart, save_chart_preview
from src.models.baselines import fit_linear_model, predict_linear_model
from src.models.cnn import build_cnn_model, fit_torch_model, predict_torch_model


@dataclass
class SampleBundle:
    metadata: pd.DataFrame
    cumulative_sequences: np.ndarray
    image_sequences: np.ndarray
    chart_images: np.ndarray
    sequence_channels: list[str]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    return value


def _to_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for row in df.to_dict(orient="records"):
        clean = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                clean[key] = value.isoformat()
            elif isinstance(value, (np.integer, np.floating)):
                clean[key] = value.item()
            else:
                clean[key] = value
        records.append(clean)
    return records


def _markdown_table(df: pd.DataFrame, float_precision: int = 4) -> str:
    headers = list(df.columns)

    def format_cell(value) -> str:
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, float):
            if np.isnan(value):
                return "nan"
            return f"{value:.{float_precision}f}"
        if isinstance(value, (np.integer,)):
            return str(int(value))
        return str(value)

    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, series in df.iterrows():
        rows.append("| " + " | ".join(format_cell(series[column]) for column in headers) + " |")
    return "\n".join(rows)


def _prepare_common_panel(common_panel: pd.DataFrame, config: PipelineConfig) -> tuple[dict[str, pd.DataFrame], list[pd.Timestamp]]:
    asset_frames: dict[str, pd.DataFrame] = {}
    for asset, asset_frame in common_panel.groupby("asset"):
        ordered = asset_frame.sort_values("date").reset_index(drop=True).copy()
        if config.include_moving_average:
            ordered["ma"] = (
                ordered["close"]
                .rolling(window=config.resolved_ma_window, min_periods=config.resolved_ma_window)
                .mean()
            )
        asset_frames[asset] = ordered

    dates = (
        common_panel["date"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return asset_frames, dates


def build_samples(common_panel: pd.DataFrame, config: PipelineConfig) -> tuple[SampleBundle, dict]:
    asset_frames, _ = _prepare_common_panel(common_panel, config)
    selected_assets = sorted(asset_frames.keys())

    cumulative_sequences = []
    image_sequences = []
    chart_images = []
    metadata_rows = []
    previews_saved = []
    sequence_channel_names = channel_names(config.include_moving_average, config.include_volume)

    for asset in selected_assets:
        asset_frame = asset_frames[asset]
        last_signal_index = len(asset_frame) - config.horizon - 1
        for idx in range(config.lookback - 1, last_signal_index + 1):
            window = asset_frame.iloc[idx - config.lookback + 1 : idx + 1].copy()
            future_slice = asset_frame.iloc[idx + 1 : idx + config.horizon + 1].copy()

            if window[REQUIRED_PRICE_FIELDS].isna().any().any():
                continue
            if config.include_volume and window["volume"].isna().any():
                continue
            if config.include_moving_average and window["ma"].isna().any():
                continue

            auxiliary_targets = compute_auxiliary_targets(
                current_close=float(window["close"].iloc[-1]),
                future_close_path=future_slice["close"].to_numpy(dtype=float),
            )
            target = resolve_target(config.label_mode, config.target_name, auxiliary_targets)

            cumulative_sequences.append(
                cumulative_return_scale_window(
                    window,
                    include_moving_average=config.include_moving_average,
                    include_volume=config.include_volume,
                )
            )
            image_sequence, _ = image_scale_window(
                window,
                include_moving_average=config.include_moving_average,
                include_volume=config.include_volume,
            )
            image_sequences.append(image_sequence)
            image_array = render_jiang_chart(
                window,
                image_height=config.image_height,
                include_moving_average=config.include_moving_average,
                include_volume=config.include_volume,
            )
            chart_images.append(image_array)

            signal_date = pd.Timestamp(window["date"].iloc[-1])
            metadata_rows.append(
                {
                    "date": signal_date,
                    "asset": asset,
                    "target": target,
                    **auxiliary_targets,
                }
            )

            if len(previews_saved) < config.sample_preview_count:
                preview_name = f"{asset}_{signal_date.strftime('%Y%m%d')}.png"
                preview_path = Path(config.output_dir) / "sample_images" / preview_name
                save_chart_preview(image_array, str(preview_path))
                previews_saved.append(str(preview_path))

    metadata = pd.DataFrame(metadata_rows)
    order = metadata.sort_values(["date", "asset"]).index.to_numpy()
    metadata = metadata.loc[order].reset_index(drop=True)
    bundle = SampleBundle(
        metadata=metadata,
        cumulative_sequences=np.stack(cumulative_sequences).astype(np.float32)[order],
        image_sequences=np.stack(image_sequences).astype(np.float32)[order],
        chart_images=np.stack(chart_images).astype(np.uint8)[order],
        sequence_channels=sequence_channel_names,
    )

    target_metadata = get_target_metadata(config.label_mode, config.target_name)
    positive_rate = float(metadata["target"].mean()) if config.label_mode == "classification" else float("nan")
    preprocessing_summary = {
        "lookback": config.lookback,
        "horizon": config.horizon,
        "label_mode": config.label_mode,
        "target_name": config.target_name,
        "canonical_target_name": target_metadata["canonical_target_name"],
        "signal_role": target_metadata["signal_role"],
        "target_description": target_metadata["description"],
        "include_moving_average": config.include_moving_average,
        "include_volume": config.include_volume,
        "ma_window": config.resolved_ma_window if config.include_moving_average else None,
        "n_samples": int(len(metadata)),
        "sequence_shape": list(bundle.image_sequences.shape[1:]),
        "chart_image_shape": list(bundle.chart_images.shape[1:]),
        "positive_rate": positive_rate,
        "target_mean": float(metadata["target"].mean()),
        "target_std": float(metadata["target"].std(ddof=0)),
        "target_min": float(metadata["target"].min()),
        "target_max": float(metadata["target"].max()),
        "sample_previews": previews_saved,
    }
    return bundle, preprocessing_summary


def assign_splits(metadata: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, dict]:
    unique_dates = sorted(metadata["date"].drop_duplicates().tolist())
    train_cut = int(len(unique_dates) * config.train_frac)
    val_cut = int(len(unique_dates) * (config.train_frac + config.val_frac))
    train_dates = set(unique_dates[:train_cut])
    val_dates = set(unique_dates[train_cut:val_cut])

    split_df = metadata.copy()
    split_df["split"] = "test"
    split_df.loc[split_df["date"].isin(train_dates), "split"] = "train"
    split_df.loc[split_df["date"].isin(val_dates), "split"] = "validation"

    split_summary = {}
    for split_name in ["train", "validation", "test"]:
        frame = split_df[split_df["split"] == split_name]
        split_summary[split_name] = {
            "start": frame["date"].min().isoformat(),
            "end": frame["date"].max().isoformat(),
            "n_dates": int(frame["date"].nunique()),
            "n_samples": int(len(frame)),
        }
    return split_df, split_summary


def _prediction_frame(
    base_metadata: pd.DataFrame,
    model_name: str,
    scores: np.ndarray,
    confidence: np.ndarray | None,
    selection_sign: float,
) -> pd.DataFrame:
    frame = base_metadata.copy()
    frame["model_name"] = model_name
    frame["signal_value"] = scores.astype(float)
    frame["confidence"] = confidence.astype(float) if confidence is not None else np.nan
    frame["selection_score"] = frame["signal_value"] * float(selection_sign)
    return frame


def _fit_predict_cnn_repeated(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    repeats = max(1, int(config.cnn_repeats))
    all_scores = []

    for repeat_idx in range(repeats):
        set_global_seed(config.seed + repeat_idx)
        model = fit_torch_model(
            build_cnn_model(model_name, tuple(train_x.shape[1:])),
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            label_mode=config.label_mode,
            epochs=config.cnn_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            patience=config.patience,
            device=config.device,
        )
        scores, _ = predict_torch_model(model, test_x, config.label_mode, batch_size=config.batch_size)
        all_scores.append(scores.astype(float))

    stacked_scores = np.vstack(all_scores)
    averaged_scores = stacked_scores.mean(axis=0)

    if config.label_mode == "classification":
        confidence = np.abs(averaged_scores - 0.5) * 2.0
        return averaged_scores, confidence
    return averaged_scores, None


def _train_and_predict_models(bundle: SampleBundle, config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata, _ = assign_splits(bundle.metadata, config)
    bundle.metadata = metadata
    target_metadata = get_target_metadata(config.label_mode, config.target_name)
    selection_sign = float(target_metadata["selection_sign"])

    train_mask = metadata["split"] == "train"
    val_mask = metadata["split"] == "validation"
    test_mask = metadata["split"] == "test"

    target = metadata["target"].to_numpy(dtype=np.float32)
    feature_sets = {
        "logistic_cumulative_scale": bundle.cumulative_sequences.reshape(len(metadata), -1),
        "logistic_image_scale": bundle.image_sequences.reshape(len(metadata), -1),
        "cnn_1d_cumulative_scale": np.transpose(bundle.cumulative_sequences, (0, 2, 1)),
        "cnn_1d_image_scale": np.transpose(bundle.image_sequences, (0, 2, 1)),
        "cnn_1d_multiscale_image_scale": np.transpose(bundle.image_sequences, (0, 2, 1)),
        "cnn_1d_dilated_cumulative_scale": np.transpose(bundle.cumulative_sequences, (0, 2, 1)),
        "cnn_1d_dilated_image_scale": np.transpose(bundle.image_sequences, (0, 2, 1)),
        "cnn_1d_attention_image_scale": np.transpose(bundle.image_sequences, (0, 2, 1)),
        "cnn_1d_attention_cumulative_scale": np.transpose(bundle.cumulative_sequences, (0, 2, 1)),
        "cnn_2d_rendered_images": (bundle.chart_images[:, None, :, :].astype(np.float32) / 255.0),
        "cnn_2d_residual_images": (bundle.chart_images[:, None, :, :].astype(np.float32) / 255.0),
    }
    if config.enabled_models:
        feature_sets = {name: value for name, value in feature_sets.items() if name in set(config.enabled_models)}
        if not feature_sets:
            raise ValueError("enabled_models filtered out every available model")

    predictions = []
    comparison_rows = []

    for model_name, features in feature_sets.items():
        train_x = features[train_mask]
        val_x = features[val_mask]
        test_x = features[test_mask]
        train_y = target[train_mask]
        val_y = target[val_mask]

        if model_name.startswith("logistic"):
            model = fit_linear_model(train_x, train_y, config.label_mode, config.logistic_max_iter)
            scores, confidence = predict_linear_model(model, test_x, config.label_mode)
        elif model_name.startswith("cnn_1d"):
            scores, confidence = _fit_predict_cnn_repeated(
                model_name=model_name,
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                test_x=test_x,
                config=config,
            )
        elif model_name.startswith("cnn_2d"):
            scores, confidence = _fit_predict_cnn_repeated(
                model_name=model_name,
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                test_x=test_x,
                config=config,
            )
        else:
            raise ValueError(f"unsupported model name: {model_name}")

        prediction_frame = _prediction_frame(metadata.loc[test_mask], model_name, scores, confidence, selection_sign)
        predictions.append(prediction_frame)

        metrics = compute_prediction_metrics(prediction_frame, config.label_mode)
        if config.label_mode == "classification":
            metrics["rank_correlation"] = mean_rank_correlation(
                prediction_frame,
                target_column="future_return",
                score_column="selection_score",
            )
        else:
            metrics["target_rank_correlation"] = mean_rank_correlation(
                prediction_frame,
                target_column="target",
                score_column="signal_value",
            )
            metrics["future_return_rank_correlation"] = mean_rank_correlation(
                prediction_frame,
                target_column="future_return",
                score_column="selection_score",
            )
        metrics.update(top_k_backtest(prediction_frame, config.horizon, config.top_k, score_column="selection_score"))
        metrics["model_name"] = model_name
        comparison_rows.append(metrics)

    predictions_df = pd.concat(predictions, ignore_index=True)
    comparison_df = pd.DataFrame(comparison_rows)
    sort_preferences = [
        ("top_k_sharpe", False),
        ("future_return_rank_correlation", False),
        ("rank_correlation", False),
        ("target_rank_correlation", False),
        ("roc_auc", False),
        ("rmse", True),
    ]
    sort_columns = [column for column, _ in sort_preferences if column in comparison_df.columns]
    ascending = [ascending_value for column, ascending_value in sort_preferences if column in comparison_df.columns]
    comparison_df = comparison_df.sort_values(sort_columns, ascending=ascending, na_position="last").reset_index(drop=True)
    return predictions_df, comparison_df


def _build_report(
    discovered_files: dict[str, str | None],
    dataset_summary: dict,
    preprocessing_summary: dict,
    split_summary: dict,
    comparison_df: pd.DataFrame,
    best_model_name: str,
) -> str:
    discovery_df = pd.DataFrame(
        [
            {"role": "ETF CSV dataset", "path": discovered_files["etf_csv"]},
            {"role": "Jiang price-image paper", "path": discovered_files["jiang_price_image_paper"]},
            {"role": "Image-based asset pricing / DV paper", "path": discovered_files["dv_image_asset_pricing_paper"]},
            {"role": "ODE portfolio optimization paper", "path": discovered_files["ode_portfolio_paper"]},
        ]
    )

    split_df = pd.DataFrame(
        [
            {"split": split_name, **values}
            for split_name, values in split_summary.items()
        ]
    )

    dataset_overview_df = pd.DataFrame(
        [
            {
                "common_start": dataset_summary["common_sample_start"],
                "common_end": dataset_summary["common_sample_end"],
                "assets": len(dataset_summary["common_sample_assets"]),
                "dates": dataset_summary["common_sample_dates"],
                "rows": dataset_summary["common_sample_rows"],
                "median_step_days": dataset_summary["median_step_days"],
            }
        ]
    )

    preprocessing_df = pd.DataFrame(
        [
            {
                "lookback": preprocessing_summary["lookback"],
                "horizon": preprocessing_summary["horizon"],
                "label_mode": preprocessing_summary["label_mode"],
                "target_name": preprocessing_summary["target_name"],
                "signal_role": preprocessing_summary["signal_role"],
                "include_ma": preprocessing_summary["include_moving_average"],
                "include_volume": preprocessing_summary["include_volume"],
                "samples": preprocessing_summary["n_samples"],
                "sequence_shape": "x".join(map(str, preprocessing_summary["sequence_shape"])),
                "chart_shape": "x".join(map(str, preprocessing_summary["chart_image_shape"])),
                "positive_rate": preprocessing_summary["positive_rate"],
                "target_mean": preprocessing_summary["target_mean"],
                "target_std": preprocessing_summary["target_std"],
                "target_min": preprocessing_summary["target_min"],
                "target_max": preprocessing_summary["target_max"],
            }
        ]
    )

    lines = [
        "# ETF Image-Based Signal Pipeline",
        "",
        "## 1. File Discovery Summary",
        _markdown_table(discovery_df, float_precision=4),
        "",
        "## 2. Dataset Summary",
        _markdown_table(dataset_overview_df, float_precision=4),
        "",
        _markdown_table(pd.DataFrame(dataset_summary["asset_rows"]), float_precision=4),
        "",
        "## 3. Preprocessing and Image-Generation Summary",
        _markdown_table(preprocessing_df, float_precision=4),
        "",
        "Sample chart previews:",
    ]
    for preview_path in preprocessing_summary["sample_previews"]:
        lines.append(f"- {preview_path}")

    lines.extend(
        [
            "",
            "## 4. Train/Validation/Test Split Summary",
            _markdown_table(split_df, float_precision=4),
            "",
            "## 5. Model Comparison Table",
            _markdown_table(comparison_df, float_precision=4),
            "",
            "## 6. How This Connects to ODE Later",
            f"- Use `{best_model_name}` `signal_value` as an expected-return-style score for `mu(t)` after a later calibration step from probability to return scale.",
            "- Reuse the same feature pipeline with regression targets such as `future_downside` or `future_skew` to supply adaptive downside or risk-control signals that can modulate time-varying ODE risk aversion.",
            "- The exported panel is date-indexed and asset-indexed, so it can be aligned directly with rolling estimates of covariance for an RK4 or Euler ODE portfolio allocator.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_pipeline(config: PipelineConfig) -> dict:
    set_global_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sample_images").mkdir(parents=True, exist_ok=True)

    discovered_files = discover_local_files(".")
    if not discovered_files["etf_csv"]:
        raise FileNotFoundError("could not detect ETF CSV file in current working folder")

    _, long_panel, assets, fields, coverage_df = load_etf_csv(config.data_path)
    selected_assets = config.selected_assets or assets

    common_required_fields = REQUIRED_PRICE_FIELDS.copy()
    if config.include_volume:
        if "volume" in fields:
            common_required_fields.extend(OPTIONAL_FIELDS)
        else:
            config.include_volume = False

    common_panel, common_summary = restrict_common_valid_sample(
        long_panel,
        selected_assets=selected_assets,
        required_fields=common_required_fields,
    )
    if common_panel.empty:
        raise ValueError("no common valid sample remains after asset/field filtering")

    dataset_summary = build_dataset_summary(
        csv_path=config.data_path,
        fields=fields,
        assets=assets,
        common_panel=common_panel,
        coverage_df=coverage_df,
    )
    dataset_summary["common_filter"] = common_summary

    bundle, preprocessing_summary = build_samples(common_panel, config)
    predictions_df, comparison_df = _train_and_predict_models(bundle, config)
    split_summary = {
        "train": bundle.metadata[bundle.metadata["split"] == "train"],
        "validation": bundle.metadata[bundle.metadata["split"] == "validation"],
        "test": bundle.metadata[bundle.metadata["split"] == "test"],
    }
    split_summary = {
        split_name: {
            "start": frame["date"].min().isoformat(),
            "end": frame["date"].max().isoformat(),
            "n_dates": int(frame["date"].nunique()),
            "n_samples": int(len(frame)),
        }
        for split_name, frame in split_summary.items()
    }

    best_model_name = str(comparison_df.iloc[0]["model_name"])

    detailed_predictions_path = output_dir / "model_predictions_detailed.csv"
    predictions_df.to_csv(detailed_predictions_path, index=False)

    required_signals = predictions_df[["date", "asset", "model_name", "signal_value", "confidence"]].copy()
    required_signals["date"] = required_signals["date"].dt.strftime("%Y-%m-%d")
    signal_path = output_dir / "etf_signals.csv"
    required_signals.to_csv(signal_path, index=False)
    best_signal_path = output_dir / "best_ode_signal.csv"
    required_signals[required_signals["model_name"] == best_model_name].to_csv(best_signal_path, index=False)

    comparison_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    report_text = _build_report(
        discovered_files=discovered_files,
        dataset_summary=dataset_summary,
        preprocessing_summary=preprocessing_summary,
        split_summary=split_summary,
        comparison_df=comparison_df,
        best_model_name=best_model_name,
    )
    report_path = output_dir / "pipeline_report.md"
    report_path.write_text(report_text, encoding="utf-8")

    payload = {
        "label_mode": config.label_mode,
        "target_name": config.target_name,
        "discovered_files": discovered_files,
        "dataset_summary": dataset_summary,
        "preprocessing_summary": preprocessing_summary,
        "split_summary": split_summary,
        "best_model_name": best_model_name,
        "comparison": _to_records(comparison_df),
        "paths": {
            "report": str(report_path),
            "signal_file": str(signal_path),
            "best_signal_file": str(best_signal_path),
            "detailed_predictions": str(detailed_predictions_path),
            "comparison_file": str(comparison_path),
        },
    }

    summary_path = output_dir / "report_data.json"
    summary_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    return payload
