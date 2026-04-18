from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.eval.metrics import compute_prediction_metrics, mean_rank_correlation, top_k_backtest
from src.features.labels import get_target_metadata
from src.models.baselines import fit_linear_model, predict_linear_model
from src.pipeline import SampleBundle, _fit_predict_cnn_repeated, _prediction_frame


def generate_walkforward_folds(dates: list, config: PipelineConfig) -> list[dict]:
    """Expanding-window walk-forward folds.

    Each fold: train on all history up to train_end, validate on next wf_val_days,
    test on following wf_test_days. Train window expands by wf_test_days each step.
    """
    n = len(dates)
    min_train = config.wf_min_train_days
    val_days = config.wf_val_days
    test_days = config.wf_test_days

    folds = []
    train_end = min_train
    while train_end + val_days + test_days <= n:
        val_end = train_end + val_days
        test_end = val_end + test_days
        folds.append(
            {
                "train_dates": set(dates[:train_end]),
                "val_dates": set(dates[train_end:val_end]),
                "test_dates": set(dates[val_end:test_end]),
                "train_end_date": dates[train_end - 1],
                "test_end_date": dates[test_end - 1],
            }
        )
        train_end += test_days
    return folds


def _build_feature_sets(bundle: SampleBundle, n: int, enabled_models: list[str] | None) -> dict[str, np.ndarray]:
    all_sets = {
        "logistic_cumulative_scale": bundle.cumulative_sequences.reshape(n, -1),
        "logistic_image_scale": bundle.image_sequences.reshape(n, -1),
        "cnn_1d_cumulative_scale": np.transpose(bundle.cumulative_sequences, (0, 2, 1)),
        "cnn_1d_image_scale": np.transpose(bundle.image_sequences, (0, 2, 1)),
        "cnn_1d_multiscale_image_scale": np.transpose(bundle.image_sequences, (0, 2, 1)),
        "cnn_1d_dilated_image_scale": np.transpose(bundle.image_sequences, (0, 2, 1)),
        "cnn_1d_attention_image_scale": np.transpose(bundle.image_sequences, (0, 2, 1)),
        "cnn_2d_rendered_images": (bundle.chart_images[:, None, :, :].astype(np.float32) / 255.0),
        "cnn_2d_residual_images": (bundle.chart_images[:, None, :, :].astype(np.float32) / 255.0),
    }
    if enabled_models:
        return {k: v for k, v in all_sets.items() if k in set(enabled_models)}
    return all_sets


def _fit_predict_fold(
    bundle: SampleBundle,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    feature_sets: dict[str, np.ndarray],
    config: PipelineConfig,
) -> pd.DataFrame:
    metadata = bundle.metadata
    target_metadata = get_target_metadata(config.label_mode, config.target_name)
    selection_sign = float(target_metadata["selection_sign"])
    target = metadata["target"].to_numpy(dtype=np.float32)

    predictions = []
    for model_name, features in feature_sets.items():
        train_x = features[train_mask]
        val_x = features[val_mask]
        test_x = features[test_mask]
        train_y = target[train_mask]
        val_y = target[val_mask]

        if model_name.startswith("logistic"):
            model = fit_linear_model(train_x, train_y, config.label_mode, config.logistic_max_iter)
            scores, confidence = predict_linear_model(model, test_x, config.label_mode)
        elif model_name.startswith("cnn"):
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

        pred_frame = _prediction_frame(metadata[test_mask], model_name, scores, confidence, selection_sign)
        predictions.append(pred_frame)

    return pd.concat(predictions, ignore_index=True)


def run_walkforward(bundle: SampleBundle, config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward OOS evaluation.

    Returns (oos_predictions_df, comparison_df).
    oos_predictions_df has one row per (date, asset, model) in the OOS windows.
    comparison_df has aggregated metrics per model across all folds.
    """
    metadata = bundle.metadata
    unique_dates = sorted(metadata["date"].drop_duplicates().tolist())
    folds = generate_walkforward_folds(unique_dates, config)

    if not folds:
        total_needed = config.wf_min_train_days + config.wf_val_days + config.wf_test_days
        raise ValueError(
            f"not enough dates for walk-forward: need at least {total_needed}, got {len(unique_dates)}"
        )

    print(f"Walk-forward: {len(folds)} folds over {len(unique_dates)} total dates")

    n = len(metadata)
    feature_sets = _build_feature_sets(bundle, n, config.enabled_models or None)

    all_predictions = []
    for fold_idx, fold in enumerate(folds):
        train_mask = metadata["date"].isin(fold["train_dates"]).to_numpy()
        val_mask = metadata["date"].isin(fold["val_dates"]).to_numpy()
        test_mask = metadata["date"].isin(fold["test_dates"]).to_numpy()

        if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        print(
            f"  fold {fold_idx + 1:2d}/{len(folds)}: "
            f"train={train_mask.sum():5d}  val={val_mask.sum():4d}  test={test_mask.sum():4d}  "
            f"test_end={fold['test_end_date'].strftime('%Y-%m-%d')}"
        )

        fold_preds = _fit_predict_fold(
            bundle=bundle,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            feature_sets=feature_sets,
            config=config,
        )
        fold_preds["fold"] = fold_idx
        all_predictions.append(fold_preds)

    oos_df = pd.concat(all_predictions, ignore_index=True)

    comparison_rows = []
    for model_name, group in oos_df.groupby("model_name"):
        group = group.copy()
        metrics = compute_prediction_metrics(group, config.label_mode)
        if config.label_mode == "classification":
            metrics["rank_correlation"] = mean_rank_correlation(group, "future_return", "selection_score")
        else:
            metrics["target_rank_correlation"] = mean_rank_correlation(group, "target", "signal_value")
            metrics["future_return_rank_correlation"] = mean_rank_correlation(group, "future_return", "selection_score")
        metrics.update(top_k_backtest(group, config.horizon, config.top_k, score_column="selection_score"))
        metrics["model_name"] = model_name
        metrics["n_folds"] = int(oos_df.loc[oos_df["model_name"] == model_name, "fold"].nunique())
        comparison_rows.append(metrics)

    comparison_df = pd.DataFrame(comparison_rows)
    sort_preferences = [
        ("top_k_sharpe", False),
        ("future_return_rank_correlation", False),
        ("rank_correlation", False),
        ("target_rank_correlation", False),
        ("roc_auc", False),
        ("rmse", True),
    ]
    sort_cols = [col for col, _ in sort_preferences if col in comparison_df.columns]
    ascending = [asc for col, asc in sort_preferences if col in comparison_df.columns]
    if sort_cols:
        comparison_df = comparison_df.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)

    return oos_df, comparison_df
