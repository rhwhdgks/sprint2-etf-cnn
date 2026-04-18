from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_prediction_metrics(predictions: pd.DataFrame, label_mode: str) -> dict[str, float]:
    if label_mode == "classification":
        y_true = predictions["target"].to_numpy()
        scores = predictions["signal_value"].to_numpy()
        labels = (scores >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_true, labels)),
            "precision": float(precision_score(y_true, labels, zero_division=0)),
            "recall": float(recall_score(y_true, labels, zero_division=0)),
            "f1": float(f1_score(y_true, labels, zero_division=0)),
        }
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        else:
            metrics["roc_auc"] = float("nan")
        return metrics

    y_true = predictions["target"].to_numpy(dtype=float)
    y_pred = predictions["signal_value"].to_numpy(dtype=float)
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def mean_rank_correlation(
    predictions: pd.DataFrame,
    target_column: str = "future_return",
    score_column: str = "signal_value",
) -> float:
    correlations = []
    for _, date_frame in predictions.groupby("date"):
        if date_frame[score_column].nunique() < 2 or date_frame[target_column].nunique() < 2:
            continue
        rho, _ = spearmanr(date_frame[score_column], date_frame[target_column])
        if np.isfinite(rho):
            correlations.append(float(rho))
    return float(np.mean(correlations)) if correlations else float("nan")


def top_k_backtest(
    predictions: pd.DataFrame,
    horizon: int,
    top_k: int,
    score_column: str = "selection_score",
) -> dict[str, float]:
    rets = []
    spreads = []
    turnovers = []
    previous_assets: set[str] | None = None
    rebal_dates = sorted(predictions["date"].drop_duplicates().tolist())[::horizon]

    for date in rebal_dates:
        date_frame = predictions[predictions["date"] == date].sort_values(score_column, ascending=False)
        if len(date_frame) < top_k:
            continue
        top_frame = date_frame.head(top_k)
        bottom_frame = date_frame.tail(top_k)
        selected_assets = set(top_frame["asset"].tolist())
        if previous_assets is not None:
            turnovers.append(1.0 - len(selected_assets & previous_assets) / float(top_k))
        previous_assets = selected_assets

        rets.append(float(top_frame["future_return"].mean()))
        spreads.append(float(top_frame["future_return"].mean() - bottom_frame["future_return"].mean()))

    if not rets:
        return {
            "top_k_cumulative_return": float("nan"),
            "top_k_sharpe": float("nan"),
            "top_k_hit_rate": float("nan"),
            "top_bottom_spread_mean": float("nan"),
            "turnover": float("nan"),
            "n_rebalances": 0.0,
        }

    returns = np.array(rets, dtype=float)
    annual_scale = math.sqrt(252.0 / horizon)
    sharpe = float(returns.mean() / returns.std(ddof=1) * annual_scale) if len(returns) > 1 and returns.std(ddof=1) > 0 else float("nan")
    cumulative_return = float(np.prod(1.0 + returns) - 1.0)
    hit_rate = float(np.mean(returns > 0.0))
    return {
        "top_k_cumulative_return": cumulative_return,
        "top_k_sharpe": sharpe,
        "top_k_hit_rate": hit_rate,
        "top_bottom_spread_mean": float(np.mean(spreads)),
        "turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "n_rebalances": float(len(returns)),
    }
