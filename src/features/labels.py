from __future__ import annotations

import numpy as np
from scipy.stats import skew


def get_target_metadata(label_mode: str, target_name: str) -> dict[str, str | float]:
    if label_mode == "classification":
        if target_name != "positive_return":
            raise ValueError(f"classification only supports target_name='positive_return', got {target_name}")
        return {
            "canonical_target_name": "positive_return",
            "signal_role": "expected_return",
            "selection_sign": 1.0,
            "description": "Probability that future return is positive.",
        }

    if label_mode != "regression":
        raise ValueError(f"unsupported label_mode: {label_mode}")

    if target_name in {"future_return", "momentum_like"}:
        return {
            "canonical_target_name": "future_return",
            "signal_role": "expected_return",
            "selection_sign": 1.0,
            "description": "Predicted future return over the forecast horizon.",
        }
    if target_name in {"downside_like", "future_downside"}:
        return {
            "canonical_target_name": "future_downside",
            "signal_role": "downside_risk",
            "selection_sign": -1.0,
            "description": "Predicted downside risk over the forecast horizon; lower is better for selection.",
        }
    if target_name in {"skew_like", "future_skew"}:
        return {
            "canonical_target_name": "future_skew",
            "signal_role": "tail_shape",
            "selection_sign": 1.0,
            "description": "Predicted skew-like tail asymmetry over the forecast horizon; higher is better for selection.",
        }
    raise ValueError(f"unsupported regression target_name: {target_name}")


def compute_auxiliary_targets(current_close: float, future_close_path: np.ndarray) -> dict[str, float]:
    close_path = np.concatenate([[current_close], future_close_path.astype(float)])
    future_daily_returns = np.diff(close_path) / close_path[:-1]
    future_return = float(future_close_path[-1] / current_close - 1.0)
    downside = float(np.mean(np.square(np.minimum(future_daily_returns, 0.0))))
    skew_like = float(skew(future_daily_returns, bias=False)) if len(future_daily_returns) > 2 else 0.0
    return {
        "future_return": future_return,
        "future_downside": downside,
        "future_skew": skew_like,
    }


def resolve_target(label_mode: str, target_name: str, auxiliary_targets: dict[str, float]) -> float:
    metadata = get_target_metadata(label_mode, target_name)
    canonical_target_name = str(metadata["canonical_target_name"])

    if label_mode == "classification":
        return float(auxiliary_targets["future_return"] > 0.0)

    if canonical_target_name == "future_return":
        return auxiliary_targets["future_return"]
    if canonical_target_name == "future_downside":
        return auxiliary_targets["future_downside"]
    if canonical_target_name == "future_skew":
        return auxiliary_targets["future_skew"]
    raise ValueError(f"unsupported regression target_name: {target_name}")
