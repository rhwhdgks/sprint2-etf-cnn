from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_FIELDS = ["open", "high", "low", "close"]
EPSILON = 1e-8


def channel_names(include_moving_average: bool, include_volume: bool) -> list[str]:
    names = PRICE_FIELDS.copy()
    if include_moving_average:
        names.append("ma")
    if include_volume:
        names.append("volume")
    return names


def _safe_divisor(value: float) -> float:
    if not np.isfinite(value) or abs(value) < EPSILON:
        return 1.0
    return float(value)


def cumulative_return_scale_window(
    window: pd.DataFrame,
    include_moving_average: bool,
    include_volume: bool,
) -> np.ndarray:
    base_close = _safe_divisor(float(window["close"].iloc[0]))
    features = [(window[field].to_numpy(dtype=float) / base_close) for field in PRICE_FIELDS]

    if include_moving_average:
        features.append(window["ma"].to_numpy(dtype=float) / base_close)

    if include_volume:
        base_volume = float(window["volume"].iloc[0])
        if not np.isfinite(base_volume) or abs(base_volume) < EPSILON:
            base_volume = float(window["volume"].replace([np.inf, -np.inf], np.nan).max())
        features.append(window["volume"].to_numpy(dtype=float) / _safe_divisor(base_volume))

    return np.column_stack(features).astype(np.float32)


def image_scale_window(
    window: pd.DataFrame,
    include_moving_average: bool,
    include_volume: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    price_arrays = [window[field].to_numpy(dtype=float) for field in PRICE_FIELDS]
    if include_moving_average:
        price_arrays.append(window["ma"].to_numpy(dtype=float))

    price_min = float(np.nanmin(np.concatenate(price_arrays)))
    price_max = float(np.nanmax(np.concatenate(price_arrays)))
    price_range = price_max - price_min
    if not np.isfinite(price_range) or price_range < EPSILON:
        price_range = 1.0

    features = []
    for field in PRICE_FIELDS:
        scaled = (window[field].to_numpy(dtype=float) - price_min) / price_range
        features.append(np.clip(scaled, 0.0, 1.0))

    if include_moving_average:
        ma_scaled = (window["ma"].to_numpy(dtype=float) - price_min) / price_range
        features.append(np.clip(ma_scaled, 0.0, 1.0))

    volume_max = 1.0
    if include_volume:
        volume = window["volume"].to_numpy(dtype=float)
        volume_max = float(np.nanmax(volume))
        if not np.isfinite(volume_max) or volume_max < EPSILON:
            volume_max = 1.0
        features.append(np.clip(volume / volume_max, 0.0, 1.0))

    return (
        np.column_stack(features).astype(np.float32),
        {"price_min": price_min, "price_max": price_max, "volume_max": volume_max},
    )
