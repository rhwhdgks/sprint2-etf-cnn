from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.features.scaling import image_scale_window


def render_jiang_chart(
    window: pd.DataFrame,
    image_height: int,
    include_moving_average: bool,
    include_volume: bool,
) -> np.ndarray:
    scaled, _ = image_scale_window(
        window,
        include_moving_average=include_moving_average,
        include_volume=include_volume,
    )

    width = len(window) * 3
    height = image_height
    price_height = int(round(height * 0.8)) if include_volume else height
    volume_height = max(height - price_height, 1)
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)

    def to_price_row(value: float) -> int:
        return int(round((price_height - 1) * (1.0 - float(value))))

    def to_volume_row(value: float) -> int:
        return price_height + int(round((volume_height - 1) * (1.0 - float(value))))

    ma_index = 4 if include_moving_average else None
    volume_index = 5 if include_moving_average and include_volume else 4 if include_volume else None

    previous_ma_point: tuple[int, int] | None = None
    for day_idx in range(len(window)):
        x0 = day_idx * 3
        x1 = x0 + 1
        x2 = x0 + 2

        open_row = to_price_row(scaled[day_idx, 0])
        high_row = to_price_row(scaled[day_idx, 1])
        low_row = to_price_row(scaled[day_idx, 2])
        close_row = to_price_row(scaled[day_idx, 3])

        draw.line([(x1, high_row), (x1, low_row)], fill=255)
        draw.line([(x0, open_row), (x1, open_row)], fill=255)
        draw.line([(x1, close_row), (x2, close_row)], fill=255)

        if include_volume and volume_index is not None:
            volume_top = to_volume_row(scaled[day_idx, volume_index])
            draw.rectangle([(x0, volume_top), (x2, height - 1)], fill=255)

        if include_moving_average and ma_index is not None:
            ma_point = (x1, to_price_row(scaled[day_idx, ma_index]))
            if previous_ma_point is not None:
                draw.line([previous_ma_point, ma_point], fill=255)
            previous_ma_point = ma_point

    return np.asarray(image, dtype=np.uint8)


def save_chart_preview(image_array: np.ndarray, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array, mode="L").save(path)
