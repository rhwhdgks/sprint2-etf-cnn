from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class PipelineConfig:
    data_path: str = "etfdata.csv"
    output_dir: str = "results/base"
    lookback: int = 20
    horizon: int = 20
    label_mode: str = "classification"
    target_name: str = "positive_return"
    selected_assets: list[str] = field(default_factory=list)
    enabled_models: list[str] = field(default_factory=list)
    include_moving_average: bool = True
    include_volume: bool = True
    ma_window: int | None = None
    image_height: int = 64
    train_frac: float = 0.70
    val_frac: float = 0.15
    top_k: int = 2
    seed: int = 42
    logistic_max_iter: int = 2000
    cnn_epochs: int = 8
    cnn_repeats: int = 1
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 2
    device: str = "cpu"
    sample_preview_count: int = 6
    wf_min_train_days: int = 500
    wf_val_days: int = 60
    wf_test_days: int = 60

    @property
    def resolved_ma_window(self) -> int:
        return self.ma_window or self.lookback

    @property
    def image_width(self) -> int:
        return self.lookback * 3

    def to_dict(self) -> dict:
        return asdict(self)
