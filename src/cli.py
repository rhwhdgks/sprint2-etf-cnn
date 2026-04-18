from __future__ import annotations

import argparse

from src.config import PipelineConfig


def parse_window_pair(value: str) -> tuple[int, int]:
    parts = value.split("/")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("window pairs must look like 20/20")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("window pair values must be integers") from exc


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    output_dir: str,
    label_mode: str,
    target_name: str,
    cnn_epochs: int,
    patience: int,
) -> argparse.ArgumentParser:
    parser.add_argument("--data-path", default="etfdata.csv")
    parser.add_argument("--output-dir", default=output_dir)
    parser.add_argument("--label-mode", choices=["classification", "regression"], default=label_mode)
    parser.add_argument("--target-name", default=target_name)
    parser.add_argument("--disable-moving-average", action="store_true")
    parser.add_argument("--disable-volume", action="store_true")
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cnn-epochs", type=int, default=cnn_epochs)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=patience)
    parser.add_argument("--assets", nargs="*", default=[])
    return parser


def config_from_args(args: argparse.Namespace, **overrides) -> PipelineConfig:
    values = {
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "lookback": getattr(args, "lookback", 20),
        "horizon": getattr(args, "horizon", 20),
        "label_mode": args.label_mode,
        "target_name": args.target_name,
        "selected_assets": list(getattr(args, "assets", [])),
        "enabled_models": list(getattr(args, "models", [])),
        "include_moving_average": not getattr(args, "disable_moving_average", False),
        "include_volume": not getattr(args, "disable_volume", False),
        "image_height": args.image_height,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "top_k": args.top_k,
        "seed": args.seed,
        "cnn_epochs": args.cnn_epochs,
        "cnn_repeats": getattr(args, "cnn_repeats", 1),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
    }
    values.update(overrides)
    return PipelineConfig(**values)
