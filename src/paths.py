from __future__ import annotations

from pathlib import Path

RESULTS_ROOT = Path("results")

DEFAULT_OUTPUT_DIRS = {
    "pipeline": RESULTS_ROOT / "base",
    "regression_suite": RESULTS_ROOT / "regression" / "suite",
    "window_sweep": RESULTS_ROOT / "windows" / "sweep",
    "cnn_focus": RESULTS_ROOT / "cnn" / "focus",
    "cnn_ablation": RESULTS_ROOT / "cnn" / "ablation",
}


def default_output_dir(name: str) -> str:
    return str(DEFAULT_OUTPUT_DIRS[name])
