from __future__ import annotations

import os
from pathlib import Path


def _iter_local_files(root: str) -> list[Path]:
    files: list[Path] = []
    skip_dirs = {".venv", "__pycache__", "outputs", "src"}
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in skip_dirs and not dirname.startswith(".")
        ]
        for filename in filenames:
            files.append(Path(current_root) / filename)
    return sorted(files)


def _match_file(files: list[Path], extension: str, keywords: tuple[str, ...]) -> str | None:
    for path in files:
        name = path.name.lower()
        if path.suffix.lower() == extension and all(keyword in name for keyword in keywords):
            return str(path)
    return None


def discover_local_files(root: str) -> dict[str, str | None]:
    files = _iter_local_files(root)
    csv_candidates = [path for path in files if path.suffix.lower() == ".csv"]
    etf_csv = None
    for path in csv_candidates:
        name = path.name.lower()
        if "etf" in name:
            etf_csv = str(path)
            break
    if etf_csv is None and csv_candidates:
        etf_csv = str(csv_candidates[0])

    jiang_paper = _match_file(files, ".pdf", ("jiang",))
    if jiang_paper is None:
        jiang_paper = _match_file(files, ".pdf", ("price", "trends"))

    dv_paper = _match_file(files, ".pdf", ("image-based", "asset", "pricing"))
    ode_paper = _match_file(files, ".pdf", ("ode", "portfolio"))

    return {
        "etf_csv": etf_csv,
        "jiang_price_image_paper": jiang_paper,
        "dv_image_asset_pricing_paper": dv_paper,
        "ode_portfolio_paper": ode_paper,
    }
