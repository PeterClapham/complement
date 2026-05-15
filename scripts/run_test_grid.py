"""Evaluate a completed model grid and save metric heatmaps."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation import evaluate_model_grid
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a completed model grid.")
    parser.add_argument("--config", type=Path, default=Path("configs/test_grid.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_model_grid(load_config(args.config))
    print(f"Wrote metrics: {result.metrics_path}")
    for path in result.heatmap_paths:
        print(f"Wrote heatmap: {path}")


if __name__ == "__main__":
    main()
