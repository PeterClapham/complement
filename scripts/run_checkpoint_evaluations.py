"""Evaluate saved milestone checkpoints across a completed model grid."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation import evaluate_checkpoint_sweep
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved epoch checkpoints.")
    parser.add_argument("--config", type=Path, default=Path("configs/test_checkpoints_80.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate_checkpoint_sweep(load_config(args.config))
    for result in results:
        print(f"Wrote metrics: {result.metrics_path}")
        for path in result.heatmap_paths:
            print(f"Wrote heatmap: {path}")


if __name__ == "__main__":
    main()
