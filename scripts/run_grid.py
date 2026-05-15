"""Run all experiment configurations in a YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

from training import run_experiment_grid
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full dataset/seed/beta experiment grid.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to a YAML experiment config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment_grid(load_config(args.config))
    print(f"Completed runs: {result.completed}/{len(result.runs)}")


if __name__ == "__main__":
    main()
