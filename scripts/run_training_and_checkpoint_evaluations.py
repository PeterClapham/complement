"""Run the full grid, then evaluate configured saved checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation import evaluate_checkpoint_sweep
from training import run_experiment_grid
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training and checkpoint evaluations.")
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_result = run_experiment_grid(load_config(args.training_config))
    print(f"Completed runs: {training_result.completed}/{len(training_result.runs)}", flush=True)
    evaluate_checkpoint_sweep(load_config(args.evaluation_config))


if __name__ == "__main__":
    main()
