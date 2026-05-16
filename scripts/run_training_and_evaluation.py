"""Run a full training grid, then evaluate the final saved models."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation import evaluate_model_grid
from training import run_experiment_grid
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training and final-model evaluation.")
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_result = run_experiment_grid(load_config(args.training_config))
    print(f"Completed runs: {training_result.completed}/{len(training_result.runs)}", flush=True)
    evaluation_result = evaluate_model_grid(load_config(args.evaluation_config))
    print(f"Wrote metrics: {evaluation_result.metrics_path}", flush=True)
    for path in evaluation_result.heatmap_paths:
        print(f"Wrote heatmap: {path}", flush=True)


if __name__ == "__main__":
    main()
