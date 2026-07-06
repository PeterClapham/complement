"""Run beta-zero boundary extensions, then evaluate both completed grids."""

from __future__ import annotations

from evaluation import evaluate_model_grid
from training import run_experiment_grid
from utils import load_config


def main() -> None:
    jobs = [
        ("configs/mnist_seed0_150ep_beta0.yaml", "configs/test_grid_mnist_150_beta0.yaml"),
        ("configs/smallnorb_seed0_200ep_beta0.yaml", "configs/test_grid_smallnorb_200_beta0.yaml"),
    ]
    for training_config, evaluation_config in jobs:
        training_result = run_experiment_grid(load_config(training_config))
        print(f"Completed runs: {training_result.completed}/{len(training_result.runs)}", flush=True)
        evaluation_result = evaluate_model_grid(load_config(evaluation_config))
        print(f"Wrote metrics: {evaluation_result.metrics_path}", flush=True)
        for path in evaluation_result.heatmap_paths:
            print(f"Wrote heatmap: {path}", flush=True)


if __name__ == "__main__":
    main()
