"""Run one experiment-grid coordinate, typically from a Slurm array task."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from training import coordinate_for_index, experiment_coordinates, run_coordinate
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one configured experiment-grid coordinate.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/slurm.yaml"),
        help="Path to a YAML experiment config.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Zero-based task id. Defaults to SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--print-count",
        action="store_true",
        help="Print the number of configured coordinates and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    coordinates = experiment_coordinates(config)

    if args.print_count:
        print(len(coordinates))
        return

    task_id = args.task_id if args.task_id is not None else _slurm_task_id()
    coordinate = coordinate_for_index(config, task_id)
    print(
        f"Task {task_id}/{len(coordinates) - 1}: "
        f"dataset={coordinate.dataset_name} seed={coordinate.seed} "
        f"beta_inf={coordinate.beta_inf} beta_opt={coordinate.beta_opt}",
        flush=True,
    )
    result = run_coordinate(config, coordinate)
    print(
        f"Task {task_id} complete: run_dir={result.run_dir} "
        f"steps={result.num_steps} resumed={result.resumed}",
        flush=True,
    )


def _slurm_task_id() -> int:
    value = os.environ.get("SLURM_ARRAY_TASK_ID")
    if value is None:
        raise ValueError("Provide --task-id or run inside a Slurm array task")
    return int(value)


if __name__ == "__main__":
    main()
