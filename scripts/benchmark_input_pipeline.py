"""Benchmark safe input-pipeline training settings."""

from __future__ import annotations

import argparse
import csv
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from training import run_gon_experiment
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark data-loader settings.")
    parser.add_argument("--config", type=Path, default=Path("configs/six_hour_seed0_batch128.yaml"))
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 6, 8])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--cache-tensors", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--dataset", type=str, default="mnist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    rows = []
    for workers in args.workers:
        rows.append(
            _run_case(
                base_config,
                args.epochs,
                workers,
                dataset_name=args.dataset,
                cache_tensors=args.cache_tensors,
                progress=args.progress,
            )
        )
    suffix = f"cache{int(args.cache_tensors)}_progress{int(args.progress)}"
    _write_rows(Path("results") / f"input_pipeline_benchmark_{suffix}.csv", rows)


def _run_case(
    base_config: dict[str, Any],
    epochs: int,
    workers: int,
    dataset_name: str,
    cache_tensors: bool,
    progress: bool,
) -> dict[str, float | int | bool]:
    config = deepcopy(base_config)
    config["experiment"]["name"] = (
        f"input_pipeline_workers{workers}_cache{int(cache_tensors)}_progress{int(progress)}_{time.time_ns()}"
    )
    config["training"]["epochs"] = epochs
    config["training"]["save_model"] = False
    config["training"]["artifact_epochs"] = []
    config["training"]["num_workers"] = workers
    config["training"]["pin_memory"] = workers > 0
    config["training"]["persistent_workers"] = workers > 0
    config["training"]["progress"] = progress
    config["datasets"][0]["cache_tensors"] = cache_tensors
    start = time.perf_counter()
    run_gon_experiment(config, seed=0, dataset_name=dataset_name, beta_inf=1.0, beta_opt=1.0)
    elapsed = time.perf_counter() - start
    row = {
        "workers": workers,
        "cache_tensors": cache_tensors,
        "progress": progress,
        "epochs": epochs,
        "total_seconds": elapsed,
        "seconds_per_epoch": elapsed / epochs,
    }
    print(row, flush=True)
    return row


def _write_rows(path: Path, rows: list[dict[str, float | int | bool]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
