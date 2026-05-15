"""Benchmark batch sizes, then probe reconstruction convergence."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

from training import run_epoch_probe
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reconstruction-focused epoch probes.")
    parser.add_argument("--config", type=Path, default=Path("configs/reconstruction_probe.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--benchmark-epochs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    batch_rows = benchmark_batch_sizes(config, args)
    selected_batch_size = _select_batch_size(batch_rows)
    print(f"Selected batch size for long probes: {selected_batch_size}", flush=True)
    run_long_probes(config, args, selected_batch_size)


def benchmark_batch_sizes(config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for batch_size in args.batch_sizes:
        benchmark_config = deepcopy(config)
        benchmark_config["experiment"]["name"] = f"reconstruction_batch_benchmark_bs{batch_size}"
        benchmark_config["training"]["epochs"] = args.benchmark_epochs
        benchmark_config["training"]["batch_size"] = batch_size
        benchmark_config["probe"]["patience"] = args.benchmark_epochs + 1
        result = run_epoch_probe(
            benchmark_config,
            seed=args.seed,
            dataset_name=args.dataset,
            beta_inf=1.0,
            beta_opt=1.0,
        )
        rows_for_run = _read_rows(result.run_dir / "epoch_metrics.csv")
        row = {
            "batch_size": batch_size,
            "mean_epoch_seconds": mean(float(item["epoch_seconds"]) for item in rows_for_run),
            "final_val_reconstruction": float(rows_for_run[-1]["val_elbo_opt_reconstruction"]),
        }
        rows.append(row)
        print(
            f"Batch benchmark: batch_size={batch_size} "
            f"mean_epoch_seconds={row['mean_epoch_seconds']:.2f} "
            f"final_val_recon={row['final_val_reconstruction']:.4f}",
            flush=True,
        )
    _write_rows(Path("results") / "reconstruction_batch_benchmark.csv", rows)
    return rows


def run_long_probes(config: dict[str, Any], args: argparse.Namespace, batch_size: int) -> None:
    coordinates = [(0.01, 0.01), (1.0, 1.0), (10.0, 10.0)]
    rows: list[dict[str, float | int | bool]] = []
    for beta_inf, beta_opt in coordinates:
        probe_config = deepcopy(config)
        probe_config["experiment"]["name"] = (
            f"reconstruction_probe_binf{_format_float(beta_inf)}_bopt{_format_float(beta_opt)}"
        )
        probe_config["training"]["batch_size"] = batch_size
        result = run_epoch_probe(
            probe_config,
            seed=args.seed,
            dataset_name=args.dataset,
            beta_inf=beta_inf,
            beta_opt=beta_opt,
        )
        rows.append(
            {
                "beta_inf": beta_inf,
                "beta_opt": beta_opt,
                "batch_size": batch_size,
                "best_epoch": result.best_epoch,
                "epochs_completed": result.epochs_completed,
                "stopped_early": result.stopped_early,
                "best_val_reconstruction": result.best_validation_reconstruction,
                "best_val_elbo": result.best_validation_elbo,
            }
        )
    _write_rows(Path("results") / "reconstruction_probe_summary.csv", rows)


def _select_batch_size(rows: list[dict[str, float | int]]) -> int:
    best_reconstruction = min(float(row["final_val_reconstruction"]) for row in rows)
    acceptable = [
        row
        for row in rows
        if float(row["final_val_reconstruction"]) <= best_reconstruction * 1.05
    ]
    return int(min(acceptable, key=lambda row: float(row["mean_epoch_seconds"]))["batch_size"])


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write_rows(path: Path, rows: list[dict[str, float | int | bool]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: float) -> str:
    return str(value).replace(".", "p")


if __name__ == "__main__":
    main()
