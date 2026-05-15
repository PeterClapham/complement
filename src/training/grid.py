"""Grid execution for experiment sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from training.experiment import TrainingRunResult, run_gon_experiment
from utils import iter_beta_grid


@dataclass(frozen=True)
class GridRunResult:
    """Summary of a full grid execution."""

    runs: list[TrainingRunResult]

    @property
    def completed(self) -> int:
        return sum(result.completed for result in self.runs)


def run_experiment_grid(config: dict[str, Any]) -> GridRunResult:
    """Train every dataset/seed/beta configuration declared in the config."""
    runs: list[TrainingRunResult] = []
    dataset_names = _dataset_names(config)
    seeds = _seeds(config)
    beta_grid = iter_beta_grid(_beta_values(config))
    total_runs = len(dataset_names) * len(seeds) * len(beta_grid)
    run_index = 0
    print(f"Grid contains {total_runs} runs.", flush=True)
    for dataset_name in dataset_names:
        for seed in seeds:
            for beta_config in beta_grid:
                run_index += 1
                print(
                    f"[{run_index}/{total_runs}] "
                    f"dataset={dataset_name} seed={seed} "
                    f"beta_inf={beta_config['beta_inf']} beta_opt={beta_config['beta_opt']}",
                    flush=True,
                )
                result = run_gon_experiment(
                    config=config,
                    seed=seed,
                    dataset_name=dataset_name,
                    beta_inf=beta_config["beta_inf"],
                    beta_opt=beta_config["beta_opt"],
                )
                runs.append(result)
                print(
                    f"Finished [{run_index}/{total_runs}] "
                    f"steps={result.num_steps} resumed={result.resumed}",
                    flush=True,
                )
    return GridRunResult(runs=runs)


def _dataset_names(config: dict[str, Any]) -> list[str]:
    datasets = config.get("datasets", [])
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a non-empty list")
    names = []
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("each dataset config must be a mapping")
        names.append(str(dataset["name"]))
    return names


def _seeds(config: dict[str, Any]) -> list[int]:
    seeds = config.get("seeds", [])
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list")
    return [int(seed) for seed in seeds]


def _beta_values(config: dict[str, Any]) -> list[float]:
    betas = config.get("betas", {})
    if not isinstance(betas, dict):
        raise ValueError("betas must be a mapping")
    values = betas.get("values", [])
    if not isinstance(values, list) or not values:
        raise ValueError("betas.values must be a non-empty list")
    return [float(value) for value in values]
