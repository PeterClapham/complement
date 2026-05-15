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


@dataclass(frozen=True)
class ExperimentCoordinate:
    """One dataset/seed/beta configuration in the experiment grid."""

    dataset_name: str
    seed: int
    beta_inf: float
    beta_opt: float


def run_experiment_grid(config: dict[str, Any]) -> GridRunResult:
    """Train every dataset/seed/beta configuration declared in the config."""
    runs: list[TrainingRunResult] = []
    coordinates = experiment_coordinates(config)
    total_runs = len(coordinates)
    print(f"Grid contains {total_runs} runs.", flush=True)
    for run_index, coordinate in enumerate(coordinates, start=1):
        print(
            f"[{run_index}/{total_runs}] "
            f"dataset={coordinate.dataset_name} seed={coordinate.seed} "
            f"beta_inf={coordinate.beta_inf} beta_opt={coordinate.beta_opt}",
            flush=True,
        )
        result = run_coordinate(config, coordinate)
        runs.append(result)
        print(
            f"Finished [{run_index}/{total_runs}] "
            f"steps={result.num_steps} resumed={result.resumed}",
            flush=True,
        )
    return GridRunResult(runs=runs)


def experiment_coordinates(config: dict[str, Any]) -> list[ExperimentCoordinate]:
    """Return every coordinate in the configured experiment grid."""
    return [
        ExperimentCoordinate(
            dataset_name=dataset_name,
            seed=seed,
            beta_inf=beta_config["beta_inf"],
            beta_opt=beta_config["beta_opt"],
        )
        for dataset_name in _dataset_names(config)
        for seed in _seeds(config)
        for beta_config in iter_beta_grid(_beta_values(config))
    ]


def coordinate_for_index(config: dict[str, Any], index: int) -> ExperimentCoordinate:
    """Return one zero-based coordinate from the configured experiment grid."""
    coordinates = experiment_coordinates(config)
    if index < 0 or index >= len(coordinates):
        raise IndexError(f"Grid index {index} is outside [0, {len(coordinates) - 1}]")
    return coordinates[index]


def run_coordinate(config: dict[str, Any], coordinate: ExperimentCoordinate) -> TrainingRunResult:
    """Train one experiment-grid coordinate."""
    return run_gon_experiment(
        config=config,
        seed=coordinate.seed,
        dataset_name=coordinate.dataset_name,
        beta_inf=coordinate.beta_inf,
        beta_opt=coordinate.beta_opt,
    )


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
