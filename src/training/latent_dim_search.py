"""Adaptive latent-dimension search using validation reconstruction loss."""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from training.probe import run_epoch_probe


@dataclass(frozen=True)
class LatentDimensionSearchResult:
    """Saved outputs and next-round suggestion from a latent-dimension search."""

    summary_path: Path
    proposal_path: Path
    rows: list[dict[str, float | int]]
    proposed_dimensions: list[int]
    predicted_optimum: float


@dataclass(frozen=True)
class LatentDimensionSearchRoundsResult:
    """Saved outputs from a multi-round adaptive latent-dimension search."""

    rounds: list[LatentDimensionSearchResult]
    combined_summary_path: Path


def run_latent_dimension_search(config: dict[str, Any]) -> LatentDimensionSearchResult:
    """Run one adaptive search round and propose the next same-size dimension array.

    The search fits a shape-preserving piecewise cubic interpolant to the mean
    final validation reconstruction loss at each tested latent dimension, then
    proposes a denser same-size array around the predicted minimum.
    """
    search_config = _mapping(config.get("latent_dimension_search", {}))
    seeds = [int(seed) for seed in config.get("seeds", [0])]
    dimensions = _dimensions(search_config)
    round_index = int(search_config.get("round", 0))
    output_dir = _output_dir(config, round_index)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int]] = []
    for latent_dim in dimensions:
        for seed in seeds:
            probe_config = _probe_config(config, latent_dim, round_index)
            result = run_epoch_probe(
                config=probe_config,
                seed=seed,
                dataset_name=str(search_config.get("dataset", "mnist")),
                beta_inf=float(search_config.get("beta_inf", 1.0)),
                beta_opt=float(search_config.get("beta_opt", 1.0)),
            )
            rows.append(
                {
                    "round": round_index,
                    "latent_dim": latent_dim,
                    "seed": seed,
                    "final_validation_reconstruction": _final_validation_reconstruction(
                        result.run_dir
                    ),
                    "best_validation_reconstruction": result.best_validation_reconstruction,
                    "best_epoch": result.best_epoch,
                }
            )

    summary_path = output_dir / "latent_dimension_results.csv"
    _write_rows(summary_path, rows)
    aggregated = aggregate_latent_dimension_results(rows)
    proposed_dimensions, predicted_optimum = propose_next_dimensions(
        aggregated,
        current_dimensions=dimensions,
        array_size=len(dimensions),
        low=int(search_config.get("low", min(dimensions))),
        high=int(search_config.get("high", max(dimensions))),
        min_step=int(search_config.get("min_step", 1)),
    )
    proposal_path = output_dir / "next_latent_dimensions.csv"
    _write_rows(proposal_path, [{"latent_dim": value} for value in proposed_dimensions])
    return LatentDimensionSearchResult(
        summary_path=summary_path,
        proposal_path=proposal_path,
        rows=rows,
        proposed_dimensions=proposed_dimensions,
        predicted_optimum=predicted_optimum,
    )


def run_latent_dimension_search_rounds(config: dict[str, Any]) -> LatentDimensionSearchRoundsResult:
    """Run several rounds, halving the searched span around the predicted minimum."""
    search_config = _mapping(config.get("latent_dimension_search", {}))
    num_rounds = int(search_config.get("rounds", 4))
    dimensions = _dimensions(search_config)
    all_rows: list[dict[str, float | int]] = []
    results = []
    low = int(search_config.get("low", min(dimensions)))
    high = int(search_config.get("high", max(dimensions)))

    for round_index in range(num_rounds):
        round_config = deepcopy(config)
        round_config["latent_dimension_search"]["round"] = round_index
        round_config["latent_dimension_search"]["dimensions"] = dimensions
        result = run_latent_dimension_search(round_config)
        results.append(result)
        all_rows.extend(result.rows)
        if round_index < num_rounds - 1:
            dimensions, _ = propose_next_dimensions(
                aggregate_latent_dimension_results(all_rows),
                current_dimensions=dimensions,
                array_size=len(dimensions),
                low=low,
                high=high,
                min_step=int(search_config.get("min_step", 1)),
            )

    combined_summary_path = _output_dir(config, 0).parent / "all_latent_dimension_results.csv"
    _write_rows(combined_summary_path, all_rows)
    return LatentDimensionSearchRoundsResult(
        rounds=results,
        combined_summary_path=combined_summary_path,
    )


def aggregate_latent_dimension_results(
    rows: list[dict[str, float | int]],
) -> list[dict[str, float | int]]:
    """Aggregate validation reconstruction means and standard errors by dimension."""
    values_by_dimension: dict[int, list[float]] = {}
    for row in rows:
        latent_dim = int(row["latent_dim"])
        values_by_dimension.setdefault(latent_dim, []).append(
            float(row["final_validation_reconstruction"])
        )

    aggregated = []
    for latent_dim, values in sorted(values_by_dimension.items()):
        standard_error = stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
        aggregated.append(
            {
                "latent_dim": latent_dim,
                "mean_final_validation_reconstruction": mean(values),
                "standard_error": standard_error,
                "num_seeds": len(values),
            }
        )
    return aggregated


def propose_next_dimensions(
    aggregated: list[dict[str, float | int]],
    current_dimensions: list[int],
    array_size: int,
    low: int,
    high: int,
    min_step: int = 1,
) -> tuple[list[int], float]:
    """Propose a same-size array spanning half the current search window."""
    dimensions = np.asarray([float(row["latent_dim"]) for row in aggregated])
    losses = np.asarray([float(row["mean_final_validation_reconstruction"]) for row in aggregated])
    if len(dimensions) < 2:
        raise ValueError("at least two latent dimensions are required to propose a next round")

    interpolator = PchipInterpolator(dimensions, losses)
    dense_dimensions = np.linspace(dimensions.min(), dimensions.max(), num=4096)
    predicted_optimum = float(dense_dimensions[np.argmin(interpolator(dense_dimensions))])
    current_span = max(current_dimensions) - min(current_dimensions)
    next_span = max(array_size - 1, int(round(current_span / 2)))
    next_span = max(next_span, (array_size - 1) * min_step)
    proposed = _window_dimensions(
        center=predicted_optimum,
        array_size=array_size,
        span=next_span,
        low=low,
        high=high,
    )
    return proposed, predicted_optimum


def within_margin_of_error(
    aggregated: list[dict[str, float | int]],
    margin_multiplier: float = 1.0,
) -> bool:
    """Return whether the best observed point is indistinguishable from a neighbor."""
    best_index = min(
        range(len(aggregated)),
        key=lambda index: float(aggregated[index]["mean_final_validation_reconstruction"]),
    )
    best = aggregated[best_index]
    neighbors = [
        aggregated[index]
        for index in [best_index - 1, best_index + 1]
        if 0 <= index < len(aggregated)
    ]
    if not neighbors:
        return False
    best_mean = float(best["mean_final_validation_reconstruction"])
    best_error = float(best["standard_error"])
    for neighbor in neighbors:
        neighbor_mean = float(neighbor["mean_final_validation_reconstruction"])
        combined_margin = margin_multiplier * (
            best_error**2 + float(neighbor["standard_error"]) ** 2
        ) ** 0.5
        if abs(neighbor_mean - best_mean) <= combined_margin:
            return True
    return False


def _probe_config(config: dict[str, Any], latent_dim: int, round_index: int) -> dict[str, Any]:
    probe_config = deepcopy(config)
    probe_config["experiment"]["name"] = (
        f"{config['experiment']['name']}_round{round_index}_latent{latent_dim}"
    )
    probe_config["model"]["latent_dim"] = latent_dim
    probe_config["probe"]["patience"] = int(probe_config["training"]["epochs"]) + 1
    probe_config["probe"]["monitor_metric"] = "elbo_opt_reconstruction"
    return probe_config


def _dimensions(search_config: dict[str, Any]) -> list[int]:
    configured = search_config.get("dimensions")
    if isinstance(configured, list):
        return [int(value) for value in configured]
    low = int(search_config.get("low", 16))
    high = int(search_config.get("high", 128))
    step = int(search_config.get("step", 16))
    return list(range(low, high + 1, step))


def _output_dir(config: dict[str, Any], round_index: int) -> Path:
    experiment_config = _mapping(config.get("experiment", {}))
    results_dir = Path(str(experiment_config.get("results_dir", "results")))
    experiment_name = str(experiment_config.get("name", "latent_dimension_search"))
    return results_dir / experiment_name / f"round-{round_index:02d}"


def _final_validation_reconstruction(run_dir: Path) -> float:
    with (run_dir / "epoch_metrics.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return float(rows[-1]["val_elbo_opt_reconstruction"])


def _write_rows(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _window_dimensions(center: float, array_size: int, span: int, low: int, high: int) -> list[int]:
    start = int(round(center - span / 2))
    stop = start + span
    if start < low:
        start = low
        stop = start + span
    if stop > high:
        stop = high
        start = stop - span
    raw = np.linspace(start, stop, num=array_size)
    proposed = [int(round(value)) for value in raw]
    proposed[0] = start
    proposed[-1] = stop
    return _deduplicate_dimensions(proposed, array_size, low, high)


def _deduplicate_dimensions(values: list[int], array_size: int, low: int, high: int) -> list[int]:
    unique = sorted(set(values))
    candidate = low
    while len(unique) < array_size and candidate <= high:
        if candidate not in unique:
            unique.append(candidate)
        candidate += 1
    return sorted(unique)[:array_size]


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("expected a mapping")
    return value
