"""Exploratory training probes for selecting an epoch budget."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import build_dataset
from models import VariationalGONGenerator
from training.gon import gon_training_step, gon_validation_step
from utils import ExperimentLogger, set_seed


@dataclass(frozen=True)
class EpochProbeResult:
    """Summary of an epoch-selection pilot run."""

    run_dir: Path
    best_epoch: int
    epochs_completed: int
    stopped_early: bool
    best_validation_elbo: float


def run_epoch_probe(
    config: dict[str, Any],
    seed: int,
    dataset_name: str,
    beta_inf: float,
    beta_opt: float,
) -> EpochProbeResult:
    """Run an exploratory train/validation probe for selecting an epoch count."""
    set_seed(seed)

    experiment_config = _mapping(config.get("experiment", {}))
    training_config = _mapping(config.get("training", {}))
    model_config = _mapping(config.get("model", {}))
    probe_config = _mapping(config.get("probe", {}))
    train_dataset_config = _dataset_config(config, dataset_name, split="train")
    validation_dataset_config = _dataset_config(config, dataset_name, split="val")

    train_dataset = build_dataset(dataset_name, train_dataset_config, seed=seed)
    validation_dataset = build_dataset(dataset_name, validation_dataset_config, seed=seed)
    batch_size = int(training_config.get("batch_size", 64))
    device = torch.device(str(training_config.get("device", "cpu")))
    model = _build_model(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_config.get("learning_rate", 1e-3)))

    run_name = f"{experiment_config.get('name', 'epoch_probe')}-{dataset_name}-seed{seed}"
    logger = ExperimentLogger(
        config={**config, "seed": seed, "dataset": dataset_name, "beta_inf": beta_inf, "beta_opt": beta_opt},
        seed=seed,
        results_dir=Path(str(experiment_config.get("results_dir", "results"))),
        run_name=run_name,
    )
    summary_path = logger.run_dir / "epoch_metrics.csv"

    max_epochs = int(training_config.get("epochs", 100))
    patience = int(probe_config.get("patience", 10))
    min_delta = float(probe_config.get("min_delta", 0.0))
    show_progress = bool(training_config.get("progress", True))
    latent_dim = int(model_config.get("latent_dim", 48))

    best_epoch = 0
    best_validation_elbo = float("inf")
    stale_epochs = 0
    stopped_early = False
    epoch_rows: list[dict[str, float | int]] = []

    for epoch in range(max_epochs):
        train_loader = _loader(train_dataset, batch_size, shuffle=True, seed=seed + epoch)
        validation_loader = _loader(validation_dataset, batch_size, shuffle=False, seed=seed)
        train_metrics = _run_training_epoch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            latent_dim=latent_dim,
            beta_inf=beta_inf,
            beta_opt=beta_opt,
            epoch=epoch,
            show_progress=show_progress,
        )
        validation_metrics = _run_validation_epoch(
            model=model,
            loader=validation_loader,
            latent_dim=latent_dim,
            beta_inf=beta_inf,
            beta_opt=beta_opt,
            epoch=epoch,
            show_progress=show_progress,
        )
        row = {
            "epoch": epoch + 1,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in validation_metrics.items()},
            "generalization_gap": validation_metrics["elbo_opt_loss"] - train_metrics["elbo_opt_loss"],
        }
        epoch_rows.append(row)
        _write_epoch_metrics(summary_path, epoch_rows)
        logger.log_metric(step=epoch, metrics=row)
        print(_probe_summary(row), flush=True)

        current_validation_elbo = validation_metrics["elbo_opt_loss"]
        if current_validation_elbo < best_validation_elbo - min_delta:
            best_validation_elbo = current_validation_elbo
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            stopped_early = True
            break

    return EpochProbeResult(
        run_dir=logger.run_dir,
        best_epoch=best_epoch,
        epochs_completed=len(epoch_rows),
        stopped_early=stopped_early,
        best_validation_elbo=best_validation_elbo,
    )


def _run_training_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    latent_dim: int,
    beta_inf: float,
    beta_opt: float,
    epoch: int,
    show_progress: bool,
) -> dict[str, float]:
    metrics = []
    batches = tqdm(loader, desc=f"Train {epoch + 1}", leave=False) if show_progress else loader
    for batch in batches:
        batch = _images(batch).to(next(model.parameters()).device)
        metrics.append(gon_training_step(model, optimizer, batch, latent_dim, beta_inf, beta_opt))
    return _mean_metrics(metrics)


def _run_validation_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    latent_dim: int,
    beta_inf: float,
    beta_opt: float,
    epoch: int,
    show_progress: bool,
) -> dict[str, float]:
    metrics = []
    batches = tqdm(loader, desc=f"Val {epoch + 1}", leave=False) if show_progress else loader
    for batch in batches:
        batch = _images(batch).to(next(model.parameters()).device)
        metrics.append(gon_validation_step(model, batch, latent_dim, beta_inf, beta_opt))
    return _mean_metrics(metrics)


def _loader(dataset: torch.utils.data.Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, generator=generator)


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = metrics[0]
    return {key: sum(item[key] for item in metrics) / len(metrics) for key in keys}


def _write_epoch_metrics(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _probe_summary(row: dict[str, float | int]) -> str:
    return (
        f"Epoch {row['epoch']}: "
        f"train_elbo_opt={row['train_elbo_opt_loss']:.4f} "
        f"val_elbo_opt={row['val_elbo_opt_loss']:.4f} "
        f"val_recon={row['val_elbo_opt_reconstruction']:.4f} "
        f"val_kl={row['val_elbo_opt_kl']:.4f} "
        f"gap={row['generalization_gap']:.4f}"
    )


def _images(batch: Any) -> torch.Tensor:
    return batch[0] if isinstance(batch, list | tuple) else batch


def _build_model(config: dict[str, Any]) -> VariationalGONGenerator:
    return VariationalGONGenerator(
        latent_dim=int(config.get("latent_dim", 48)),
        base_channels=int(config.get("base_channels", 32)),
        output_channels=int(config.get("output_channels", 1)),
    )


def _dataset_config(config: dict[str, Any], dataset_name: str, split: str) -> dict[str, Any]:
    for dataset in config.get("datasets", []):
        dataset_config = _mapping(dataset)
        if dataset_config.get("name") == dataset_name:
            return {**dataset_config, "split": split}
    raise ValueError(f"Dataset not found in config: {dataset_name}")


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("expected a mapping")
    return value
