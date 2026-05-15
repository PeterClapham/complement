"""End-to-end experiment runners."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from artifacts import save_reconstruction_grid
from data import build_dataset
from models import VariationalGONGenerator
from training.gon import gon_training_step
from utils import ExperimentLogger, set_seed


@dataclass(frozen=True)
class TrainingRunResult:
    """Summary of a completed training run."""

    run_dir: Path
    model_path: Path | None
    num_steps: int
    parameter_update_norm: float
    completed: bool
    resumed: bool


def run_gon_experiment(
    config: dict[str, Any],
    seed: int,
    dataset_name: str,
    beta_inf: float,
    beta_opt: float,
) -> TrainingRunResult:
    """Run a complete GON experiment for one dataset/seed/beta configuration."""
    set_seed(seed)

    dataset_config = _dataset_config(config, dataset_name)
    training_config = _mapping(config.get("training", {}))
    model_config = _mapping(config.get("model", {}))
    experiment_config = _mapping(config.get("experiment", {}))

    dataset = build_dataset(dataset_name, dataset_config, seed=seed)
    batch_size = int(training_config.get("batch_size", 32))
    num_workers = int(training_config.get("num_workers", 0))
    pin_memory = bool(training_config.get("pin_memory", False))
    persistent_workers = bool(training_config.get("persistent_workers", False))
    device = torch.device(str(training_config.get("device", "cpu")))
    model = _build_model(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_config.get("learning_rate", 1e-3)))

    run_name = (
        f"{experiment_config.get('name', 'variational_gon')}"
        f"-{dataset_name}-seed{seed}-binf{beta_inf}-bopt{beta_opt}"
    )
    run_dir = _run_dir(
        Path(str(experiment_config.get("results_dir", "results"))),
        str(experiment_config.get("name", "variational_gon")),
        dataset_name,
        seed,
        beta_inf,
        beta_opt,
    )
    run_config = {
        **config,
        "seed": seed,
        "dataset": dataset_name,
        "beta_inf": beta_inf,
        "beta_opt": beta_opt,
    }
    logger = ExperimentLogger(
        config=run_config,
        seed=seed,
        results_dir=Path(str(experiment_config.get("results_dir", "results"))),
        run_name=run_name,
        run_dir=run_dir,
    )

    checkpoint_path = logger.run_dir / "checkpoint.pt"
    latent_dim = int(model_config.get("latent_dim", 48))
    epochs = int(training_config.get("epochs", 1))
    artifact_epochs = _artifact_epochs(training_config, epochs)
    start_epoch = 0
    start_batch = 0
    num_steps = 0
    resumed = False
    show_progress = bool(training_config.get("progress", True))

    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
            raise RuntimeError(
                f"Checkpoint could not be loaded and may be corrupted: {checkpoint_path}. "
                "Delete or move this checkpoint, then rerun to restart this configuration."
            ) from error
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"])
        start_batch = int(checkpoint["next_batch"])
        num_steps = int(checkpoint["num_steps"])
        resumed = True
        if bool(checkpoint.get("completed", False)):
            print(f"Already complete: {run_dir}", flush=True)
            model_path = logger.run_dir / "model.pt"
            return TrainingRunResult(
                run_dir=logger.run_dir,
                model_path=model_path if model_path.exists() else None,
                num_steps=num_steps,
                parameter_update_norm=0.0,
                completed=True,
                resumed=True,
            )
        print(
            f"Resuming {run_dir} from epoch {start_epoch + 1}/{epochs}, "
            f"batch {start_batch + 1}",
            flush=True,
        )
    else:
        print(f"Starting {run_dir}", flush=True)

    initial_parameters = parameters_to_vector(model.parameters()).detach().clone()
    sampler = EpochRandomSampler(dataset, seed=seed)
    loader = _loader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    current_epoch = start_epoch
    current_next_batch = start_batch
    try:
        for epoch in range(start_epoch, epochs):
            current_epoch = epoch
            current_next_batch = start_batch
            sampler.set_epoch(epoch)
            epoch_metrics: list[dict[str, float]] = []
            batches = enumerate(loader)
            if show_progress:
                batches = tqdm(
                    batches,
                    total=len(loader),
                    initial=start_batch if epoch == start_epoch else 0,
                    desc=f"Epoch {epoch + 1}/{epochs}",
                    leave=False,
                )
            for batch_index, batch in batches:
                if epoch == start_epoch and batch_index < start_batch:
                    continue
                if isinstance(batch, list | tuple):
                    batch = batch[0]
                batch = batch.to(device, non_blocking=pin_memory)
                metrics = gon_training_step(
                    model=model,
                    optimizer=optimizer,
                    batch=batch,
                    latent_dim=latent_dim,
                    beta_inf=beta_inf,
                    beta_opt=beta_opt,
                )
                logger.log_metric(step=num_steps, metrics={"epoch": epoch, **metrics})
                epoch_metrics.append(metrics)
                num_steps += 1
                current_next_batch = batch_index + 1
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                next_batch=0,
                num_steps=num_steps,
                completed=False,
                run_config=run_config,
            )
            if epoch_metrics:
                print(_epoch_summary(epoch, epochs, epoch_metrics), flush=True)
            completed_epoch = epoch + 1
            if completed_epoch in artifact_epochs:
                _save_epoch_artifacts(
                    logger=logger,
                    model=model,
                    dataset=dataset,
                    latent_dim=latent_dim,
                    beta_inf=beta_inf,
                    batch_size=batch_size,
                    device=device,
                    epoch=completed_epoch,
                )
            start_batch = 0
            current_next_batch = 0
    except KeyboardInterrupt:
        _save_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=current_epoch,
            next_batch=current_next_batch,
            num_steps=num_steps,
            completed=False,
            run_config=run_config,
        )
        raise

    final_parameters = parameters_to_vector(model.parameters()).detach().cpu()
    parameter_update_norm = torch.linalg.vector_norm(final_parameters - initial_parameters.cpu()).item()

    model_path = None
    if bool(training_config.get("save_model", True)):
        model_path = logger.save_model(model)
    reconstruction_grid_path = logger.run_dir / "reconstruction_grid.png"
    save_reconstruction_grid(
        model=model,
        dataset=dataset,
        output_path=reconstruction_grid_path,
        latent_dim=latent_dim,
        beta_inf=beta_inf,
        batch_size=batch_size,
        device=device,
    )
    _save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        next_batch=0,
        num_steps=num_steps,
        completed=True,
        run_config=run_config,
    )

    return TrainingRunResult(
        run_dir=logger.run_dir,
        model_path=model_path,
        num_steps=num_steps,
        parameter_update_norm=float(parameter_update_norm),
        completed=True,
        resumed=resumed,
    )


def _build_model(config: dict[str, Any]) -> VariationalGONGenerator:
    name = str(config.get("name", "variational_gon"))
    if name != "variational_gon":
        raise ValueError(f"Unknown model: {name}")

    return VariationalGONGenerator(
        latent_dim=int(config.get("latent_dim", 48)),
        base_channels=int(config.get("base_channels", 32)),
        output_channels=int(config.get("output_channels", 1)),
    )


def _dataset_config(config: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    datasets = config.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError("datasets must be a list")
    for dataset in datasets:
        dataset_config = _mapping(dataset)
        if dataset_config.get("name") == dataset_name:
            return dataset_config
    raise ValueError(f"Dataset not found in config: {dataset_name}")


class EpochRandomSampler(Sampler[int]):
    """Deterministic per-epoch random sampler for a persistent DataLoader."""

    def __init__(self, dataset: torch.utils.data.Dataset, seed: int) -> None:
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch whose deterministic permutation should be yielded."""
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        # Match DataLoader's seeded-shuffle sequence: iterator creation consumes
        # one random int for the worker base seed before RandomSampler draws.
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.dataset)


def _loader(
    dataset: torch.utils.data.Dataset,
    sampler: Sampler[int],
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("expected a mapping")
    return value


def _run_dir(
    results_dir: Path,
    experiment_name: str,
    dataset_name: str,
    seed: int,
    beta_inf: float,
    beta_opt: float,
) -> Path:
    return (
        results_dir
        / _safe_name(experiment_name)
        / _safe_name(dataset_name)
        / f"seed-{seed}"
        / f"beta-inf-{_format_float(beta_inf)}__beta-opt-{_format_float(beta_opt)}"
    )


def _save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    next_batch: int,
    num_steps: int,
    completed: bool,
    run_config: dict[str, Any],
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "next_batch": next_batch,
        "num_steps": num_steps,
        "completed": completed,
        "config": run_config,
    }
    temporary_path = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)


def _safe_name(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return safe.strip("-_") or "run"


def _format_float(value: float) -> str:
    return str(value).replace(".", "p")


def _artifact_epochs(training_config: dict[str, Any], epochs: int) -> set[int]:
    values = training_config.get("artifact_epochs", [])
    if not isinstance(values, list):
        raise ValueError("training.artifact_epochs must be a list")
    artifact_epochs = {int(value) for value in values}
    invalid = sorted(value for value in artifact_epochs if value <= 0 or value > epochs)
    if invalid:
        raise ValueError(f"training.artifact_epochs must be within [1, {epochs}], got {invalid}")
    return artifact_epochs


def _save_epoch_artifacts(
    logger: ExperimentLogger,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    latent_dim: int,
    beta_inf: float,
    batch_size: int,
    device: torch.device,
    epoch: int,
) -> None:
    model_path = logger.run_dir / f"model_epoch-{epoch:04d}.pt"
    torch.save(model.state_dict(), model_path)
    reconstruction_grid_path = logger.run_dir / f"reconstruction_grid_epoch-{epoch:04d}.png"
    save_reconstruction_grid(
        model=model,
        dataset=dataset,
        output_path=reconstruction_grid_path,
        latent_dim=latent_dim,
        beta_inf=beta_inf,
        batch_size=batch_size,
        device=device,
    )


def _epoch_summary(epoch: int, epochs: int, metrics: list[dict[str, float]]) -> str:
    mean_inf = sum(item["elbo_inf_loss"] for item in metrics) / len(metrics)
    mean_opt = sum(item["elbo_opt_loss"] for item in metrics) / len(metrics)
    mean_recon = sum(item["elbo_opt_reconstruction"] for item in metrics) / len(metrics)
    mean_kl = sum(item["elbo_opt_kl"] for item in metrics) / len(metrics)
    return (
        f"Epoch {epoch + 1}/{epochs}: "
        f"elbo_inf={mean_inf:.4f} "
        f"elbo_opt={mean_opt:.4f} "
        f"recon={mean_recon:.4f} "
        f"kl={mean_kl:.4f}"
    )
