"""Continue training a model from given weights and log per-epoch collapse.

This engine loads an existing set of weights, optionally resets the optimizer,
and trains for a fixed number of epochs at a chosen ``(beta_inf, beta_opt)``.
After every epoch it runs latent inference on one fixed evaluation batch and
records which latent coordinates are active. The per-epoch active mask is written
verbatim so that a later analysis can ask whether the *same* collapsed
coordinates revive -- the measurement that separates reversible from irreversible
collapse. It is the shared workhorse for the irreversibility and hysteresis
studies.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from metrics import decoder_weight_scores, latent_activity_report
from models import build_model
from training.experiment import EpochRandomSampler
from training.gon import gon_training_step
from training.loss import negative_beta_elbo
from utils import set_seed


@dataclass(frozen=True)
class ContinuationResult:
    """Trajectory of a continued-training run."""

    run_dir: Path
    trajectory_path: Path
    rows: list[dict[str, Any]]
    initial_active_mask: torch.Tensor
    final_active_mask: torch.Tensor
    final_state: dict[str, torch.Tensor]


def infer_batch_statistics(
    model: torch.nn.Module,
    images: torch.Tensor,
    latent_dim: int,
    beta_inf: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Infer ``mu``/``logvar`` for one batch with the GON gradient-origin step.

    The model is placed in eval mode so the returned statistics are deterministic.
    Only the inference gradient is taken; no network parameters are updated and no
    graph is retained past this call.
    """
    was_training = model.training
    model.eval()
    with torch.enable_grad():
        origin = torch.zeros(
            images.size(0),
            latent_dim,
            device=images.device,
            dtype=images.dtype,
            requires_grad=True,
        )
        reconstruction, mu, logvar = model(origin)
        inner = negative_beta_elbo(reconstruction, images, mu, logvar, beta=beta_inf)
        latent_grad = torch.autograd.grad(inner.loss, [origin])[0]
        _, mu, logvar = model(-latent_grad)
        mu = mu.detach()
        logvar = logvar.detach()
    if was_training:
        model.train()
    return mu, logvar


def continue_training(
    source_state: dict[str, torch.Tensor],
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    dataset: Dataset,
    eval_images: torch.Tensor,
    beta_inf: float,
    beta_opt: float,
    seed: int,
    run_dir: Path,
    active_sigma_threshold: float = 0.5,
    reset_optimizer: bool = True,
    progress: bool = False,
) -> ContinuationResult:
    """Train from ``source_state`` for ``epochs`` and log per-epoch activity.

    ``eval_images`` is a fixed batch used only for diagnostics; it is moved to the
    training device once and reused each epoch. Row ``epoch=0`` records the
    activity of the source weights before any continued training.
    """
    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(str(training_config.get("device", "cpu")))
    latent_dim = int(model_config.get("latent_dim", 48))
    batch_size = int(training_config.get("batch_size", 128))
    epochs = int(training_config.get("epochs", 1))
    learning_rate = float(training_config.get("learning_rate", 1e-4))

    model = build_model(model_config).to(device)
    model.load_state_dict(source_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if not reset_optimizer and "optimizer_state" in training_config:
        optimizer.load_state_dict(training_config["optimizer_state"])

    eval_images = eval_images.to(device)
    sampler = EpochRandomSampler(dataset, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

    rows: list[dict[str, Any]] = []
    initial_report = _record_epoch(
        rows=rows,
        epoch=0,
        model=model,
        eval_images=eval_images,
        latent_dim=latent_dim,
        beta_inf=beta_inf,
        beta_opt=beta_opt,
        active_sigma_threshold=active_sigma_threshold,
        mean_train_loss=float("nan"),
    )
    initial_active_mask = initial_report.active_mask.clone()

    final_report = initial_report
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        batch_losses: list[float] = []
        batches = loader
        if progress:
            batches = tqdm(loader, desc=f"Continue {epoch + 1}/{epochs}", leave=False)
        for batch in batches:
            images = (batch[0] if isinstance(batch, list | tuple) else batch).to(device)
            metrics = gon_training_step(
                model=model,
                optimizer=optimizer,
                batch=images,
                latent_dim=latent_dim,
                beta_inf=beta_inf,
                beta_opt=beta_opt,
            )
            batch_losses.append(metrics["elbo_opt_loss"])
        mean_train_loss = sum(batch_losses) / len(batch_losses) if batch_losses else float("nan")
        final_report = _record_epoch(
            rows=rows,
            epoch=epoch + 1,
            model=model,
            eval_images=eval_images,
            latent_dim=latent_dim,
            beta_inf=beta_inf,
            beta_opt=beta_opt,
            active_sigma_threshold=active_sigma_threshold,
            mean_train_loss=mean_train_loss,
        )

    trajectory_path = run_dir / "collapse_trajectory.csv"
    _write_rows(trajectory_path, rows)
    model_path = run_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    return ContinuationResult(
        run_dir=run_dir,
        trajectory_path=trajectory_path,
        rows=rows,
        initial_active_mask=initial_active_mask,
        final_active_mask=final_report.active_mask.clone(),
        final_state=final_state,
    )


def latent_activity_row(
    model: torch.nn.Module,
    eval_images: torch.Tensor,
    latent_dim: int,
    beta_inf: float,
    beta_opt: float,
    active_sigma_threshold: float,
    epoch: int,
    mean_train_loss: float,
):
    """Build one trajectory row plus the activity report for a model snapshot.

    Shared by the continuation and replay engines so both log identical columns.
    """
    mu, logvar = infer_batch_statistics(model, eval_images, latent_dim, beta_inf)
    report = latent_activity_report(mu, logvar, active_sigma_threshold=active_sigma_threshold)
    usage = decoder_weight_scores(model).decoder_input_usage
    active = report.active_mask.cpu()
    dead = ~active
    row = {
        "epoch": epoch,
        "beta_inf": float(beta_inf),
        "beta_opt": float(beta_opt),
        "mean_train_loss": mean_train_loss,
        "active_fraction": report.active_fraction,
        "num_active": int(active.sum()),
        "active_mask": "".join("1" if bit else "0" for bit in active.tolist()),
        "mean_decoder_usage_active": _masked_mean(usage, active),
        "mean_decoder_usage_dead": _masked_mean(usage, dead),
    }
    del mu, logvar
    return row, report


def _record_epoch(
    rows: list[dict[str, Any]],
    epoch: int,
    model: torch.nn.Module,
    eval_images: torch.Tensor,
    latent_dim: int,
    beta_inf: float,
    beta_opt: float,
    active_sigma_threshold: float,
    mean_train_loss: float,
):
    row, report = latent_activity_row(
        model=model,
        eval_images=eval_images,
        latent_dim=latent_dim,
        beta_inf=beta_inf,
        beta_opt=beta_opt,
        active_sigma_threshold=active_sigma_threshold,
        epoch=epoch,
        mean_train_loss=mean_train_loss,
    )
    rows.append(row)
    return report


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    if selected.numel() == 0:
        return float("nan")
    return float(selected.mean())


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
