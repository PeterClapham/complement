"""Aggregate-posterior replay for the GON training loop.

The complementary-learning-systems reading of posterior collapse predicts that
rehearsing the generative network on samples from its own aggregate posterior --
the empirical distribution of inferred codes -- should protect active latent
coordinates from collapsing under strong ``beta_opt`` pressure. This module
implements that rehearsal: a capped buffer stores recently inferred codes, and an
interleaved replay step trains the decoder to reconstruct images it generates from
codes resampled out of that buffer.

Memory discipline: the buffer is a fixed-size tensor of detached CPU codes;
generated rehearsal images are produced under ``no_grad`` and consumed as constant
targets, so no computation graph is retained between updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from models import build_model
from training.continuation import ContinuationResult, latent_activity_row
from training.experiment import EpochRandomSampler
from training.loss import elbo_inf_loss, elbo_opt_loss
from utils import set_seed


@dataclass
class ReplayBuffer:
    """Fixed-capacity ring buffer of inferred latent codes on CPU."""

    capacity: int
    latent_dim: int

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self._data = torch.zeros(self.capacity, self.latent_dim)
        self._size = 0
        self._position = 0

    @property
    def size(self) -> int:
        return self._size

    def is_warm(self, minimum_size: int) -> bool:
        return self._size >= max(1, minimum_size)

    def add(self, codes: torch.Tensor) -> None:
        """Insert a batch of codes, overwriting oldest entries when full."""
        batch = codes.detach().to("cpu", dtype=self._data.dtype)
        if batch.ndim != 2 or batch.size(1) != self.latent_dim:
            raise ValueError("codes must have shape [batch, latent_dim]")
        count = batch.size(0)
        for start in range(0, count, self.capacity):
            chunk = batch[start : start + self.capacity]
            chunk_len = chunk.size(0)
            end = self._position + chunk_len
            if end <= self.capacity:
                self._data[self._position : end] = chunk
            else:
                first = self.capacity - self._position
                self._data[self._position :] = chunk[:first]
                self._data[: chunk_len - first] = chunk[first:]
            self._position = (self._position + chunk_len) % self.capacity
            self._size = min(self.capacity, self._size + chunk_len)

    def sample(self, count: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Sample ``count`` codes with replacement from stored entries."""
        if self._size == 0:
            raise RuntimeError("cannot sample from an empty buffer")
        indices = torch.randint(0, self._size, (count,), generator=generator)
        return self._data[indices].clone()


def gon_replay_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
    latent_dim: int,
    beta_inf: float,
    beta_opt: float,
    buffer: ReplayBuffer,
    replay_weight: float,
    do_replay: bool,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Run one real GON step, store its codes, and optionally one replay step.

    The real step matches ``gon_training_step`` but also captures the inferred
    codes for the aggregate-posterior buffer. When ``do_replay`` is true and the
    buffer holds samples, an additional interleaved update trains the network to
    reconstruct images it generates from resampled codes.
    """
    real_metrics = _gon_step(model, optimizer, batch, latent_dim, beta_inf, beta_opt)
    buffer.add(real_metrics.pop("_codes"))

    replay_loss = float("nan")
    if do_replay and replay_weight > 0.0 and buffer.size > 0:
        device = batch.device
        codes = buffer.sample(batch.size(0), generator=generator).to(device, dtype=batch.dtype)
        with torch.no_grad():
            rehearsed = model.decoder(codes.unsqueeze(-1).unsqueeze(-1))
        replay_metrics = _gon_step(
            model, optimizer, rehearsed, latent_dim, beta_inf, beta_opt, loss_scale=replay_weight
        )
        replay_loss = replay_metrics["elbo_opt_loss"]

    real_metrics["replay_loss"] = replay_loss
    return real_metrics


def _gon_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
    latent_dim: int,
    beta_inf: float,
    beta_opt: float,
    loss_scale: float = 1.0,
) -> dict[str, float]:
    origin = torch.zeros(
        batch.size(0), latent_dim, device=batch.device, dtype=batch.dtype, requires_grad=True
    )
    reconstruction, mu, logvar = model(origin)
    inner = elbo_inf_loss(reconstruction, batch, mu, logvar, beta_inf=beta_inf)
    latent_grad = torch.autograd.grad(inner.loss, [origin], create_graph=True, retain_graph=True)[0]
    reconstruction, mu, logvar = model(-latent_grad)
    outer = elbo_opt_loss(reconstruction, batch, mu, logvar, beta_opt=beta_opt)

    optimizer.zero_grad()
    (loss_scale * outer.loss).backward()
    optimizer.step()
    return {
        "elbo_opt_loss": float(outer.loss.detach().cpu().item()),
        "elbo_opt_reconstruction": float(outer.reconstruction.detach().cpu().item()),
        "elbo_opt_kl": float(outer.kl_divergence.detach().cpu().item()),
        "_codes": mu.detach(),
    }


def run_replay_training(
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    dataset: Dataset,
    eval_images: torch.Tensor,
    beta_inf: float,
    beta_opt: float,
    seed: int,
    run_dir: Path,
    replay_weight: float,
    replay_every: int = 1,
    buffer_capacity: int = 4096,
    warmup_batches: int = 8,
    active_sigma_threshold: float = 0.5,
    source_state: dict[str, torch.Tensor] | None = None,
    progress: bool = False,
) -> ContinuationResult:
    """Train a GON with optional aggregate-posterior replay and log activity.

    Set ``replay_weight = 0`` for the no-replay control. Row ``epoch=0`` records
    the activity of the initial weights before any training.
    """
    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(str(training_config.get("device", "cpu")))
    latent_dim = int(model_config.get("latent_dim", 48))
    batch_size = int(training_config.get("batch_size", 128))
    epochs = int(training_config.get("epochs", 1))
    learning_rate = float(training_config.get("learning_rate", 1e-4))

    model = build_model(model_config).to(device)
    if source_state is not None:
        model.load_state_dict(source_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(capacity=int(buffer_capacity), latent_dim=latent_dim)
    warmup_size = int(warmup_batches) * batch_size

    eval_images = eval_images.to(device)
    sampler = EpochRandomSampler(dataset, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

    rows: list[dict[str, Any]] = []
    initial_row, initial_report = latent_activity_row(
        model, eval_images, latent_dim, beta_inf, beta_opt, active_sigma_threshold, 0, float("nan")
    )
    rows.append(initial_row)
    initial_active_mask = initial_report.active_mask.clone()

    final_report = initial_report
    step = 0
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        batch_losses: list[float] = []
        batches = loader
        if progress:
            batches = tqdm(loader, desc=f"Replay {epoch + 1}/{epochs}", leave=False)
        for batch in batches:
            images = (batch[0] if isinstance(batch, list | tuple) else batch).to(device)
            do_replay = (
                replay_weight > 0.0
                and buffer.is_warm(warmup_size)
                and (step % max(1, int(replay_every)) == 0)
            )
            metrics = gon_replay_training_step(
                model=model,
                optimizer=optimizer,
                batch=images,
                latent_dim=latent_dim,
                beta_inf=beta_inf,
                beta_opt=beta_opt,
                buffer=buffer,
                replay_weight=replay_weight,
                do_replay=do_replay,
            )
            batch_losses.append(metrics["elbo_opt_loss"])
            step += 1
        mean_train_loss = sum(batch_losses) / len(batch_losses) if batch_losses else float("nan")
        row, final_report = latent_activity_row(
            model, eval_images, latent_dim, beta_inf, beta_opt,
            active_sigma_threshold, epoch + 1, mean_train_loss,
        )
        rows.append(row)

    trajectory_path = run_dir / "collapse_trajectory.csv"
    _write_rows(trajectory_path, rows)
    torch.save(model.state_dict(), run_dir / "model.pt")
    final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    return ContinuationResult(
        run_dir=run_dir,
        trajectory_path=trajectory_path,
        rows=rows,
        initial_active_mask=initial_active_mask,
        final_active_mask=final_report.active_mask.clone(),
        final_state=final_state,
    )


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
