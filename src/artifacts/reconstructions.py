"""Reconstruction image artifacts for trained GON models."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image


def save_reconstruction_grid(
    model: torch.nn.Module,
    dataset: Dataset,
    output_path: Path,
    latent_dim: int,
    beta_inf: float,
    batch_size: int,
    device: torch.device | str,
    nrow: int = 8,
) -> Path:
    """Save a grid of reconstructions from the first deterministic dataset batch.

    The reconstructions use the same one-step latent inference estimator as GON
    training: differentiate negative ``ELBO_inf`` with respect to a zero latent
    origin, negate that gradient, then decode the inferred latent vectors.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    batch = next(iter(loader))
    if isinstance(batch, list | tuple):
        batch = batch[0]
    batch = batch.to(device)

    was_training = model.training
    model.eval()
    with torch.enable_grad():
        from training.loss import elbo_inf_loss

        latent_origin = torch.zeros(
            batch.size(0),
            latent_dim,
            device=batch.device,
            dtype=batch.dtype,
            requires_grad=True,
        )
        reconstruction, mu, logvar = model(latent_origin)
        inner_terms = elbo_inf_loss(reconstruction, batch, mu, logvar, beta_inf=beta_inf)
        latent_grad = torch.autograd.grad(inner_terms.loss, [latent_origin])[0]
        inferred_latent = -latent_grad
        reconstruction, _, _ = model(inferred_latent)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(reconstruction.detach().cpu(), output_path, nrow=nrow)
    if was_training:
        model.train()
    return output_path
