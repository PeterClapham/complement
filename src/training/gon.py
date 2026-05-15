"""Training helpers for Gradient Origin Networks."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from training.loss import elbo_inf_loss, elbo_opt_loss


def gon_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
    latent_dim: int,
    beta_inf: float,
    beta_opt: float,
) -> dict[str, float]:
    """Run one GON inner/outer optimization step.

    ELBO_inf is used for latent inference by differentiating the inner loss with
    respect to a zero latent origin. ELBO_opt is then used to update the network
    parameters from the inferred latent representation.
    """
    latent_origin = torch.zeros(
        batch.size(0),
        latent_dim,
        device=batch.device,
        dtype=batch.dtype,
        requires_grad=True,
    )
    reconstruction, mu, logvar = model(latent_origin)
    inner_terms = elbo_inf_loss(reconstruction, batch, mu, logvar, beta_inf=beta_inf)
    latent_grad = torch.autograd.grad(
        inner_terms.loss,
        [latent_origin],
        create_graph=True,
        retain_graph=True,
    )[0]

    inferred_latent = -latent_grad
    reconstruction, mu, logvar = model(inferred_latent)
    outer_terms = elbo_opt_loss(reconstruction, batch, mu, logvar, beta_opt=beta_opt)

    optimizer.zero_grad()
    outer_terms.loss.backward()
    optimizer.step()

    return {
        "elbo_inf_loss": _item(inner_terms.loss),
        "elbo_inf_reconstruction": _item(inner_terms.reconstruction),
        "elbo_inf_kl": _item(inner_terms.kl_divergence),
        "elbo_opt_loss": _item(outer_terms.loss),
        "elbo_opt_reconstruction": _item(outer_terms.reconstruction),
        "elbo_opt_kl": _item(outer_terms.kl_divergence),
        "beta_inf": float(beta_inf),
        "beta_opt": float(beta_opt),
    }


def _item(value: Any) -> float:
    return float(value.detach().cpu().item())
