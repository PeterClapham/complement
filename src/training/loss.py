"""Loss functions used by training code."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ELBOLossTerms:
    """Terms for the negative beta-ELBO objective used as a minimization loss."""

    loss: torch.Tensor
    reconstruction: torch.Tensor
    kl_divergence: torch.Tensor
    beta: float


def negative_beta_elbo(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> ELBOLossTerms:
    """Compute the negative beta-ELBO objective used for minimization.

    The reconstruction term is summed over non-batch dimensions and averaged over
    the batch. The KL term estimates the analytic KL divergence from the diagonal
    Gaussian posterior N(mu, exp(logvar)) to a standard normal prior, averaged over
    the batch. The returned loss is reconstruction + beta * KL, which is the
    negative beta-ELBO up to constants for Bernoulli observations.
    """
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")
    if reconstruction.shape != target.shape:
        raise ValueError(
            "reconstruction and target must have the same shape: "
            f"{tuple(reconstruction.shape)} != {tuple(target.shape)}"
        )
    if mu.shape != logvar.shape:
        raise ValueError(
            "mu and logvar must have the same shape: "
            f"{tuple(mu.shape)} != {tuple(logvar.shape)}"
        )
    if reconstruction.size(0) != mu.size(0):
        raise ValueError("reconstruction and latent statistics must have the same batch size")

    reconstruction_loss = F.binary_cross_entropy(
        reconstruction.flatten(start_dim=1),
        target.flatten(start_dim=1),
        reduction="none",
    ).sum(dim=1).mean()
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    total = reconstruction_loss + beta * kl_divergence
    return ELBOLossTerms(
        loss=total,
        reconstruction=reconstruction_loss,
        kl_divergence=kl_divergence,
        beta=beta,
    )


def elbo_inf_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta_inf: float,
) -> ELBOLossTerms:
    """Negative ELBO_inf used to infer the latent representation.

    In the GON training step, this loss is differentiated with respect to the
    latent origin. Network parameters are not updated from this objective.
    """
    return negative_beta_elbo(reconstruction, target, mu, logvar, beta=beta_inf)


def elbo_opt_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta_opt: float,
) -> ELBOLossTerms:
    """Negative ELBO_opt used to update network parameters after latent inference."""
    return negative_beta_elbo(reconstruction, target, mu, logvar, beta=beta_opt)


def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward-compatible wrapper for the original VAE-style loss."""
    terms = negative_beta_elbo(reconstruction, target, mu, logvar, beta=kl_weight)
    return terms.loss, terms.reconstruction, terms.kl_divergence
