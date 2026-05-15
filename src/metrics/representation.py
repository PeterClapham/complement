"""Metrics for latent representations."""

from __future__ import annotations

import torch


def representation_entropy(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Estimate mean differential entropy of diagonal-Gaussian posteriors.

    For each sample, this computes the analytic entropy of
    ``N(mu, diag(exp(logvar)))`` and returns the batch mean.
    """
    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must have the same shape")
    constant = torch.log(torch.tensor(2.0 * torch.pi * torch.e, device=logvar.device, dtype=logvar.dtype))
    per_sample = 0.5 * torch.sum(constant + logvar, dim=1)
    return per_sample.mean()


def representation_perplexity(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Return ``exp(entropy)`` for the posterior representation distribution."""
    return torch.exp(representation_entropy(mu, logvar))
