"""Metrics for latent representations."""

from __future__ import annotations

import torch


def representation_entropy(mu: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Estimate entropy of deterministic mean codes with a diagonal Gaussian fit.

    Given inferred posterior means ``mu`` over a dataset, this fits a diagonal
    Gaussian using the empirical per-dimension variance of those mean codes and
    returns the analytic differential entropy of that fitted distribution.
    This measures the spread of the mean representation across examples; it does
    not use posterior sampling noise or ``logvar``.
    """
    if mu.ndim != 2:
        raise ValueError("mu must have shape [num_examples, latent_dim]")
    variance = torch.var(mu, dim=0, unbiased=False).clamp_min(eps)
    constant = torch.log(torch.tensor(2.0 * torch.pi * torch.e, device=mu.device, dtype=mu.dtype))
    return 0.5 * torch.sum(constant + torch.log(variance))


def representation_perplexity(mu: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return ``exp(entropy)`` for the fitted mean-code distribution."""
    return torch.exp(representation_entropy(mu, eps=eps))
