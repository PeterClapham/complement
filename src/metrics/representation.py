"""Metrics for latent representations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PosteriorCollapseSummary:
    """Dimension fractions under active/passive/mixed posterior criteria."""

    active_fraction: torch.Tensor
    passive_fraction: torch.Tensor
    mixed_fraction: torch.Tensor


def representation_entropy(mu: torch.Tensor, bins: int = 30, eps: float = 1e-12) -> torch.Tensor:
    """Estimate mean coordinate entropy of deterministic mean codes.

    For each latent coordinate ``mu_i``, this forms an equal-width histogram over
    the values of that coordinate across data examples, computes Shannon entropy
    ``-sum_j p_j log p_j`` from the empirical bin probabilities, and returns the
    mean entropy over latent coordinates.
    """
    if mu.ndim != 2:
        raise ValueError("mu must have shape [num_examples, latent_dim]")
    if bins <= 0:
        raise ValueError("bins must be positive")

    entropies = []
    for dimension in range(mu.size(1)):
        values = mu[:, dimension]
        minimum = values.min()
        maximum = values.max()
        if torch.isclose(minimum, maximum):
            entropies.append(torch.zeros((), device=mu.device, dtype=mu.dtype))
            continue
        counts = torch.histc(values, bins=bins, min=float(minimum), max=float(maximum))
        probabilities = counts / counts.sum()
        nonzero = probabilities > 0
        entropies.append(-torch.sum(probabilities[nonzero] * torch.log(probabilities[nonzero].clamp_min(eps))))
    return torch.stack(entropies).mean()


def representation_perplexity(mu: torch.Tensor, bins: int = 30, eps: float = 1e-12) -> torch.Tensor:
    """Return ``exp(mean coordinate entropy)`` for deterministic mean codes."""
    return torch.exp(representation_entropy(mu, bins=bins, eps=eps))


def posterior_collapse_summary(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    active_sigma_threshold: float = 0.5,
    passive_sigma_mean_tolerance: float = 0.1,
    passive_sigma_var_threshold: float = 0.01,
    passive_mu_mean_tolerance: float = 0.1,
    passive_mu_var_threshold: float = 0.01,
) -> PosteriorCollapseSummary:
    """Summarize active, passive, and mixed posterior dimensions.

    ``sigma = exp(0.5 * logvar)`` is computed per example and dimension.
    A dimension is active when its mean posterior standard deviation is below
    ``active_sigma_threshold``. A dimension is passive when its mean sigma is
    close to one, sigma variance is small, mean mu is close to zero, and mu
    variance is small. A dimension is mixed when some examples look active and
    some examples look passive under the per-example sigma/mu thresholds.
    The passive fraction is the direct posterior-collapse measure.
    """
    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must have the same shape")
    if mu.ndim != 2:
        raise ValueError("mu and logvar must have shape [num_examples, latent_dim]")

    sigma = torch.exp(0.5 * logvar)
    mean_sigma = sigma.mean(dim=0)
    var_sigma = sigma.var(dim=0, unbiased=False)
    mean_mu = mu.mean(dim=0)
    var_mu = mu.var(dim=0, unbiased=False)

    active = mean_sigma < active_sigma_threshold
    passive = (
        (torch.abs(mean_sigma - 1.0) <= passive_sigma_mean_tolerance)
        & (var_sigma <= passive_sigma_var_threshold)
        & (torch.abs(mean_mu) <= passive_mu_mean_tolerance)
        & (var_mu <= passive_mu_var_threshold)
    )

    active_examples = sigma < active_sigma_threshold
    passive_examples = (
        (torch.abs(sigma - 1.0) <= passive_sigma_mean_tolerance)
        & (torch.abs(mu) <= passive_mu_mean_tolerance)
    )
    mixed = active_examples.any(dim=0) & passive_examples.any(dim=0)

    return PosteriorCollapseSummary(
        active_fraction=active.float().mean(),
        passive_fraction=passive.float().mean(),
        mixed_fraction=mixed.float().mean(),
    )
