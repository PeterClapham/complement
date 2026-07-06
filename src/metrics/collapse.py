"""Per-dimension latent collapse diagnostics.

These functions identify *which* latent coordinates are active versus collapsed,
both from data (the posterior standard deviation of each coordinate) and from the
decoder weights alone (the weight columns feeding each coordinate). Tracking the
identity of collapsed coordinates -- not just their count -- lets an experiment
ask whether the *same* dead coordinates revive across continued training, which
is the measurement that distinguishes reversible from irreversible collapse.

Estimator notes:
- ``mean_sigma`` is the mean over examples of ``sigma = exp(0.5 * logvar)`` for
  each coordinate. A coordinate is called *active* when ``mean_sigma`` is below a
  threshold (its posterior is meaningfully tighter than the unit-variance prior).
- ``decoder_input_usage`` is the L2 norm of the first transposed-convolution
  filter bank driven by each latent coordinate. It uses no data, so it reports
  whether the generative network structurally reads a coordinate at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LatentActivityReport:
    """Per-coordinate data-based activity of a latent representation."""

    mean_sigma: torch.Tensor  # [latent_dim]
    active_mask: torch.Tensor  # [latent_dim] bool
    active_fraction: float
    active_indices: list[int]


@dataclass(frozen=True)
class DecoderWeightScores:
    """Per-coordinate decoder weight magnitudes (no data required)."""

    decoder_input_usage: torch.Tensor  # [latent_dim]
    mu_row_norm: torch.Tensor  # [latent_dim]
    logvar_row_norm: torch.Tensor  # [latent_dim]
    combined: torch.Tensor  # [latent_dim]


@dataclass(frozen=True)
class ActiveSetComparison:
    """How a per-coordinate active mask changed between two snapshots."""

    revived: list[int]  # dead before, active after
    newly_collapsed: list[int]  # active before, dead after
    stable_active: list[int]
    stable_dead: list[int]
    num_active_before: int
    num_active_after: int


def latent_activity_report(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    active_sigma_threshold: float = 0.5,
) -> LatentActivityReport:
    """Classify each latent coordinate as active or collapsed from posterior sigma.

    A coordinate is active when its mean posterior standard deviation is below
    ``active_sigma_threshold``. ``mu`` is accepted for shape validation and future
    use but the activity criterion depends only on ``logvar``.
    """
    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must have the same shape")
    if logvar.ndim != 2:
        raise ValueError("mu and logvar must have shape [num_examples, latent_dim]")
    if active_sigma_threshold <= 0:
        raise ValueError("active_sigma_threshold must be positive")

    sigma = torch.exp(0.5 * logvar)
    mean_sigma = sigma.mean(dim=0)
    active_mask = mean_sigma < active_sigma_threshold
    active_indices = torch.nonzero(active_mask, as_tuple=False).flatten().tolist()
    return LatentActivityReport(
        mean_sigma=mean_sigma,
        active_mask=active_mask,
        active_fraction=float(active_mask.float().mean()),
        active_indices=[int(index) for index in active_indices],
    )


def decoder_weight_scores(model: nn.Module) -> DecoderWeightScores:
    """Return per-coordinate decoder weight magnitudes using no data.

    ``decoder_input_usage[d]`` is the norm of the first transposed-convolution
    filters driven by latent coordinate ``d`` (input channel ``d`` of
    ``model.decoder[0]``). ``mu_row_norm[d]`` and ``logvar_row_norm[d]`` are the
    norms of the rows of ``fc_mu`` / ``fc_logvar`` that *produce* coordinate ``d``.
    A structurally dead coordinate has near-zero decoder input usage.
    """
    if not hasattr(model, "fc_mu") or not hasattr(model, "fc_logvar"):
        raise AttributeError("model must expose fc_mu and fc_logvar linear heads")
    if not hasattr(model, "decoder"):
        raise AttributeError("model must expose a decoder sequential module")

    with torch.no_grad():
        w_mu = model.fc_mu.weight.detach()  # [out=latent, in=latent]
        w_logvar = model.fc_logvar.weight.detach()
        # First decoder layer is ConvTranspose2d(latent_dim, ...): weight is
        # [in_channels=latent_dim, out_channels, kH, kW]; index 0 selects the
        # filters read from each latent coordinate.
        first_conv = model.decoder[0]
        deconv = first_conv.weight.detach()
        decoder_input_usage = deconv.flatten(start_dim=1).norm(dim=1)
        mu_row_norm = w_mu.norm(dim=1)
        logvar_row_norm = w_logvar.norm(dim=1)
        combined = decoder_input_usage + mu_row_norm + logvar_row_norm

    return DecoderWeightScores(
        decoder_input_usage=decoder_input_usage.cpu(),
        mu_row_norm=mu_row_norm.cpu(),
        logvar_row_norm=logvar_row_norm.cpu(),
        combined=combined.cpu(),
    )


def compare_active_sets(
    active_before: torch.Tensor,
    active_after: torch.Tensor,
) -> ActiveSetComparison:
    """Compare two boolean per-coordinate active masks by coordinate identity."""
    if active_before.shape != active_after.shape:
        raise ValueError("active masks must have the same shape")
    if active_before.ndim != 1:
        raise ValueError("active masks must be one-dimensional")

    before = active_before.bool()
    after = active_after.bool()
    revived = torch.nonzero(~before & after, as_tuple=False).flatten().tolist()
    newly_collapsed = torch.nonzero(before & ~after, as_tuple=False).flatten().tolist()
    stable_active = torch.nonzero(before & after, as_tuple=False).flatten().tolist()
    stable_dead = torch.nonzero(~before & ~after, as_tuple=False).flatten().tolist()
    return ActiveSetComparison(
        revived=[int(index) for index in revived],
        newly_collapsed=[int(index) for index in newly_collapsed],
        stable_active=[int(index) for index in stable_active],
        stable_dead=[int(index) for index in stable_dead],
        num_active_before=int(before.sum()),
        num_active_after=int(after.sum()),
    )
