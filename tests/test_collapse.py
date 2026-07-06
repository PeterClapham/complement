"""Tests for per-dimension collapse diagnostics."""

from __future__ import annotations

import torch

from metrics import (
    compare_active_sets,
    decoder_weight_scores,
    latent_activity_report,
)
from models import build_model


def _tiny_model() -> torch.nn.Module:
    return build_model(
        {"name": "variational_gon_groupnorm", "latent_dim": 6, "base_channels": 8, "output_channels": 1}
    )


def test_latent_activity_report_flags_low_sigma_dimensions_as_active() -> None:
    num_examples, latent_dim = 64, 4
    logvar = torch.zeros(num_examples, latent_dim)
    # Dimensions 0 and 2 have small sigma (active); 1 and 3 sit at the prior.
    logvar[:, 0] = 2.0 * torch.log(torch.tensor(0.1))  # sigma = 0.1
    logvar[:, 2] = 2.0 * torch.log(torch.tensor(0.2))  # sigma = 0.2
    # sigma = 1.0 for the remaining dimensions.
    mu = torch.zeros_like(logvar)

    report = latent_activity_report(mu, logvar, active_sigma_threshold=0.5)

    assert report.active_indices == [0, 2]
    assert report.active_mask.tolist() == [True, False, True, False]
    assert abs(report.active_fraction - 0.5) < 1e-6


def test_decoder_weight_scores_track_zeroed_input_channel() -> None:
    model = _tiny_model()
    scores_before = decoder_weight_scores(model)
    assert scores_before.decoder_input_usage.shape == (6,)

    # Zero the first decoder input channel; its usage must drop to zero.
    with torch.no_grad():
        model.decoder[0].weight[3].zero_()
    scores_after = decoder_weight_scores(model)

    assert float(scores_after.decoder_input_usage[3]) == 0.0
    assert float(scores_after.decoder_input_usage[0]) > 0.0


def test_compare_active_sets_identifies_revived_and_collapsed() -> None:
    before = torch.tensor([True, False, True, False])
    after = torch.tensor([False, True, True, False])

    comparison = compare_active_sets(before, after)

    assert comparison.revived == [1]
    assert comparison.newly_collapsed == [0]
    assert comparison.stable_active == [2]
    assert comparison.stable_dead == [3]
    assert comparison.num_active_before == 2
    assert comparison.num_active_after == 2
