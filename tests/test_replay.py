"""Tests for aggregate-posterior replay training."""

from __future__ import annotations

import torch
from torch.nn.utils import parameters_to_vector

from models import build_model
from training.replay import ReplayBuffer, gon_replay_training_step


def _tiny_model() -> torch.nn.Module:
    return build_model(
        {"name": "variational_gon_groupnorm", "latent_dim": 6, "base_channels": 8, "output_channels": 1}
    )


def test_replay_buffer_wraps_and_samples() -> None:
    buffer = ReplayBuffer(capacity=4, latent_dim=3)
    assert buffer.size == 0

    buffer.add(torch.ones(3, 3))
    assert buffer.size == 3
    # Adding three more wraps around and caps at capacity.
    buffer.add(2.0 * torch.ones(3, 3))
    assert buffer.size == 4

    sample = buffer.sample(10)
    assert sample.shape == (10, 3)
    # Every sampled value came from one of the inserted batches.
    assert set(sample.flatten().tolist()) <= {1.0, 2.0}


def test_replay_buffer_add_larger_than_capacity() -> None:
    buffer = ReplayBuffer(capacity=4, latent_dim=2)
    buffer.add(torch.arange(20, dtype=torch.float32).reshape(10, 2))
    assert buffer.size == 4
    assert buffer.sample(5).shape == (5, 2)


def test_replay_training_step_updates_parameters_and_fills_buffer() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=32, latent_dim=6)
    batch = (torch.rand(4, 1, 32, 32) > 0.5).float()

    before = parameters_to_vector(model.parameters()).detach().clone()
    metrics = gon_replay_training_step(
        model=model,
        optimizer=optimizer,
        batch=batch,
        latent_dim=6,
        beta_inf=1.0,
        beta_opt=1.0,
        buffer=buffer,
        replay_weight=1.0,
        do_replay=False,  # buffer starts empty; real step only
    )
    after = parameters_to_vector(model.parameters()).detach().clone()

    assert buffer.size == 4
    assert torch.linalg.vector_norm(after - before) > 0
    assert "elbo_opt_loss" in metrics and "replay_loss" in metrics


def test_replay_step_runs_when_buffer_is_warm() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=32, latent_dim=6)
    buffer.add(torch.randn(16, 6))
    batch = (torch.rand(4, 1, 32, 32) > 0.5).float()

    metrics = gon_replay_training_step(
        model=model,
        optimizer=optimizer,
        batch=batch,
        latent_dim=6,
        beta_inf=1.0,
        beta_opt=1.0,
        buffer=buffer,
        replay_weight=0.5,
        do_replay=True,
    )
    # A replay update actually ran, so the recorded replay loss is finite.
    assert metrics["replay_loss"] == metrics["replay_loss"]  # not NaN
