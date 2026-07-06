"""Smoke test for the continuation-training engine."""

from __future__ import annotations

import torch

from data import build_dataset
from models import build_model
from training.continuation import continue_training


def test_continue_training_writes_trajectory_with_initial_row(tmp_path) -> None:
    model_config = {
        "name": "variational_gon_groupnorm",
        "latent_dim": 6,
        "base_channels": 8,
        "output_channels": 1,
    }
    training_config = {"epochs": 2, "batch_size": 4, "learning_rate": 1e-3, "device": "cpu"}
    dataset = build_dataset("synthetic_binary", {"num_samples": 16, "image_size": 32}, seed=0)
    eval_images = torch.stack([dataset[i] for i in range(8)])
    source_state = build_model(model_config).state_dict()

    result = continue_training(
        source_state=source_state,
        model_config=model_config,
        training_config=training_config,
        dataset=dataset,
        eval_images=eval_images,
        beta_inf=1.0,
        beta_opt=1.0,
        seed=0,
        run_dir=tmp_path / "continue",
        active_sigma_threshold=0.5,
    )

    assert result.trajectory_path.exists()
    assert (result.run_dir / "model.pt").exists()
    # One initial row (epoch 0) plus one row per completed epoch.
    assert [row["epoch"] for row in result.rows] == [0, 1, 2]
    assert result.initial_active_mask.shape == (6,)
    assert result.final_active_mask.shape == (6,)
    for row in result.rows:
        assert len(row["active_mask"]) == 6
