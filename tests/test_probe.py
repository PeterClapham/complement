import csv

import torch

from models import VariationalGONGenerator
from training import gon_validation_step, run_epoch_probe


def test_gon_validation_step_returns_elbo_metrics_without_changing_mode():
    model = VariationalGONGenerator(latent_dim=8, base_channels=4)
    model.train()
    batch = torch.rand(2, 1, 32, 32)

    metrics = gon_validation_step(model, batch, latent_dim=8, beta_inf=0.1, beta_opt=1.0)

    assert model.training
    assert metrics["beta_inf"] == 0.1
    assert metrics["beta_opt"] == 1.0
    assert metrics["elbo_opt_loss"] > 0.0


def test_epoch_probe_writes_epoch_metrics(tmp_path):
    config = {
        "experiment": {"name": "probe", "results_dir": str(tmp_path)},
        "datasets": [
            {
                "name": "synthetic_binary",
                "num_samples": 8,
                "image_size": 32,
                "channels": 1,
            }
        ],
        "model": {"latent_dim": 8, "base_channels": 4, "output_channels": 1},
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "device": "cpu",
            "progress": False,
        },
        "probe": {"patience": 5, "min_delta": 0.0},
    }

    result = run_epoch_probe(config, seed=0, dataset_name="synthetic_binary", beta_inf=0.1, beta_opt=1.0)

    assert result.epochs_completed == 2
    assert result.best_epoch >= 1
    with (result.run_dir / "epoch_metrics.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert "val_elbo_opt_loss" in rows[0]
