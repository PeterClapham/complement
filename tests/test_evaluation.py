from pathlib import Path

import torch

from evaluation import evaluate_checkpoint_sweep, evaluate_model_grid
from metrics import posterior_collapse_summary, representation_entropy, representation_perplexity
from models import VariationalGONGenerator


def test_representation_metrics_average_histogram_entropy_over_coordinates():
    mu = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])

    entropy = representation_entropy(mu, bins=2)
    perplexity = representation_perplexity(mu, bins=2)

    expected_entropy = torch.log(torch.tensor(2.0))
    assert torch.isclose(entropy, expected_entropy)
    assert torch.isclose(perplexity, torch.exp(expected_entropy))


def test_posterior_collapse_summary_classifies_active_passive_and_mixed_dimensions():
    mu = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    sigma = torch.tensor(
        [
            [0.1, 1.0, 0.1],
            [0.1, 1.0, 0.1],
            [0.1, 1.0, 1.0],
            [0.1, 1.0, 1.0],
        ]
    )
    summary = posterior_collapse_summary(mu, 2.0 * torch.log(sigma))

    assert torch.isclose(summary.active_fraction, torch.tensor(1 / 3))
    assert torch.isclose(summary.passive_fraction, torch.tensor(1 / 3))
    assert torch.isclose(summary.mixed_fraction, torch.tensor(1 / 3))


def test_evaluate_model_grid_writes_metrics_and_heatmaps(tmp_path):
    config = _evaluation_config(tmp_path)
    for beta_inf in [0.01, 1.0]:
        for beta_opt in [0.01, 1.0]:
            model_dir = (
                Path(tmp_path)
                / "smoke_eval"
                / "synthetic_binary"
                / "seed-0"
                / f"beta-inf-{_fmt(beta_inf)}__beta-opt-{_fmt(beta_opt)}"
            )
            model_dir.mkdir(parents=True)
            torch.save(VariationalGONGenerator(latent_dim=8, base_channels=4).state_dict(), model_dir / "model.pt")

    result = evaluate_model_grid(config)

    assert result.metrics_path.exists()
    assert len(result.rows) == 4
    assert len(result.heatmap_paths) == 7
    assert all(path.exists() for path in result.heatmap_paths)


def test_evaluate_checkpoint_sweep_writes_one_directory_per_epoch(tmp_path):
    config = _evaluation_config(tmp_path)
    config["evaluation"]["checkpoint_epochs"] = [50, 80]
    for beta_inf in [0.01, 1.0]:
        for beta_opt in [0.01, 1.0]:
            model_dir = (
                Path(tmp_path)
                / "smoke_eval"
                / "synthetic_binary"
                / "seed-0"
                / f"beta-inf-{_fmt(beta_inf)}__beta-opt-{_fmt(beta_opt)}"
            )
            model_dir.mkdir(parents=True, exist_ok=True)
            state = VariationalGONGenerator(latent_dim=8, base_channels=4).state_dict()
            torch.save(state, model_dir / "model_epoch-0050.pt")
            torch.save(state, model_dir / "model_epoch-0080.pt")

    results = evaluate_checkpoint_sweep(config)

    assert len(results) == 2
    assert results[0].metrics_path.parent.name == "epoch-0050"
    assert results[1].metrics_path.parent.name == "epoch-0080"


def _evaluation_config(tmp_path):
    return {
        "experiment": {"name": "smoke_eval", "results_dir": str(tmp_path)},
        "seeds": [0],
        "datasets": [
            {
                "name": "synthetic_binary",
                "num_samples": 8,
                "image_size": 32,
                "channels": 1,
            }
        ],
        "model": {"latent_dim": 8, "base_channels": 4, "output_channels": 1},
        "training": {"batch_size": 4, "device": "cpu"},
        "evaluation": {"batch_size": 4, "device": "cpu", "output_dir": str(Path(tmp_path) / "evaluation")},
        "betas": {"values": [0.01, 1.0]},
    }


def _fmt(value: float) -> str:
    return str(value).replace(".", "p")
