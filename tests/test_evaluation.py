from pathlib import Path

import torch

from evaluation import evaluate_checkpoint_sweep, evaluate_model_grid
from metrics import representation_entropy, representation_perplexity
from models import VariationalGONGenerator


def test_representation_metrics_fit_diagonal_gaussian_to_mean_codes():
    mu = torch.tensor([[0.0, 0.0], [2.0, 4.0]])

    entropy = representation_entropy(mu)
    perplexity = representation_perplexity(mu)

    variances = torch.tensor([1.0, 4.0])
    expected_entropy = 0.5 * torch.sum(torch.log(2.0 * torch.pi * torch.e * variances))
    assert torch.isclose(entropy, expected_entropy)
    assert torch.isclose(perplexity, torch.exp(expected_entropy))


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
    assert len(result.heatmap_paths) == 3
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
