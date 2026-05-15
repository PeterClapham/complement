import json

import torch
import yaml

from data import SyntheticBinaryImageDataset, build_dataset
from training import run_experiment_grid, run_gon_experiment


def test_synthetic_dataset_is_deterministic_and_has_expected_shape():
    first = SyntheticBinaryImageDataset(num_samples=4, image_size=32, channels=1, seed=7)
    second = SyntheticBinaryImageDataset(num_samples=4, image_size=32, channels=1, seed=7)

    assert len(first) == 4
    assert first[0].shape == (1, 32, 32)
    assert torch.equal(first[0], second[0])
    assert set(torch.unique(first[0]).tolist()).issubset({0.0, 1.0})


def test_two_epoch_synthetic_pipeline_saves_artifacts_and_updates_parameters(tmp_path):
    config = _smoke_config(tmp_path)

    result = run_gon_experiment(
        config=config,
        seed=3,
        dataset_name="synthetic_binary",
        beta_inf=0.01,
        beta_opt=10.0,
    )

    assert result.num_steps == 4
    assert result.parameter_update_norm > 0.0
    assert result.model_path is not None
    assert result.model_path.exists()
    assert (result.run_dir / "checkpoint.pt").exists()
    assert result.completed
    assert not result.resumed

    with (result.run_dir / "config.yaml").open(encoding="utf-8") as file:
        saved_config = yaml.safe_load(file)
    assert saved_config["dataset"] == "synthetic_binary"
    assert saved_config["model"]["name"] == "variational_gon"
    assert saved_config["beta_inf"] == 0.01
    assert saved_config["beta_opt"] == 10.0

    metrics = [
        json.loads(line)
        for line in (result.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(metrics) == 4
    assert metrics[0]["metrics"]["beta_inf"] == 0.01
    assert metrics[0]["metrics"]["beta_opt"] == 10.0
    assert metrics[-1]["metrics"]["epoch"] == 1


def test_completed_run_is_resumed_without_retraining(tmp_path):
    config = _smoke_config(tmp_path)

    first = run_gon_experiment(
        config=config,
        seed=3,
        dataset_name="synthetic_binary",
        beta_inf=0.01,
        beta_opt=10.0,
    )
    second = run_gon_experiment(
        config=config,
        seed=3,
        dataset_name="synthetic_binary",
        beta_inf=0.01,
        beta_opt=10.0,
    )

    assert first.run_dir == second.run_dir
    assert second.resumed
    assert second.completed
    assert second.num_steps == first.num_steps


def test_pipeline_runs_different_beta_configuration(tmp_path):
    config = _smoke_config(tmp_path)

    first = run_gon_experiment(
        config=config,
        seed=3,
        dataset_name="synthetic_binary",
        beta_inf=0.01,
        beta_opt=0.01,
    )
    second = run_gon_experiment(
        config=config,
        seed=3,
        dataset_name="synthetic_binary",
        beta_inf=10.0,
        beta_opt=10.0,
    )

    assert first.run_dir != second.run_dir
    assert first.parameter_update_norm > 0.0
    assert second.parameter_update_norm > 0.0

    with (second.run_dir / "config.yaml").open(encoding="utf-8") as file:
        saved_config = yaml.safe_load(file)
    assert saved_config["beta_inf"] == 10.0
    assert saved_config["beta_opt"] == 10.0


def test_grid_runner_executes_all_configured_synthetic_coordinates(tmp_path):
    config = _smoke_config(tmp_path)

    result = run_experiment_grid(config)

    assert len(result.runs) == 4
    assert result.completed == 4
    assert {run.run_dir for run in result.runs}


def test_mnist_dataset_requires_explicit_data_dir():
    try:
        build_dataset("mnist", {"split": "train"}, seed=0)
    except ValueError as error:
        assert "data_dir" in str(error)
    else:
        raise AssertionError("Expected MNIST config without data_dir to fail")


def _smoke_config(tmp_path):
    return {
        "experiment": {"name": "smoke", "results_dir": str(tmp_path)},
        "seeds": [3],
        "datasets": [
            {
                "name": "synthetic_binary",
                "num_samples": 8,
                "image_size": 32,
                "channels": 1,
            }
        ],
        "model": {
            "name": "variational_gon",
            "latent_dim": 8,
            "base_channels": 4,
            "output_channels": 1,
        },
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "device": "cpu",
            "save_model": True,
        },
        "betas": {"values": [0.01, 10.0]},
    }
