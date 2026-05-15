"""Evaluation over completed experiment grids."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import build_dataset
from metrics import representation_entropy, representation_perplexity
from models import VariationalGONGenerator
from training import experiment_coordinates
from training.loss import negative_beta_elbo

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class EvaluationGridResult:
    """Saved outputs from a completed evaluation grid."""

    metrics_path: Path
    heatmap_paths: list[Path]
    rows: list[dict[str, float | int | str]]


def evaluate_model_grid(config: dict[str, Any], model_filename: str = "model.pt") -> EvaluationGridResult:
    """Evaluate every completed model in a configured grid and save heatmaps."""
    rows: list[dict[str, float | int | str]] = []
    for coordinate in experiment_coordinates(config):
        rows.append(_evaluate_coordinate(config, coordinate, model_filename))

    output_dir = _evaluation_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "test_metrics.csv"
    _write_rows(metrics_path, rows)
    heatmap_paths = _save_heatmaps(config, rows, output_dir)
    return EvaluationGridResult(metrics_path=metrics_path, heatmap_paths=heatmap_paths, rows=rows)


def evaluate_checkpoint_sweep(config: dict[str, Any]) -> list[EvaluationGridResult]:
    """Evaluate every configured saved-epoch model grid."""
    evaluation_config = _mapping(config.get("evaluation", {}))
    checkpoint_epochs = [int(value) for value in evaluation_config.get("checkpoint_epochs", [])]
    if not checkpoint_epochs:
        raise ValueError("evaluation.checkpoint_epochs must be a non-empty list")

    base_output_dir = _evaluation_dir(config)
    results = []
    for epoch in checkpoint_epochs:
        epoch_config = {
            **config,
            "evaluation": {
                **evaluation_config,
                "output_dir": str(base_output_dir / f"epoch-{epoch:04d}"),
            },
        }
        results.append(evaluate_model_grid(epoch_config, model_filename=f"model_epoch-{epoch:04d}.pt"))
    return results


def _evaluate_coordinate(
    config: dict[str, Any],
    coordinate: Any,
    model_filename: str,
) -> dict[str, float | int | str]:
    dataset_config = _dataset_config(config, coordinate.dataset_name)
    evaluation_config = _mapping(config.get("evaluation", {}))
    model_config = _mapping(config.get("model", {}))
    training_config = _mapping(config.get("training", {}))

    dataset = build_dataset(coordinate.dataset_name, {**dataset_config, "split": "test"}, seed=coordinate.seed)
    batch_size = int(evaluation_config.get("batch_size", training_config.get("batch_size", 64)))
    device = torch.device(str(evaluation_config.get("device", training_config.get("device", "cpu"))))
    model = _build_model(model_config).to(device)
    model_path = _model_path(config, coordinate, model_filename)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    latent_dim = int(model_config.get("latent_dim", 48))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    metric_sums = {
        "elbo": 0.0,
        "reconstruction": 0.0,
        "kl": 0.0,
    }
    representation_means = []
    num_examples = 0
    batches = tqdm(loader, desc=f"Eval b_inf={coordinate.beta_inf} b_opt={coordinate.beta_opt}", leave=False)
    for batch in batches:
        images = _images(batch).to(device)
        reconstruction, mu, logvar = _infer_latent(model, images, latent_dim, coordinate.beta_inf)
        elbo_terms = negative_beta_elbo(reconstruction, images, mu, logvar, beta=1.0)
        batch_size_actual = images.size(0)
        metric_sums["elbo"] += -float(elbo_terms.loss.detach().cpu()) * batch_size_actual
        metric_sums["reconstruction"] += float(elbo_terms.reconstruction.detach().cpu()) * batch_size_actual
        metric_sums["kl"] += float(elbo_terms.kl_divergence.detach().cpu()) * batch_size_actual
        representation_means.append(mu.detach().cpu())
        num_examples += batch_size_actual

    means = {key: value / num_examples for key, value in metric_sums.items()}
    all_means = torch.cat(representation_means, dim=0)
    means["representation_entropy"] = float(representation_entropy(all_means))
    means["representation_perplexity"] = float(representation_perplexity(all_means))
    return {
        "dataset": coordinate.dataset_name,
        "seed": coordinate.seed,
        "beta_inf": coordinate.beta_inf,
        "beta_opt": coordinate.beta_opt,
        **means,
    }


def _infer_latent(
    model: torch.nn.Module,
    images: torch.Tensor,
    latent_dim: int,
    beta_inf: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        latent_origin = torch.zeros(
            images.size(0),
            latent_dim,
            device=images.device,
            dtype=images.dtype,
            requires_grad=True,
        )
        reconstruction, mu, logvar = model(latent_origin)
        inner_terms = negative_beta_elbo(reconstruction, images, mu, logvar, beta=beta_inf)
        latent_grad = torch.autograd.grad(inner_terms.loss, [latent_origin])[0]
        inferred_latent = -latent_grad
        return model(inferred_latent)


def _save_heatmaps(
    config: dict[str, Any],
    rows: list[dict[str, float | int | str]],
    output_dir: Path,
) -> list[Path]:
    beta_values = _beta_values(config)
    metric_names = ["elbo", "representation_perplexity", "representation_entropy"]
    heatmap_paths = []
    for metric_name in metric_names:
        values = np.full((len(beta_values), len(beta_values)), np.nan)
        counts = np.zeros((len(beta_values), len(beta_values)))
        for row in rows:
            row_index = beta_values.index(float(row["beta_inf"]))
            column_index = beta_values.index(float(row["beta_opt"]))
            if np.isnan(values[row_index, column_index]):
                values[row_index, column_index] = 0.0
            values[row_index, column_index] += float(row[metric_name])
            counts[row_index, column_index] += 1.0
        values = values / np.where(counts == 0.0, np.nan, counts)

        figure, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(values, origin="lower", aspect="auto")
        axis.set_xticks(range(len(beta_values)), beta_values)
        axis.set_yticks(range(len(beta_values)), beta_values)
        axis.set_xlabel("beta_opt")
        axis.set_ylabel("beta_inf")
        axis.set_title(metric_name.replace("_", " "))
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        path = output_dir / f"{metric_name}_heatmap.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        heatmap_paths.append(path)
    return heatmap_paths


def _write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _evaluation_dir(config: dict[str, Any]) -> Path:
    evaluation_config = _mapping(config.get("evaluation", {}))
    if "output_dir" in evaluation_config:
        return Path(str(evaluation_config["output_dir"]))
    experiment_config = _mapping(config.get("experiment", {}))
    return Path(str(experiment_config.get("results_dir", "results"))) / "evaluation"


def _model_path(config: dict[str, Any], coordinate: Any, model_filename: str) -> Path:
    experiment_config = _mapping(config.get("experiment", {}))
    results_dir = Path(str(experiment_config.get("results_dir", "results")))
    experiment_name = str(experiment_config.get("name", "variational_gon"))
    return (
        results_dir
        / experiment_name
        / coordinate.dataset_name
        / f"seed-{coordinate.seed}"
        / f"beta-inf-{_format_float(coordinate.beta_inf)}__beta-opt-{_format_float(coordinate.beta_opt)}"
        / model_filename
    )


def _build_model(config: dict[str, Any]) -> VariationalGONGenerator:
    return VariationalGONGenerator(
        latent_dim=int(config.get("latent_dim", 48)),
        base_channels=int(config.get("base_channels", 32)),
        output_channels=int(config.get("output_channels", 1)),
    )


def _dataset_config(config: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    for dataset in config.get("datasets", []):
        dataset_config = _mapping(dataset)
        if dataset_config.get("name") == dataset_name:
            return dataset_config
    raise ValueError(f"Dataset not found in config: {dataset_name}")


def _beta_values(config: dict[str, Any]) -> list[float]:
    return [float(value) for value in _mapping(config.get("betas", {})).get("values", [])]


def _images(batch: Any) -> torch.Tensor:
    return batch[0] if isinstance(batch, list | tuple) else batch


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("expected a mapping")
    return value


def _format_float(value: float) -> str:
    return str(value).replace(".", "p")
