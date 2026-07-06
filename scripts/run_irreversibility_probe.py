"""Irreversibility probe (prediction P-A).

Load a collapsed model from a completed beta grid, then continue training under a
released ``beta_opt`` (the regularization pressure that caused collapse is removed
or reduced). If posterior collapse is an equilibrium property, the previously
dead latent coordinates should re-activate once the pressure is gone. If collapse
is a consolidation phenomenon, the same coordinates should stay dead. A hold
control continues at the original ``beta_opt`` for comparison.

Usage:
    python scripts/run_irreversibility_probe.py --config configs/irreversibility_mnist.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from data import build_dataset
from metrics import compare_active_sets
from training import continue_training
from utils import beta_grid_run_dir, collapse_dynamics_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue a collapsed model under released beta_opt.")
    parser.add_argument("--config", type=Path, default=Path("configs/irreversibility_mnist.yaml"))
    return parser.parse_args()


def _images(item: Any) -> torch.Tensor:
    return item[0] if isinstance(item, list | tuple) else item


def _eval_batch(dataset: Any, count: int) -> torch.Tensor:
    count = min(count, len(dataset))
    return torch.stack([_images(dataset[index]) for index in range(count)])


def _dataset_entry(config: dict[str, Any], name: str) -> dict[str, Any]:
    for entry in config.get("datasets", []):
        if entry.get("name") == name:
            return entry
    raise ValueError(f"Dataset not found in config: {name}")


def _run_arm(
    label: str,
    source_state: dict[str, torch.Tensor],
    config: dict[str, Any],
    train_dataset: Any,
    eval_images: torch.Tensor,
    beta_inf: float,
    beta_opt: float,
    seed: int,
    output_dir: Path,
    active_sigma_threshold: float,
) -> dict[str, Any]:
    result = continue_training(
        source_state=source_state,
        model_config=config["model"],
        training_config=config["training"],
        dataset=train_dataset,
        eval_images=eval_images,
        beta_inf=beta_inf,
        beta_opt=beta_opt,
        seed=seed,
        run_dir=output_dir / label,
        active_sigma_threshold=active_sigma_threshold,
        progress=bool(config["training"].get("progress", False)),
    )
    comparison = compare_active_sets(result.initial_active_mask, result.final_active_mask)
    summary = {
        "arm": label,
        "beta_opt": beta_opt,
        "num_active_before": comparison.num_active_before,
        "num_active_after": comparison.num_active_after,
        "revived": comparison.revived,
        "newly_collapsed": comparison.newly_collapsed,
        "num_revived": len(comparison.revived),
        "trajectory": str(result.trajectory_path),
    }
    print(
        f"[{label}] beta_opt={beta_opt}: active {comparison.num_active_before} -> "
        f"{comparison.num_active_after}, revived {len(comparison.revived)} dims",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    settings = config["irreversibility"]
    results_dir = Path(str(config["experiment"].get("results_dir", "results")))
    seed = int(settings.get("seed", 0))
    dataset_name = str(settings["dataset"])
    beta_inf = float(settings.get("beta_inf", 1.0))
    collapsed_beta_opt = float(settings["collapsed_beta_opt"])
    release_beta_opt = float(settings["release_beta_opt"])
    active_sigma_threshold = float(settings.get("active_sigma_threshold", 0.5))

    source_run_dir = beta_grid_run_dir(
        results_dir,
        str(settings["source_experiment_name"]),
        dataset_name,
        seed,
        beta_inf,
        collapsed_beta_opt,
    )
    source_model_path = source_run_dir / "model.pt"
    if not source_model_path.exists():
        raise FileNotFoundError(f"Source model not found: {source_model_path}")
    device = torch.device(str(config["training"].get("device", "cpu")))
    source_state = torch.load(source_model_path, map_location=device, weights_only=True)

    entry = _dataset_entry(config, dataset_name)
    eval_dataset = build_dataset(dataset_name, {**entry, "split": "test"}, seed=seed)
    train_dataset = build_dataset(dataset_name, {**entry, "split": "train"}, seed=seed)
    eval_images = _eval_batch(eval_dataset, int(settings.get("eval_batch_size", 512)))

    output_dir = collapse_dynamics_dir(results_dir, str(config["experiment"]["name"])) / "irreversibility"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        _run_arm(
            "release", source_state, config, train_dataset, eval_images,
            beta_inf, release_beta_opt, seed, output_dir, active_sigma_threshold,
        ),
        _run_arm(
            "hold", source_state, config, train_dataset, eval_images,
            beta_inf, collapsed_beta_opt, seed, output_dir, active_sigma_threshold,
        ),
    ]

    summary_path = output_dir / "irreversibility_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "source_model": str(source_model_path),
                "beta_inf": beta_inf,
                "collapsed_beta_opt": collapsed_beta_opt,
                "release_beta_opt": release_beta_opt,
                "epochs": int(config["training"]["epochs"]),
                "arms": summaries,
            },
            file,
            indent=2,
        )
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
