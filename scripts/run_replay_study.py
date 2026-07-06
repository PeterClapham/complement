"""Aggregate-posterior replay study (prediction P-C).

Train two GONs from the same initialization at a collapse-inducing ``beta_opt``:
one plain (control) and one with interleaved aggregate-posterior replay. If replay
implements the consolidation-protecting rehearsal that the complementary-learning-
systems reading predicts, the replay arm should retain more active latent
coordinates than the control. Rate-distortion accounts of collapse predict replay
makes no difference.

Usage:
    python scripts/run_replay_study.py --config configs/replay_mnist.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from data import build_dataset
from training import run_replay_training
from utils import collapse_dynamics_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare replay vs no-replay training under high beta_opt.")
    parser.add_argument("--config", type=Path, default=Path("configs/replay_mnist.yaml"))
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
    replay_weight: float,
    config: dict[str, Any],
    settings: dict[str, Any],
    train_dataset: Any,
    eval_images: torch.Tensor,
    output_dir: Path,
) -> dict[str, Any]:
    result = run_replay_training(
        model_config=config["model"],
        training_config=config["training"],
        dataset=train_dataset,
        eval_images=eval_images,
        beta_inf=float(settings.get("beta_inf", 1.0)),
        beta_opt=float(settings["beta_opt"]),
        seed=int(settings.get("seed", 0)),
        run_dir=output_dir / label,
        replay_weight=replay_weight,
        replay_every=int(settings.get("replay_every", 1)),
        buffer_capacity=int(settings.get("buffer_capacity", 4096)),
        warmup_batches=int(settings.get("warmup_batches", 8)),
        active_sigma_threshold=float(settings.get("active_sigma_threshold", 0.5)),
        progress=bool(config["training"].get("progress", False)),
    )
    final_row = result.rows[-1]
    print(
        f"[{label}] replay_weight={replay_weight}: final num_active={final_row['num_active']} "
        f"(active_fraction={final_row['active_fraction']:.3f})",
        flush=True,
    )
    return {
        "arm": label,
        "replay_weight": replay_weight,
        "final_num_active": final_row["num_active"],
        "final_active_fraction": final_row["active_fraction"],
        "trajectory": str(result.trajectory_path),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    settings = config["replay_study"]
    results_dir = Path(str(config["experiment"].get("results_dir", "results")))
    seed = int(settings.get("seed", 0))
    dataset_name = str(settings["dataset"])

    entry = _dataset_entry(config, dataset_name)
    eval_dataset = build_dataset(dataset_name, {**entry, "split": "test"}, seed=seed)
    train_dataset = build_dataset(dataset_name, {**entry, "split": "train"}, seed=seed)
    eval_images = _eval_batch(eval_dataset, int(settings.get("eval_batch_size", 512)))

    output_dir = collapse_dynamics_dir(results_dir, str(config["experiment"]["name"])) / "replay"
    output_dir.mkdir(parents=True, exist_ok=True)

    arms = [
        _run_arm("control", 0.0, config, settings, train_dataset, eval_images, output_dir),
        _run_arm(
            "replay", float(settings.get("replay_weight", 1.0)),
            config, settings, train_dataset, eval_images, output_dir,
        ),
    ]

    summary_path = output_dir / "replay_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "beta_opt": float(settings["beta_opt"]),
                "beta_inf": float(settings.get("beta_inf", 1.0)),
                "epochs": int(config["training"]["epochs"]),
                "arms": arms,
            },
            file,
            indent=2,
        )
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
