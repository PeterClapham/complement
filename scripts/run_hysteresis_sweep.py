"""Hysteresis sweep (prediction P-B).

Drive one model continuously up a ``beta_opt`` ladder and then back down, reusing
the evolving weights at every stage. Record the active-coordinate fraction at each
stage. An equilibrium view predicts the up-leg and down-leg values coincide at
matched ``beta_opt`` (a single-valued curve). A consolidation view predicts a
hysteresis loop: collapse forms readily on the way up and resists reversal on the
way down.

Usage:
    python scripts/run_hysteresis_sweep.py --config configs/hysteresis_mnist.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from data import build_dataset
from models import build_model
from training import continue_training
from utils import collapse_dynamics_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous up/down beta_opt hysteresis sweep.")
    parser.add_argument("--config", type=Path, default=Path("configs/hysteresis_mnist.yaml"))
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    settings = config["hysteresis"]
    results_dir = Path(str(config["experiment"].get("results_dir", "results")))
    seed = int(settings.get("seed", 0))
    dataset_name = str(settings["dataset"])
    beta_inf = float(settings.get("beta_inf", 1.0))
    ladder = [float(value) for value in settings["beta_opt_ladder"]]
    active_sigma_threshold = float(settings.get("active_sigma_threshold", 0.5))

    entry = _dataset_entry(config, dataset_name)
    eval_dataset = build_dataset(dataset_name, {**entry, "split": "test"}, seed=seed)
    train_dataset = build_dataset(dataset_name, {**entry, "split": "train"}, seed=seed)
    eval_images = _eval_batch(eval_dataset, int(settings.get("eval_batch_size", 512)))

    output_dir = collapse_dynamics_dir(results_dir, str(config["experiment"]["name"])) / "hysteresis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Continuous protocol: ascend the ladder, then descend without the repeated top.
    up_stages = [("up", beta_opt) for beta_opt in ladder]
    down_stages = [("down", beta_opt) for beta_opt in reversed(ladder[:-1])]
    stages = up_stages + down_stages

    state = build_model(config["model"]).state_dict()
    rows: list[dict[str, Any]] = []
    for index, (leg, beta_opt) in enumerate(stages):
        result = continue_training(
            source_state=state,
            model_config=config["model"],
            training_config=config["training"],
            dataset=train_dataset,
            eval_images=eval_images,
            beta_inf=beta_inf,
            beta_opt=beta_opt,
            seed=seed,
            run_dir=output_dir / f"stage-{index:02d}-{leg}-bopt-{str(beta_opt).replace('.', 'p')}",
            active_sigma_threshold=active_sigma_threshold,
            progress=bool(config["training"].get("progress", False)),
        )
        state = result.final_state
        final_row = result.rows[-1]
        rows.append(
            {
                "stage": index,
                "leg": leg,
                "beta_opt": beta_opt,
                "active_fraction": final_row["active_fraction"],
                "num_active": final_row["num_active"],
                "active_mask": final_row["active_mask"],
            }
        )
        print(f"stage {index:02d} [{leg}] beta_opt={beta_opt}: num_active={final_row['num_active']}", flush=True)

    sweep_path = output_dir / "hysteresis_sweep.csv"
    with sweep_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Pair up-leg and down-leg active fractions at matched beta_opt for the loop.
    up_by_beta = {row["beta_opt"]: row["num_active"] for row in rows if row["leg"] == "up"}
    down_by_beta = {row["beta_opt"]: row["num_active"] for row in rows if row["leg"] == "down"}
    loop = [
        {
            "beta_opt": beta_opt,
            "num_active_up": up_by_beta.get(beta_opt),
            "num_active_down": down_by_beta.get(beta_opt),
            "hysteresis_gap": (
                None
                if down_by_beta.get(beta_opt) is None
                else down_by_beta[beta_opt] - up_by_beta[beta_opt]
            ),
        }
        for beta_opt in ladder
    ]
    with (output_dir / "hysteresis_loop.json").open("w", encoding="utf-8") as file:
        json.dump({"beta_inf": beta_inf, "ladder": ladder, "loop": loop}, file, indent=2)
    print(f"Wrote {sweep_path}", flush=True)


if __name__ == "__main__":
    main()
