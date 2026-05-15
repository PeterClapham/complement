"""Training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from training import run_gon_experiment
from utils import DEFAULT_BETA_VALUES, iter_beta_grid, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a GON training experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to a YAML experiment config.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for this run.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name for this run.")
    parser.add_argument("--beta-inf", type=float, default=None, help="ELBO_inf beta for latent inference.")
    parser.add_argument("--beta-opt", type=float, default=None, help="ELBO_opt beta for parameter updates.")
    parser.add_argument(
        "--print-grid",
        action="store_true",
        help="Print configured beta_inf/beta_opt combinations and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    beta_values = _beta_values(config)

    if args.print_grid:
        for index, beta_config in enumerate(iter_beta_grid(beta_values)):
            print(
                f"{index:02d}: beta_inf={beta_config['beta_inf']} "
                f"beta_opt={beta_config['beta_opt']}"
            )
        return

    seed = args.seed if args.seed is not None else _first_seed(config)
    dataset_name = args.dataset if args.dataset is not None else _first_dataset_name(config)
    beta_inf = args.beta_inf if args.beta_inf is not None else beta_values[0]
    beta_opt = args.beta_opt if args.beta_opt is not None else beta_values[0]

    result = run_gon_experiment(
        config=config,
        seed=seed,
        dataset_name=dataset_name,
        beta_inf=beta_inf,
        beta_opt=beta_opt,
    )

    if result.model_path is not None:
        print(f"Saved model state: {result.model_path}")
    print(f"Created run directory: {result.run_dir}")
    print(f"Training steps: {result.num_steps}")
    print(f"Parameter update norm: {result.parameter_update_norm:.6f}")


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("experiment config must be a mapping")
    return value


def _beta_values(config: dict[str, Any]) -> list[float]:
    betas_config = _mapping(config.get("betas", {}))
    values = betas_config.get("values", DEFAULT_BETA_VALUES)
    return [float(value) for value in values]


def _first_seed(config: dict[str, Any]) -> int:
    seeds = config.get("seeds", [0])
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list")
    return int(seeds[0])


def _first_dataset_name(config: dict[str, Any]) -> str:
    datasets = config.get("datasets", [{"name": "dataset"}])
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a non-empty list")
    dataset = _mapping(datasets[0])
    return str(dataset.get("name", "dataset"))


if __name__ == "__main__":
    main()
