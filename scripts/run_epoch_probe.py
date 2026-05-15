"""Run an exploratory train/validation probe for choosing an epoch budget."""

from __future__ import annotations

import argparse
from pathlib import Path

from training import run_epoch_probe
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an epoch-selection probe.")
    parser.add_argument("--config", type=Path, default=Path("configs/epoch_probe.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--beta-inf", type=float, default=1.0)
    parser.add_argument("--beta-opt", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_epoch_probe(
        config=load_config(args.config),
        seed=args.seed,
        dataset_name=args.dataset,
        beta_inf=args.beta_inf,
        beta_opt=args.beta_opt,
    )
    print(
        f"Probe complete: best_epoch={result.best_epoch} "
        f"epochs_completed={result.epochs_completed} "
        f"best_val_elbo={result.best_validation_elbo:.4f} "
        f"stopped_early={result.stopped_early} "
        f"run_dir={result.run_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
