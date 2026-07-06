"""Run one adaptive latent-dimension search round."""

from __future__ import annotations

import argparse
from pathlib import Path

from training import run_latent_dimension_search, run_latent_dimension_search_rounds
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a latent-dimension search round.")
    parser.add_argument("--config", type=Path, default=Path("configs/latent_dim_search.yaml"))
    parser.add_argument("--single-round", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.single_round:
        result = run_latent_dimension_search(config)
        print(f"Wrote results: {result.summary_path}")
        print(f"Predicted optimum: {result.predicted_optimum:.2f}")
        print(f"Proposed next dimensions: {result.proposed_dimensions}")
        print(f"Wrote proposal: {result.proposal_path}")
        return

    result = run_latent_dimension_search_rounds(config)
    for index, round_result in enumerate(result.rounds):
        print(f"Round {index}:")
        print(f"  Wrote results: {round_result.summary_path}")
        print(f"  Predicted optimum: {round_result.predicted_optimum:.2f}")
        print(f"  Proposed next dimensions: {round_result.proposed_dimensions}")
    print(f"Wrote combined results: {result.combined_summary_path}")


if __name__ == "__main__":
    main()
