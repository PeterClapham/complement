"""Run one adaptive latent-dimension search round."""

from __future__ import annotations

import argparse
from pathlib import Path

from training import run_latent_dimension_search_rounds
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a latent-dimension search round.")
    parser.add_argument("--config", type=Path, default=Path("configs/latent_dim_search.yaml"))
    return parser.parse_args()


def main() -> None:
    result = run_latent_dimension_search_rounds(load_config(parse_args().config))
    for index, round_result in enumerate(result.rounds):
        print(f"Round {index}:")
        print(f"  Wrote results: {round_result.summary_path}")
        print(f"  Predicted optimum: {round_result.predicted_optimum:.2f}")
        print(f"  Proposed next dimensions: {round_result.proposed_dimensions}")
    print(f"Wrote combined results: {result.combined_summary_path}")


if __name__ == "__main__":
    main()
