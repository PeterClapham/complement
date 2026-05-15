"""Save reconstruction grids from trained GON checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from artifacts import save_reconstruction_grid
from data import build_dataset
from models import VariationalGONGenerator
from training import experiment_coordinates
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save reconstruction grids from completed models.")
    parser.add_argument("--config", type=Path, default=Path("configs/test_grid.yaml"))
    parser.add_argument(
        "--all",
        action="store_true",
        help="Save grids for every configured coordinate with a completed model.",
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--beta-inf", type=float, default=None)
    parser.add_argument("--beta-opt", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    coordinates = experiment_coordinates(config)
    selected = coordinates if args.all else [_select_coordinate(coordinates, args)]
    saved = 0
    for coordinate in selected:
        model_path = _model_path(config, coordinate)
        if not model_path.exists():
            continue
        output_path = _run_dir(config, coordinate) / "reconstruction_grid.png"
        _save_coordinate_grid(config, coordinate, model_path, output_path)
        saved += 1
        print(f"Saved reconstruction grid: {output_path}")
    print(f"Saved grids: {saved}")


def _save_coordinate_grid(config: dict[str, Any], coordinate: Any, model_path: Path, output_path: Path) -> None:
    dataset_config = _dataset_config(config, coordinate.dataset_name)
    training_config = _mapping(config.get("training", {}))
    model_config = _mapping(config.get("model", {}))
    device = torch.device(str(training_config.get("device", "cpu")))
    dataset = build_dataset(coordinate.dataset_name, dataset_config, seed=coordinate.seed)
    model = VariationalGONGenerator(
        latent_dim=int(model_config.get("latent_dim", 48)),
        base_channels=int(model_config.get("base_channels", 32)),
        output_channels=int(model_config.get("output_channels", 1)),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    save_reconstruction_grid(
        model=model,
        dataset=dataset,
        output_path=output_path,
        latent_dim=int(model_config.get("latent_dim", 48)),
        beta_inf=coordinate.beta_inf,
        batch_size=int(training_config.get("batch_size", 64)),
        device=device,
    )


def _select_coordinate(coordinates: list[Any], args: argparse.Namespace) -> Any:
    for coordinate in coordinates:
        if args.dataset is not None and coordinate.dataset_name != args.dataset:
            continue
        if args.seed is not None and coordinate.seed != args.seed:
            continue
        if args.beta_inf is not None and coordinate.beta_inf != args.beta_inf:
            continue
        if args.beta_opt is not None and coordinate.beta_opt != args.beta_opt:
            continue
        return coordinate
    raise ValueError("No configured coordinate matched the requested filters.")


def _dataset_config(config: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    for dataset in config.get("datasets", []):
        dataset_config = _mapping(dataset)
        if dataset_config.get("name") == dataset_name:
            return dataset_config
    raise ValueError(f"Dataset not found in config: {dataset_name}")


def _model_path(config: dict[str, Any], coordinate: Any) -> Path:
    return _run_dir(config, coordinate) / "model.pt"


def _run_dir(config: dict[str, Any], coordinate: Any) -> Path:
    experiment_config = _mapping(config.get("experiment", {}))
    return (
        Path(str(experiment_config.get("results_dir", "results")))
        / _safe_name(str(experiment_config.get("name", "variational_gon")))
        / _safe_name(coordinate.dataset_name)
        / f"seed-{coordinate.seed}"
        / f"beta-inf-{_format_float(coordinate.beta_inf)}__beta-opt-{_format_float(coordinate.beta_opt)}"
    )


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("expected a mapping")
    return value


def _safe_name(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return safe.strip("-_") or "run"


def _format_float(value: float) -> str:
    return str(value).replace(".", "p")


if __name__ == "__main__":
    main()
