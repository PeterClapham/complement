"""Move legacy result artifacts into the structured results layout."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path("results")


def main() -> None:
    _move_named_studies()
    _move_evaluations()
    _move_timestamped_runs()
    _move_loose_files()


def _move_named_studies() -> None:
    for name in ["variational_gon", "variational_gon_seed0_batch128_80ep", "smallnorb_seed0_batch128_80ep"]:
        _move(ROOT / name, ROOT / "experiments" / "beta_grids" / name)
    for name in ["latent_dim_search_mnist"]:
        _move(ROOT / name, ROOT / "experiments" / "latent_dim_search" / name)
    for name in [
        "batch128_timing",
        "input_pipeline_benchmark_base",
        "input_pipeline_benchmark_base_seq",
        "input_pipeline_benchmark_workers",
        "input_pipeline_benchmark_workers_seq",
        "mnist_full_epoch_nobatchckpt_1778815937",
        "mnist_full_epoch_speed_1778809591",
        "mnist_speed",
        "mnist_speed_fresh_1778809570",
        "smallnorb_speed",
    ]:
        _move(ROOT / name, ROOT / "experiments" / "benchmarks" / name)
    for name in ["progress_smoke", "smoke_variational_gon", "local_gate_timing_1778816553"]:
        _move(ROOT / name, ROOT / "scratch" / name)


def _move_evaluations() -> None:
    mapping = {
        "mnist_seed0": "beta_grids/variational_gon/final",
        "mnist_seed0_80ep": "beta_grids/variational_gon_seed0_batch128_80ep/epoch_sweep",
        "mnist_seed0_150ep": "beta_grids/variational_gon_seed0_batch128_80ep/final-150ep",
        "smallnorb_seed0_80ep": "beta_grids/smallnorb_seed0_batch128_80ep/final-080ep",
        "smallnorb_seed0_200ep": "beta_grids/smallnorb_seed0_batch128_80ep/final-200ep",
        "smoke_saved_models": "scratch/smoke_saved_models",
    }
    for old_name, new_suffix in mapping.items():
        _move(ROOT / "evaluation" / old_name, ROOT / "evaluations" / new_suffix)
    source = ROOT / "evaluation"
    if source.exists() and not any(source.iterdir()):
        source.rmdir()


def _move_timestamped_runs() -> None:
    pattern = re.compile(
        r"^\d{8}-\d{6}-\d{6}-(?P<study>.+?)(?:-mnist|-synthetic_binary)-seed(?P<seed>\d+)$"
    )
    latent_pattern = re.compile(
        r"^\d{8}-\d{6}-\d{6}-latent_dim_search_mnist_round(?P<round>\d+)_latent(?P<latent>\d+)-mnist-seed(?P<seed>\d+)$"
    )
    for path in list(ROOT.iterdir()):
        if not path.is_dir():
            continue
        latent_match = latent_pattern.match(path.name)
        if latent_match:
            destination = (
                ROOT
                / "experiments"
                / "latent_dim_search"
                / "latent_dim_search_mnist"
                / f"round-{int(latent_match['round']):02d}"
                / "runs"
                / f"latent-{int(latent_match['latent']):03d}"
                / f"seed-{latent_match['seed']}"
            )
            _move(path, destination)
            continue
        match = pattern.match(path.name)
        if match:
            study = match["study"]
            category = "reconstruction" if "reconstruction" in study else "epoch_budget"
            _move(
                path,
                ROOT / "experiments" / "probes" / category / study / f"seed-{match['seed']}" / path.name[:22],
            )
            continue
        if path.name.startswith("20260515-"):
            _move(path, ROOT / "scratch" / "ad_hoc" / path.name)
        elif path.name.startswith("input_pipeline_workers"):
            _move(path, ROOT / "experiments" / "benchmarks" / path.name)


def _move_loose_files() -> None:
    for path in list(ROOT.glob("*.log")):
        _move(path, ROOT / "logs" / path.name)
    for path in list(ROOT.glob("*.csv")):
        category = "benchmarks" if "benchmark" in path.name else "summaries"
        _move(path, ROOT / "experiments" / category / path.name)


def _move(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    shutil.move(str(source), str(destination))
    print(f"{source} -> {destination}")


if __name__ == "__main__":
    main()
