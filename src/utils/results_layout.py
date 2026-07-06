"""Canonical paths for research outputs."""

from __future__ import annotations

from pathlib import Path


def beta_grid_dir(results_dir: Path, experiment_name: str) -> Path:
    """Return the root directory for one beta-grid study."""
    return results_dir / "experiments" / "beta_grids" / experiment_name


def beta_grid_run_dir(
    results_dir: Path,
    experiment_name: str,
    dataset_name: str,
    seed: int,
    beta_inf: float,
    beta_opt: float,
) -> Path:
    """Return the directory for one beta-grid coordinate."""
    return (
        beta_grid_dir(results_dir, experiment_name)
        / dataset_name
        / f"seed-{seed}"
        / f"beta-inf-{format_float(beta_inf)}__beta-opt-{format_float(beta_opt)}"
    )


def evaluation_dir(results_dir: Path, study_name: str) -> Path:
    """Return the root directory for derived evaluation artifacts."""
    return results_dir / "evaluations" / study_name


def probe_dir(results_dir: Path, study_name: str, dataset_name: str, seed: int) -> Path:
    """Return the root directory for one probe study/seed."""
    return results_dir / "experiments" / "probes" / study_name / dataset_name / f"seed-{seed}"


def latent_search_dir(results_dir: Path, study_name: str) -> Path:
    """Return the root directory for one latent-dimension search."""
    return results_dir / "experiments" / "latent_dim_search" / study_name


def latent_search_run_dir(
    results_dir: Path,
    study_name: str,
    round_index: int,
    latent_dim: int,
    seed: int,
) -> Path:
    """Return one latent-dimension search run directory."""
    return (
        latent_search_dir(results_dir, study_name)
        / f"round-{round_index:02d}"
        / "runs"
        / f"latent-{latent_dim:03d}"
        / f"seed-{seed}"
    )


def format_float(value: float) -> str:
    """Format float values for stable filesystem names."""
    return str(value).replace(".", "p")
