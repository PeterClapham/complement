"""Experiment logging utilities."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from torch import nn


class ExperimentLogger:
    """Create and write the standard files for a single experiment run."""

    def __init__(
        self,
        config: Mapping[str, Any],
        seed: int,
        results_dir: str | Path = "results",
        run_name: str | None = None,
        run_dir: str | Path | None = None,
    ) -> None:
        self.seed = seed
        self.results_dir = Path(results_dir)
        self.run_dir = self._prepare_run_dir(run_dir) if run_dir is not None else self._create_run_dir(run_name)
        self.metrics_path = self.run_dir / "metrics.jsonl"

        self._save_config(config)
        self._save_seed()
        self._save_environment_info()
        self.metrics_path.touch()

    def log_metric(self, step: int, metrics: Mapping[str, Any]) -> None:
        """Append metrics for one training or evaluation step to metrics.jsonl."""
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")

        row = {"step": step, "metrics": _to_jsonable(dict(metrics))}
        with self.metrics_path.open("a", encoding="utf-8") as file:
            json.dump(row, file, sort_keys=True)
            file.write("\n")

    def save_model(self, model: nn.Module, filename: str = "model.pt") -> Path:
        """Save a model state dict inside the run directory."""
        import torch

        model_path = self.run_dir / filename
        torch.save(model.state_dict(), model_path)
        return model_path

    def _create_run_dir(self, run_name: str | None) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        name = _sanitize_run_name(run_name) if run_name else "run"
        base_path = self.results_dir / f"{timestamp}-{name}"

        for index in range(1000):
            run_dir = base_path if index == 0 else Path(f"{base_path}-{index:03d}")
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
                return run_dir
            except FileExistsError:
                continue

        raise RuntimeError(f"Could not create a unique run directory under {self.results_dir}")

    def _prepare_run_dir(self, run_dir: str | Path) -> Path:
        path = Path(run_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _save_config(self, config: Mapping[str, Any]) -> None:
        with (self.run_dir / "config.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(dict(config), file, sort_keys=False)

    def _save_seed(self) -> None:
        with (self.run_dir / "seed.json").open("w", encoding="utf-8") as file:
            json.dump({"seed": self.seed}, file, sort_keys=True)
            file.write("\n")

    def _save_environment_info(self) -> None:
        with (self.run_dir / "environment.json").open("w", encoding="utf-8") as file:
            json.dump(_environment_info(), file, indent=2, sort_keys=True)
            file.write("\n")


def _environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "git_commit": _git_commit_hash(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
    }

    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        }
    except ImportError:
        info["torch"] = None

    return info


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None

    return result.stdout.strip() or None


def _sanitize_run_name(run_name: str) -> str:
    sanitized = "".join(character if character.isalnum() or character in "-_" else "-" for character in run_name)
    sanitized = sanitized.strip("-_")
    return sanitized or "run"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except ValueError:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    return value
