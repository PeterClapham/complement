"""Experiment grids for beta hyper-parameters."""

from __future__ import annotations

from collections.abc import Iterable


DEFAULT_BETA_VALUES = (0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0)


def iter_beta_grid(beta_values: Iterable[float] = DEFAULT_BETA_VALUES) -> list[dict[str, float]]:
    """Return all beta_inf/beta_opt combinations for the phase-diagram sweep."""
    values = [float(value) for value in beta_values]
    return [{"beta_inf": beta_inf, "beta_opt": beta_opt} for beta_inf in values for beta_opt in values]
