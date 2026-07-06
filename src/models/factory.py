"""Model construction helpers."""

from __future__ import annotations

from typing import Any

import torch

from models.variational_gon import GroupNormVariationalGONGenerator, VariationalGONGenerator


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    """Build a configured model variant.

    ``variational_gon`` preserves the historical BatchNorm model used by
    existing studies. ``variational_gon_groupnorm`` is the default for new
    experiments.
    """
    name = str(config.get("name", "variational_gon_groupnorm"))
    kwargs = {
        "latent_dim": int(config.get("latent_dim", 48)),
        "base_channels": int(config.get("base_channels", 32)),
        "output_channels": int(config.get("output_channels", 1)),
    }
    if name == "variational_gon":
        return VariationalGONGenerator(**kwargs)
    if name == "variational_gon_groupnorm":
        return GroupNormVariationalGONGenerator(
            **kwargs,
            num_groups=int(config.get("num_groups", 8)),
        )
    raise ValueError(f"Unknown model: {name}")
