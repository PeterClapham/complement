"""Dataset factory functions."""

from __future__ import annotations

from torch.utils.data import Dataset

from data.mnist import build_mnist_dataset
from data.synthetic import SyntheticBinaryImageDataset


def build_dataset(name: str, config: dict, seed: int) -> Dataset:
    """Build a dataset from a dataset config entry."""
    if name == "synthetic_binary":
        return SyntheticBinaryImageDataset(
            num_samples=int(config.get("num_samples", 16)),
            image_size=int(config.get("image_size", 32)),
            channels=int(config.get("channels", 1)),
            seed=int(config.get("seed", seed)),
        )
    if name == "mnist":
        return build_mnist_dataset(config, seed=seed)

    raise ValueError(f"Unknown dataset: {name}")
