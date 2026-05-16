"""Dataset factory functions."""

from __future__ import annotations

from torch.utils.data import Dataset

from data.celeba import build_celeba_dataset
from data.dsprites import build_dsprites_dataset
from data.mnist import build_mnist_dataset
from data.smallnorb import build_smallnorb_dataset
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
    if name == "celeba":
        return build_celeba_dataset(config, seed=seed)
    if name == "smallnorb":
        return build_smallnorb_dataset(config, seed=seed)
    if name == "dsprites":
        return build_dsprites_dataset(config, seed=seed)
    if name == "dsprites_noisy":
        return build_dsprites_dataset(config, seed=seed, noisy=True)

    raise ValueError(f"Unknown dataset: {name}")
