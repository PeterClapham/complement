"""MNIST dataset construction."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import datasets, transforms


def build_mnist_dataset(config: dict, seed: int) -> Dataset:
    """Build a reproducible MNIST split for image reconstruction experiments."""
    data_dir = config.get("data_dir")
    if data_dir is None:
        raise ValueError("MNIST config requires data_dir")

    split = str(config.get("split", "train"))
    image_size = int(config.get("image_size", 32))
    download = bool(config.get("download", False))
    binarize = bool(config.get("binarize", False))
    validation_fraction = float(config.get("validation_fraction", 0.0))
    max_samples = config.get("max_samples")

    transform_steps = [transforms.Resize(image_size), transforms.ToTensor()]
    if binarize:
        transform_steps.append(transforms.Lambda(lambda image: (image > 0.5).float()))

    if split == "test":
        test_dataset = datasets.MNIST(
            root=Path(data_dir),
            train=False,
            download=download,
            transform=transforms.Compose(transform_steps),
        )
        return _limit_dataset(test_dataset, max_samples)

    full_train = datasets.MNIST(
        root=Path(data_dir),
        train=True,
        download=download,
        transform=transforms.Compose(transform_steps),
    )
    if split == "train" and validation_fraction <= 0.0:
        return _limit_dataset(full_train, max_samples)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1) when using train/val splits")

    validation_size = int(round(len(full_train) * validation_fraction))
    train_size = len(full_train) - validation_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = random_split(
        full_train,
        [train_size, validation_size],
        generator=generator,
    )

    if split == "train":
        return _limit_dataset(train_dataset, max_samples)
    if split == "val":
        return _limit_dataset(validation_dataset, max_samples)
    raise ValueError(f"Unknown MNIST split: {split}")


def _limit_dataset(dataset: Dataset, max_samples: object) -> Dataset:
    if max_samples is None:
        return dataset

    limit = int(max_samples)
    if limit <= 0:
        raise ValueError("max_samples must be positive when provided")
    if limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))
