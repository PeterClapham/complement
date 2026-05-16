"""Shared helpers for image reconstruction datasets."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset, TensorDataset, random_split


def limit_dataset(dataset: Dataset, max_samples: object) -> Dataset:
    """Return the first ``max_samples`` items when a positive limit is configured."""
    if max_samples is None:
        return dataset
    limit = int(max_samples)
    if limit <= 0:
        raise ValueError("max_samples must be positive when provided")
    if limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def split_train_validation(
    dataset: Dataset,
    split: str,
    validation_fraction: float,
    seed: int,
) -> Dataset:
    """Build deterministic train/validation subsets from a training dataset."""
    if split == "train" and validation_fraction <= 0.0:
        return dataset
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1) when using train/val splits")

    validation_size = int(round(len(dataset) * validation_fraction))
    train_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )
    if split == "train":
        return train_dataset
    if split == "val":
        return validation_dataset
    raise ValueError(f"Unknown split: {split}")


def maybe_cache_dataset(dataset: Dataset, data_dir: Path, cache_key: str) -> Dataset:
    """Materialize transformed images once and reuse them as a TensorDataset."""
    cache_dir = data_dir / "tensor_cache"
    cache_path = cache_dir / f"{cache_key}.pt"
    if cache_path.exists():
        images = torch.load(cache_path, weights_only=True)
        return TensorDataset(images)

    cache_dir.mkdir(parents=True, exist_ok=True)
    images = torch.stack([image_from_item(dataset[index]) for index in range(len(dataset))])
    temporary_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    torch.save(images, temporary_path)
    temporary_path.replace(cache_path)
    return TensorDataset(images)


def image_from_item(item: object) -> torch.Tensor:
    """Return the image tensor from datasets that may also emit targets."""
    return item[0] if isinstance(item, list | tuple) else item
