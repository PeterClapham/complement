"""CelebA dataset construction."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets, transforms

from data.common import limit_dataset, maybe_cache_dataset


def build_celeba_dataset(config: dict, seed: int) -> Dataset:
    """Build a grayscale CelebA split for image reconstruction experiments."""
    del seed
    data_dir = config.get("data_dir")
    if data_dir is None:
        raise ValueError("CelebA config requires data_dir")

    split = str(config.get("split", "train"))
    split_name = "valid" if split == "val" else split
    if split_name not in {"train", "valid", "test", "all"}:
        raise ValueError(f"Unknown CelebA split: {split}")
    image_size = int(config.get("image_size", 32))
    download = bool(config.get("download", False))
    max_samples = config.get("max_samples")
    cache_tensors = bool(config.get("cache_tensors", False))

    dataset = datasets.CelebA(
        root=Path(data_dir),
        split=split_name,
        download=download,
        transform=transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.CenterCrop(178),
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        ),
    )
    if cache_tensors:
        dataset = maybe_cache_dataset(dataset, Path(data_dir), f"celeba_{split_name}_{image_size}px")
    return limit_dataset(dataset, max_samples)
