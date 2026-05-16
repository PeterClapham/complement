"""smallNORB dataset construction."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.common import limit_dataset, maybe_cache_dataset, split_train_validation

_URLS = {
    "train": "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/"
    "smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz",
    "test": "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/"
    "smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz",
}


class SmallNORBTensorDataset(Dataset):
    """Tensor-backed smallNORB split containing both stereo views."""

    def __init__(self, images: torch.Tensor, transform: transforms.Compose) -> None:
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.fromarray(self.images[index].numpy(), mode="L")
        return self.transform(image)


def build_smallnorb_dataset(config: dict, seed: int) -> Dataset:
    """Build a reproducible smallNORB split for reconstruction experiments."""
    data_dir = config.get("data_dir")
    if data_dir is None:
        raise ValueError("smallNORB config requires data_dir")

    split = str(config.get("split", "train"))
    image_size = int(config.get("image_size", 32))
    download = bool(config.get("download", False))
    validation_fraction = float(config.get("validation_fraction", 0.0))
    max_samples = config.get("max_samples")
    cache_tensors = bool(config.get("cache_tensors", False))
    root = Path(data_dir)

    source_split = "test" if split == "test" else "train"
    dataset: Dataset = SmallNORBTensorDataset(
        _load_images(root, source_split, download),
        transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()]),
    )
    if cache_tensors:
        dataset = maybe_cache_dataset(dataset, root, f"smallnorb_{source_split}_{image_size}px")
    if split != "test":
        dataset = split_train_validation(dataset, split, validation_fraction, seed)
    return limit_dataset(dataset, max_samples)


def _load_images(root: Path, split: str, download: bool) -> torch.Tensor:
    processed_path = root / "processed" / f"{split}_images.pt"
    if processed_path.exists():
        return torch.load(processed_path, weights_only=True)
    if not download:
        raise RuntimeError("smallNORB not found. Set download: true to fetch it.")

    raw_path = root / "raw" / Path(_URLS[split]).name
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        urlretrieve(_URLS[split], raw_path)
    images = _parse_dat_gzip(raw_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(images, processed_path)
    return images


def _parse_dat_gzip(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as file:
        file.read(4)
        ndim = struct.unpack("<i", file.read(4))[0]
        dimensions = struct.unpack(f"<{ndim}i", file.read(4 * ndim))
        num_examples, num_views, height, width = dimensions
        data = np.frombuffer(file.read(), dtype=np.uint8)
    images = data.reshape(num_examples * num_views, height, width).copy()
    return torch.from_numpy(images)
