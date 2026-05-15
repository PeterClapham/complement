"""Dataset loading and preprocessing code."""

from data.factory import build_dataset
from data.mnist import build_mnist_dataset
from data.synthetic import SyntheticBinaryImageDataset

__all__ = ["SyntheticBinaryImageDataset", "build_dataset", "build_mnist_dataset"]
