import random

import numpy as np
import torch

from utils import set_seed


def test_set_seed_makes_random_generators_reproducible():
    set_seed(123)
    first = (random.random(), np.random.rand(), torch.rand(1).item())

    set_seed(123)
    second = (random.random(), np.random.rand(), torch.rand(1).item())

    assert first == second
