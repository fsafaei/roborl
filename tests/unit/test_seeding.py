"""seed_everything makes the global RNGs reproducible."""

import random

import numpy as np
import pytest
import torch

from roborl.utils.seeding import seed_everything


@pytest.mark.unit
def test_same_seed_same_streams() -> None:
    seed_everything(123)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    seed_everything(123)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert first == second


@pytest.mark.unit
def test_different_seed_different_streams() -> None:
    seed_everything(123)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    seed_everything(124)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert first != second
