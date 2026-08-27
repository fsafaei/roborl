"""Global RNG seeding for reproducible experiments."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, *, torch_deterministic: bool = True) -> None:
    """Seed every random number generator the training stack uses.

    Seeds Python's ``random``, NumPy's legacy global generator, and torch
    (CPU, plus CUDA and MPS when present). Environments are seeded separately
    through :func:`roborl.envs.factory.make_env`, because Gymnasium keeps
    per-environment RNG state.

    What this does *not* guarantee: bitwise reproducibility on GPUs. Some CUDA
    kernels are nondeterministic by design (e.g. atomics-based scatter/gather),
    and MPS makes weaker determinism promises than CPU. CPU runs with the same
    seed are expected to be exactly reproducible; GPU runs are only
    statistically comparable. See ``docs/debugging-rl.md`` for how to use
    same-seed runs as a debugging instrument.

    Args:
        seed: The seed shared by all generators.
        torch_deterministic: When True (default), sets
            ``torch.backends.cudnn.deterministic`` and disables cuDNN
            benchmarking, trading a little speed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices and MPS when present
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = torch_deterministic
        torch.backends.cudnn.benchmark = not torch_deterministic
