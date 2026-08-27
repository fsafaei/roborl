"""Torch device resolution with a CPU-first, GPU-optional policy."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_VALID = ("auto", "cpu", "cuda", "mps")


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a device request to a concrete :class:`torch.device`.

    ``"auto"`` picks the best available accelerator in the order
    ``cuda > mps > cpu``. An explicit request always wins — but requesting an
    accelerator that is not available raises instead of silently falling back,
    so a typo in a launch script cannot quietly turn a GPU run into a CPU run.

    The resolution is logged at INFO level so every run records which device
    it actually used.

    Args:
        device: One of ``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``.

    Returns:
        The resolved torch device.

    Raises:
        ValueError: If ``device`` is not a recognized name, or names an
            accelerator that is unavailable on this machine.
    """
    if device not in _VALID:
        raise ValueError(f"Unknown device {device!r}; expected one of {_VALID}.")

    if device == "auto":
        if torch.cuda.is_available():
            resolved = torch.device("cuda")
        elif torch.backends.mps.is_available():
            resolved = torch.device("mps")
        else:
            resolved = torch.device("cpu")
    else:
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Device 'cuda' was requested but CUDA is not available.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("Device 'mps' was requested but MPS is not available.")
        resolved = torch.device(device)

    logger.info("Resolved device %r -> %s", device, resolved)
    return resolved
