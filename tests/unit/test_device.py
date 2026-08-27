"""resolve_device: auto-resolution order and explicit-override behavior."""

import pytest
import torch

from roborl.utils.device import resolve_device


def _availability(monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_prefers_cuda_then_mps_then_cpu(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, expected: str
) -> None:
    _availability(monkeypatch, cuda, mps)
    assert resolve_device("auto").type == expected


@pytest.mark.unit
def test_explicit_cpu_always_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    _availability(monkeypatch, cuda=True, mps=True)
    assert resolve_device("cpu").type == "cpu"


@pytest.mark.unit
@pytest.mark.parametrize("requested", ["cuda", "mps"])
def test_unavailable_accelerator_raises(monkeypatch: pytest.MonkeyPatch, requested: str) -> None:
    _availability(monkeypatch, cuda=False, mps=False)
    with pytest.raises(ValueError, match=requested):
        resolve_device(requested)


@pytest.mark.unit
def test_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu")
