"""Placeholder test so pytest's exit code is meaningful before Phase 2.

Deleted when the first real unit tests land.
"""

import pytest

import roborl


@pytest.mark.unit
def test_package_imports_and_has_version() -> None:
    assert isinstance(roborl.__version__, str)
    assert roborl.__version__
