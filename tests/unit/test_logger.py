"""RunLogger: disabled mode is a true no-op; offline mode writes locally."""

from pathlib import Path

import pytest

from roborl.config import ExperimentConfig
from roborl.telemetry.logger import RunLogger


@pytest.mark.unit
def test_disabled_mode_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    logger = RunLogger(ExperimentConfig(track=False), resolved_device="cpu")
    logger.start()
    logger.log({"charts/episodic_return": 1.0}, step=1)
    logger.finish()
    assert logger.url is None
    assert list(tmp_path.iterdir()) == []  # no wandb/ dir, no files at all


@pytest.mark.unit
def test_offline_mode_writes_locally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    logger = RunLogger(ExperimentConfig(exp_name="logtest", track=True), resolved_device="cpu")
    logger.start()
    logger.log({"charts/episodic_return": 1.0}, step=1)
    logger.finish()
    offline_runs = list((tmp_path / "wandb").glob("offline-run-*"))
    assert len(offline_runs) == 1
