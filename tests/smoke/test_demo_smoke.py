"""End-to-end pipeline smoke test: short CPU demo, telemetry disabled."""

from pathlib import Path

import pytest

from roborl.demo import DemoConfig, run_demo


@pytest.mark.smoke
def test_demo_pipeline_200_steps() -> None:
    summary = run_demo(DemoConfig(total_timesteps=200, device="cpu", track=False))
    assert summary.steps == 200
    assert summary.sps > 0
    assert len(summary.episodic_returns) >= 1
    assert "demo finished" in summary.render()


@pytest.mark.smoke
def test_demo_rejects_save_policy_path(tmp_path: Path) -> None:
    from roborl.demo import DemoConfig, run_demo

    with pytest.raises(ValueError, match="save_policy_path"):
        run_demo(DemoConfig(device="cpu", save_policy_path=str(tmp_path / "x.pt")))
