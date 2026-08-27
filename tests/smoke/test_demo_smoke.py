"""End-to-end pipeline smoke test: short CPU demo, telemetry disabled."""

import pytest

from roborl.demo import DemoConfig, run_demo


@pytest.mark.smoke
def test_demo_pipeline_200_steps() -> None:
    summary = run_demo(DemoConfig(total_timesteps=200, device="cpu", track=False))
    assert summary.steps == 200
    assert summary.sps > 0
    assert len(summary.episodic_returns) >= 1
    assert "demo finished" in summary.render()
