"""Two same-seed demo runs must be bitwise identical (CPU, tracking off)."""

import pytest

from roborl.demo import DemoConfig, run_demo


@pytest.mark.unit
def test_same_seed_runs_are_identical() -> None:
    config = DemoConfig(seed=5, total_timesteps=500, device="cpu", track=False)
    first = run_demo(config)
    second = run_demo(config)
    assert first.episodic_returns == second.episodic_returns
    assert first.episodic_lengths == second.episodic_lengths
    assert len(first.episodic_returns) > 0


@pytest.mark.unit
def test_different_seeds_differ() -> None:
    first = run_demo(DemoConfig(seed=1, total_timesteps=500, device="cpu", track=False))
    second = run_demo(DemoConfig(seed=2, total_timesteps=500, device="cpu", track=False))
    assert first.episodic_returns != second.episodic_returns
