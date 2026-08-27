"""Two same-seed SAC runs on CPU must match exactly (debugging protocol step 4)."""

import pytest

from roborl.algos.sac.sac import SacConfig, run_sac

_CONFIG = SacConfig(
    env_id="Pendulum-v1",
    total_timesteps=350,
    learning_starts=150,  # past warmup, so gradient updates are exercised
    batch_size=32,
    buffer_size=500,
    seed=7,
    device="cpu",
    track=False,
)


@pytest.mark.unit
def test_same_seed_runs_are_identical() -> None:
    first = run_sac(_CONFIG)
    second = run_sac(_CONFIG)
    assert first.episodic_returns == second.episodic_returns
    assert first.episodic_lengths == second.episodic_lengths
    assert len(first.episodic_returns) >= 1
