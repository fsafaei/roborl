"""Two same-seed PPO runs on CPU must match exactly (debugging protocol step 4)."""

import pytest

from roborl.algos.ppo.ppo import PpoConfig, run_ppo

_CONFIG = PpoConfig(
    env_id="CartPole-v1",
    total_timesteps=1024,  # two full iterations at the default batch geometry
    seed=7,
    device="cpu",
    track=False,
)


@pytest.mark.unit
def test_same_seed_runs_are_identical() -> None:
    first = run_ppo(_CONFIG)
    second = run_ppo(_CONFIG)
    assert first.episodic_returns == second.episodic_returns
    assert first.episodic_lengths == second.episodic_lengths
    assert first.episode_end_steps == second.episode_end_steps
    assert len(first.episodic_returns) >= 1
