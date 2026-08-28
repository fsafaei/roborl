"""Two same-seed continuous-PPO runs on CPU must match exactly."""

import pytest

from roborl.algos.ppo.ppo_continuous import PpoContinuousConfig, run_ppo_continuous

# Shrunk batch geometry so the test runs in seconds; long enough that each
# env finishes at least one 200-step Pendulum episode.
_CONFIG = PpoContinuousConfig(
    env_id="Pendulum-v1",
    total_timesteps=512,
    num_envs=2,
    num_steps=64,
    num_minibatches=4,
    update_epochs=2,
    seed=7,
    device="cpu",
    track=False,
)


@pytest.mark.unit
def test_same_seed_runs_are_identical() -> None:
    first = run_ppo_continuous(_CONFIG)
    second = run_ppo_continuous(_CONFIG)
    assert first.episodic_returns == second.episodic_returns
    assert first.episodic_lengths == second.episodic_lengths
    assert first.episode_end_steps == second.episode_end_steps
    assert len(first.episodic_returns) >= 1
