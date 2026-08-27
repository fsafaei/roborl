"""Lifecycle step 4: PPO solves CartPole-v1 well beyond random-agent level.

A random agent scores about 22 on CartPole-v1; a working PPO reaches
~300 within 60k steps at CleanRL's default hyperparameters (and touches the
500 cap soon after, though PPO's post-cap returns oscillate, so the gate
stays well below the cap). Takes a few seconds on CPU — marker ``slow``,
run via ``make test-all``.
"""

import numpy as np
import pytest

from roborl.algos.ppo.ppo import PpoConfig, run_ppo


@pytest.mark.slow
def test_ppo_solves_cartpole() -> None:
    summary = run_ppo(
        PpoConfig(
            env_id="CartPole-v1",
            total_timesteps=60_000,
            seed=1,
            device="cpu",
            track=False,
        )
    )
    last_10_mean = float(np.mean(summary.episodic_returns[-10:]))
    assert last_10_mean > 100.0, f"PPO did not solve CartPole: last-10 mean {last_10_mean:.1f}"
