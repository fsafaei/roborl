"""Lifecycle step 4: continuous PPO learns Pendulum-v1 well beyond random level.

A random agent scores about -1230 on Pendulum-v1. At CleanRL's MuJoCo
defaults (gamma 0.99, one env) PPO learns Pendulum notoriously slowly —
this is a property of the reference hyperparameters, not a bug: CleanRL's
own ``ppo_continuous_action.py``, run with the same env-API adaptations,
reaches a last-10 mean of -952 at 500k steps on this seed, while this
implementation reaches -762 (calibration runs, 2026-08-28). The gate sits
between untrained (~-1100) and our observed result. Takes ~90 seconds on
CPU — marker ``slow``, run via ``make test-all``.
"""

import numpy as np
import pytest

from roborl.algos.ppo.ppo_continuous import PpoContinuousConfig, run_ppo_continuous


@pytest.mark.slow
def test_ppo_continuous_learns_pendulum() -> None:
    summary = run_ppo_continuous(
        PpoContinuousConfig(
            env_id="Pendulum-v1",
            total_timesteps=500_000,
            seed=1,
            device="cpu",
            track=False,
        )
    )
    last_10_mean = float(np.mean(summary.episodic_returns[-10:]))
    assert last_10_mean > -1000.0, (
        f"PPO continuous did not learn Pendulum: last-10 mean {last_10_mean:.1f}"
    )
