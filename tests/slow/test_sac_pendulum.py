"""Lifecycle step 4: SAC solves Pendulum-v1 well beyond random-agent level.

A random agent scores about -1200 on Pendulum-v1; a working SAC reaches
better than -300 within 20k steps. Takes a few minutes on CPU — marker
``slow``, run via ``make test-all``.
"""

import numpy as np
import pytest

from roborl.algos.sac.sac import SacConfig, run_sac


@pytest.mark.slow
def test_sac_solves_pendulum() -> None:
    summary = run_sac(
        SacConfig(
            env_id="Pendulum-v1",
            total_timesteps=20_000,
            learning_starts=1_000,  # deviation from CleanRL's 5k: short-budget sanity only
            seed=1,
            device="cpu",
            track=False,
        )
    )
    last_10_mean = float(np.mean(summary.episodic_returns[-10:]))
    assert last_10_mean > -300.0, f"SAC did not solve Pendulum: last-10 mean {last_10_mean:.1f}"
