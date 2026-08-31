"""Lifecycle step 4: FlashSAC solves Pendulum-v1 well beyond random-agent level.

A random agent scores about -1200 on Pendulum-v1; a working FlashSAC
reaches better than -300 within 20k steps — the same bar ``sac.py``
cleared. Takes a few minutes on CPU — marker ``slow``, run via
``make test-all``.
"""

import numpy as np
import pytest

from roborl.algos.flashsac.flashsac import FlashSacConfig, run_flashsac


@pytest.mark.slow
def test_flashsac_solves_pendulum() -> None:
    summary = run_flashsac(
        FlashSacConfig(
            env_id="Pendulum-v1",
            total_timesteps=20_000,
            learning_starts=1_000,  # deviation from the paper's 10k: short-budget sanity only
            seed=1,
            device="cpu",
            track=False,
        )
    )
    last_10_mean = float(np.mean(summary.episodic_returns[-10:]))
    assert last_10_mean > -300.0, (
        f"FlashSAC did not solve Pendulum: last-10 mean {last_10_mean:.1f}"
    )
