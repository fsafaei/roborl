"""Two same-seed HER+SAC runs on CPU must match exactly, with evaluation on.

Covers pitfalls 14 (all sampling through NumPy's global RNG) and 17 (the
evaluation pass touches neither the training stream nor the RNG): the
training curves, the success flags, the eval results, the final global RNG
state, and a post-run relabeling draw all have to coincide.
"""

import hashlib

import numpy as np
import pytest

pytest.importorskip("gymnasium_robotics", reason="fetch extra not installed")

from roborl.algos.her.buffer import HerReplayBuffer
from roborl.algos.her.her_sac import HerSacConfig, run_her_sac

_CONFIG = HerSacConfig(
    env_id="FetchReach-v4",
    total_timesteps=250,
    learning_starts=100,  # past warmup, so gradient updates (and relabeling) are exercised
    batch_size=32,
    buffer_size=500,
    net_arch=(32, 32),
    eval_interval=100,
    eval_episodes=1,
    seed=7,
    device="cpu",
    track=False,
)


def _run() -> tuple[list[float], list[int], list[float], list[int], list[float], str, str]:
    digest: dict[str, str] = {}

    def audit(rb: HerReplayBuffer) -> None:
        state = np.random.get_state()[1]
        digest["rng"] = hashlib.sha256(np.asarray(state).tobytes()).hexdigest()
        sample = rb.sample_arrays(256)
        digest["goals"] = hashlib.sha256(sample.goals.tobytes()).hexdigest()

    summary = run_her_sac(_CONFIG, buffer_audit=audit)
    return (
        summary.episodic_returns,
        summary.episodic_lengths,
        summary.episodic_successes,
        summary.eval_steps,
        summary.eval_success_rates,
        digest["rng"],
        digest["goals"],
    )


@pytest.mark.unit
def test_same_seed_runs_are_identical() -> None:
    first = _run()
    second = _run()
    assert first == second
    assert len(first[0]) >= 1
