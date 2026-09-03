"""Lifecycle step 4: HER+SAC solves FetchReach-v4 (deterministic success >= 0.9 at budget).

Reach is the easy Fetch task — it saturates near 1.0 within a few thousand
steps — so this is a plumbing gate, not a benchmark. Uses the full recipe
(512^3 nets, batch 2048) at the 100k budget of the verification plan; takes
a while on CPU — marker ``slow``, run via ``make test-all``.
"""

import pytest

pytest.importorskip("gymnasium_robotics", reason="fetch extra not installed")

from roborl.algos.her.her_sac import HerSacConfig, run_her_sac


@pytest.mark.slow
def test_her_sac_solves_fetch_reach() -> None:
    summary = run_her_sac(
        HerSacConfig(
            env_id="FetchReach-v4",
            total_timesteps=100_000,
            seed=1,
            device="cpu",
            track=False,
        )
    )
    final_eval = summary.eval_success_rates[-1]
    assert final_eval >= 0.9, f"HER+SAC did not solve FetchReach: eval success {final_eval:.2f}"
