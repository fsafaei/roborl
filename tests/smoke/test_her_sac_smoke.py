"""HER+SAC smoke test: a few hundred FetchReach steps on CPU run, log, eval, and audit clean.

Skips cleanly without the ``fetch`` extra. Beyond "it runs", the buffer
audit checks the Fetch structural facts the design leans on: fixed horizon,
no terminations, no autoreset leakage into ``next_achieved``, and stored
rewards reproducible from the env's own ``compute_reward``.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium_robotics", reason="fetch extra not installed")

from roborl.algos.her.buffer import HerReplayBuffer
from roborl.algos.her.her_sac import HerSacConfig, run_her_sac


@pytest.mark.smoke
def test_her_sac_300_steps_cpu(tmp_path: Path) -> None:
    audited: dict[str, object] = {}

    def audit(rb: HerReplayBuffer) -> None:
        lengths = rb.committed_lengths()
        audited["n_episodes"] = rb.n_episodes
        assert lengths.size >= 5 and np.all(lengths == 50)  # Fetch: fixed horizon
        for slot in range(rb.n_episodes):
            episode = rb.stored_episode(slot)
            assert np.all(episode.terminateds == 0.0)  # Fetch never terminates
            # Recompute-consistency against the REAL env oracle (pitfall 3 / API drift).
            recomputed = rb._compute_reward(
                episode.next_achieved_goals, episode.desired_goals, None
            )
            np.testing.assert_array_equal(np.asarray(recomputed, np.float32), episode.rewards)
        # No autoreset leakage: the stored final next_achieved of episode i is the
        # true final state, not the reset state that opens episode i+1.
        for slot in range(rb.n_episodes - 1):
            last_next_ag = rb.stored_episode(slot).next_achieved_goals[-1]
            first_ag = rb.stored_episode(slot + 1).achieved_goals[0]
            assert not np.allclose(last_next_ag, first_ag)
        # Relabeling on the real data: 80% virtual, some own-successor zeros.
        np.random.seed(0)
        sample = rb.sample_arrays(1000)
        assert int(sample.virtual.sum()) == 800
        assert 0.0 < sample.virtual_reward_zero_fraction < 1.0

    summary = run_her_sac(
        HerSacConfig(
            env_id="FetchReach-v4",
            total_timesteps=300,
            learning_starts=100,  # past warmup: critic, actor, and alpha all update
            batch_size=32,
            buffer_size=500,  # 10 episodes
            net_arch=(32, 32),
            eval_interval=150,
            eval_episodes=1,
            device="cpu",
            track=False,
            save_episodes=True,
            episode_dir=str(tmp_path),
        ),
        buffer_audit=audit,
    )
    assert summary.steps == 300
    assert summary.sps > 0
    assert len(summary.episodic_returns) == 6  # 300 // 50
    assert audited["n_episodes"] == 6
    assert np.isfinite(summary.episodic_returns).all()  # a NaN policy dies here
    assert set(summary.episodic_successes) <= {0.0, 1.0}
    assert summary.eval_steps == [150, 300]
    assert len(summary.eval_success_rates) == 2
    assert "her-sac finished" in summary.render()
    assert summary.episodes_csv is not None and summary.success_csv is not None
    assert Path(summary.episodes_csv).read_text().startswith("run_id,global_step,episodic_return")
    success_lines = Path(summary.success_csv).read_text().splitlines()
    assert success_lines[0] == "run_id,global_step,episodic_success"
    assert len(success_lines) == 7
    assert success_lines[1].endswith(",50,0") or success_lines[1].endswith(",50,1")
