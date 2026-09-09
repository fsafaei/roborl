"""Continuous-PPO smoke test: a few CPU iterations run, log, and stay finite."""

from pathlib import Path

import numpy as np
import pytest

from roborl.algos.ppo.ppo_continuous import PpoContinuousConfig, run_ppo_continuous
from roborl.io import load_policy


@pytest.mark.smoke
def test_ppo_continuous_four_iterations_cpu(tmp_path: Path) -> None:
    summary = run_ppo_continuous(
        PpoContinuousConfig(
            env_id="Pendulum-v1",
            total_timesteps=1024,  # 4 iterations x 256 batch: annealing, epochs, GAE all exercised
            num_envs=2,
            num_steps=128,
            device="cpu",
            track=False,
            save_episodes=True,
            episode_dir=str(tmp_path),
            save_policy_path=str(tmp_path / "ppo.pt"),
        )
    )
    assert summary.steps == 1024
    assert summary.sps > 0
    assert len(summary.episodic_returns) >= 1
    assert np.isfinite(summary.episodic_returns).all()  # a NaN policy dies here
    # Raw Pendulum units — reward normalization must not leak into episode stats.
    assert all(-2000.0 < r < 0.0 for r in summary.episodic_returns)
    assert summary.episode_end_steps == sorted(summary.episode_end_steps)
    assert summary.episodes_csv is not None
    csv_text = Path(summary.episodes_csv).read_text()
    assert csv_text.startswith("run_id,global_step,episodic_return")
    assert len(csv_text.strip().splitlines()) == len(summary.episodic_returns) + 1
    policy = load_policy(tmp_path / "ppo.pt")
    assert summary.policy_path == str(tmp_path / "ppo.pt")
    assert summary.policy_sha256 == policy.sha256
    assert "policy checkpoint:" in summary.render()
    assert policy.spec.algo == "ppo_continuous"
    assert policy.metadata["global_step"] == summary.steps
    assert policy.metadata["env_id"] == "Pendulum-v1"
    assert policy.metadata["config"]["seed"] == policy.metadata["seed"]
    action = policy.act(np.zeros(policy.spec.obs_dim, dtype=np.float32))
    assert action.shape == (policy.spec.act_dim,) and np.all(np.isfinite(action))
    assert np.all(action >= policy.spec.action_low) and np.all(action <= policy.spec.action_high)
    # Pendulum's true bounds; ClipAction itself advertises an unbounded action space.
    assert policy.spec.action_low == (-2.0,) and policy.spec.action_high == (2.0,)
    assert policy.obs_normalizer is not None and policy.obs_normalizer.clip == 10.0
    assert policy.obs_normalizer.mean.shape == (policy.spec.obs_dim,)
    assert np.all(np.isfinite(policy.obs_normalizer.var)) and np.all(policy.obs_normalizer.var > 0)


@pytest.mark.smoke
def test_ppo_continuous_rejects_discrete_action_space() -> None:
    with pytest.raises(ValueError, match="continuous"):
        run_ppo_continuous(
            PpoContinuousConfig(
                env_id="CartPole-v1",
                total_timesteps=2048,
                device="cpu",
                track=False,
            )
        )
