"""Continuous-PPO smoke test: a few CPU iterations run, log, and stay finite."""

from pathlib import Path

import numpy as np
import pytest

from roborl.algos.ppo.ppo_continuous import PpoContinuousConfig, run_ppo_continuous


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
