"""PPO smoke test: a few CPU iterations run, log, and stay finite."""

from pathlib import Path

import numpy as np
import pytest

from roborl.algos.ppo.ppo import PpoConfig, run_ppo


@pytest.mark.smoke
def test_ppo_four_iterations_cpu(tmp_path: Path) -> None:
    summary = run_ppo(
        PpoConfig(
            env_id="CartPole-v1",
            total_timesteps=2048,  # 4 iterations x 512 batch: annealing, epochs, GAE all exercised
            device="cpu",
            track=False,
            save_episodes=True,
            episode_dir=str(tmp_path),
        )
    )
    assert summary.steps == 2048
    assert summary.sps > 0
    assert len(summary.episodic_returns) >= 1
    assert np.isfinite(summary.episodic_returns).all()  # a NaN policy dies here
    assert summary.episode_end_steps == sorted(summary.episode_end_steps)
    assert "ppo finished" in summary.render()
    assert summary.episodes_csv is not None
    csv_text = Path(summary.episodes_csv).read_text()
    assert csv_text.startswith("run_id,global_step,episodic_return")
    assert len(csv_text.strip().splitlines()) == len(summary.episodic_returns) + 1


@pytest.mark.smoke
def test_ppo_rejects_continuous_action_space() -> None:
    with pytest.raises(ValueError, match="discrete"):
        run_ppo(PpoConfig(env_id="Pendulum-v1", total_timesteps=512, device="cpu", track=False))
