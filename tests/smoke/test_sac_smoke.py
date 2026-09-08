"""SAC smoke test: a few hundred CPU steps run, log, and stay finite."""

from pathlib import Path

import numpy as np
import pytest

from roborl.algos.sac.sac import SacConfig, run_sac
from roborl.io import load_policy


@pytest.mark.smoke
def test_sac_400_steps_cpu(tmp_path: Path) -> None:
    summary = run_sac(
        SacConfig(
            env_id="Pendulum-v1",
            total_timesteps=400,
            learning_starts=150,  # past warmup: critic, actor, and alpha all update
            batch_size=32,
            buffer_size=500,
            device="cpu",
            track=False,
            save_episodes=True,
            episode_dir=str(tmp_path),
            save_policy_path=str(tmp_path / "sac.pt"),
        )
    )
    assert summary.steps == 400
    assert summary.sps > 0
    assert len(summary.episodic_returns) >= 1
    assert np.isfinite(summary.episodic_returns).all()  # a NaN policy dies here
    assert "sac finished" in summary.render()
    assert summary.episodes_csv is not None
    csv_text = Path(summary.episodes_csv).read_text()
    assert csv_text.startswith("run_id,global_step,episodic_return")
    policy = load_policy(tmp_path / "sac.pt")
    assert summary.policy_path == str(tmp_path / "sac.pt")
    assert summary.policy_sha256 == policy.sha256
    assert "policy checkpoint:" in summary.render()
    assert policy.spec.algo == "sac"
    assert policy.metadata["global_step"] == summary.steps
    assert policy.metadata["env_id"] == "Pendulum-v1"
    assert policy.metadata["config"]["seed"] == policy.metadata["seed"]
    action = policy.act(np.zeros(policy.spec.obs_dim, dtype=np.float32))
    assert action.shape == (policy.spec.act_dim,) and np.all(np.isfinite(action))
    assert np.all(action >= policy.spec.action_low) and np.all(action <= policy.spec.action_high)
    assert policy.spec.action_low == (-2.0,) and policy.spec.action_high == (2.0,)
