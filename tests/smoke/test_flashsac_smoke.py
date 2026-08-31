"""FlashSAC smoke test: a few hundred CPU steps run, log, and stay finite."""

from pathlib import Path

import numpy as np
import pytest

from roborl.algos.flashsac.flashsac import FlashSacConfig, run_flashsac


@pytest.mark.smoke
def test_flashsac_400_steps_cpu(tmp_path: Path) -> None:
    summary = run_flashsac(
        FlashSacConfig(
            env_id="Pendulum-v1",
            total_timesteps=400,
            learning_starts=150,  # past warmup: critic, actor, and alpha all update
            batch_size=32,
            buffer_size=500,
            device="cpu",
            track=False,
            save_episodes=True,
            episode_dir=str(tmp_path),
        )
    )
    assert summary.steps == 400
    assert summary.sps > 0
    assert len(summary.episodic_returns) >= 1
    assert np.isfinite(summary.episodic_returns).all()  # a NaN policy dies here
    assert "flashsac finished" in summary.render()
    assert summary.episodes_csv is not None
    csv_text = Path(summary.episodes_csv).read_text()
    assert csv_text.startswith("run_id,global_step,episodic_return")


@pytest.mark.smoke
@pytest.mark.parametrize(
    "flags",
    [
        # rung 2: architecture only — scalar critic, no RMSNorm/weight norm,
        # SAC-style exploration and entropy target
        {
            "use_rmsnorm": False,
            "use_distributional": False,
            "use_weight_norm": False,
            "use_flash_exploration": False,
            "alpha_init": 1.0,
        },
        # rung 5: everything except the exploration changes
        {"use_flash_exploration": False, "alpha_init": 1.0},
    ],
    ids=["rung2-arch-only", "rung5-no-flash-exploration"],
)
def test_ablation_rungs_run_and_stay_finite(tmp_path: Path, flags: dict) -> None:
    summary = run_flashsac(
        FlashSacConfig(
            env_id="Pendulum-v1",
            total_timesteps=400,
            learning_starts=150,
            batch_size=32,
            buffer_size=500,
            device="cpu",
            track=False,
            **flags,
        )
    )
    assert summary.steps == 400
    assert len(summary.episodic_returns) >= 1
    assert np.isfinite(summary.episodic_returns).all()
