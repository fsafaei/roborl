"""The continuous-PPO wrapper stack matches CleanRL's, in CleanRL's order."""

import gymnasium as gym
import numpy as np
import pytest

from roborl.algos.ppo.ppo_continuous import make_continuous_env


@pytest.mark.unit
def test_wrapper_order_matches_cleanrl() -> None:
    # Outermost first. RecordEpisodeStatistics below the reward wrappers is
    # the guarantee that episodic returns stay in raw env units (detail C8's
    # scaling must not leak into charts/episodic_return).
    env = make_continuous_env("Pendulum-v1", seed=1, gamma=0.99)()
    expected = [
        gym.wrappers.TransformReward,
        gym.wrappers.NormalizeReward,
        gym.wrappers.TransformObservation,
        gym.wrappers.NormalizeObservation,
        gym.wrappers.ClipAction,
        gym.wrappers.RecordEpisodeStatistics,
        gym.wrappers.FlattenObservation,
    ]
    layer: gym.Env = env
    for wrapper_cls in expected:
        assert isinstance(layer, wrapper_cls), f"expected {wrapper_cls.__name__}, got {layer}"
        assert isinstance(layer, gym.Wrapper)
        layer = layer.env
    env.close()


@pytest.mark.unit
def test_observations_normalized_and_clipped_rewards_clipped() -> None:
    env = make_continuous_env("Pendulum-v1", seed=1, gamma=0.99)()
    obs, _ = env.reset()
    for _ in range(50):
        # Out-of-bounds action: ClipAction must make this legal (detail C5).
        obs, reward, terminated, truncated, _ = env.step(np.array([37.0]))
        assert np.all(np.abs(obs) <= 10.0)
        assert -10.0 <= float(reward) <= 10.0
        assert not (terminated or truncated)
    env.close()


@pytest.mark.unit
def test_episode_stats_report_raw_returns() -> None:
    # Run one full Pendulum episode; the recorded return must be the raw
    # (untransformed) reward sum: always negative, bounded by the env's
    # worst case, and far outside the +/-10-clipped normalized scale.
    env = make_continuous_env("Pendulum-v1", seed=1, gamma=0.99)()
    env.reset()
    episode_return = None
    for _ in range(200):
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            episode_return = float(info["episode"]["r"])
            break
    assert episode_return is not None, "Pendulum episode did not finish in 200 steps"
    assert -2000.0 < episode_return < -100.0
    env.close()
