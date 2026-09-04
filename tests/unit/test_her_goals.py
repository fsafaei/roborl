"""Goal-env plumbing: flattening, contract guard, success extraction, the real Fetch contract."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from roborl.algos.her.goals import (
    GOAL_OBS_KEYS,
    check_goal_env,
    episode_success,
    flatten_goal_obs,
    goal_space_dims,
)


class FakeGoalEnv(gym.Env):
    """A minimal goal env: 1-D point moving toward a goal on a line, T = 5."""

    def __init__(self, obs_dim: int = 2, goal_dim: int = 1, vectorised: bool = True) -> None:
        self.metadata = {"render_modes": []}
        box = lambda n: gym.spaces.Box(-np.inf, np.inf, (n,), dtype=np.float64)  # noqa: E731
        self.observation_space = gym.spaces.Dict(
            {
                "observation": box(obs_dim),
                "achieved_goal": box(goal_dim),
                "desired_goal": box(goal_dim),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (goal_dim,), dtype=np.float32)
        self._vectorised = vectorised
        self._t = 0

    def compute_reward(self, achieved: np.ndarray, desired: np.ndarray, info: Any) -> Any:
        distance = np.linalg.norm(achieved - desired, axis=-1)
        reward = -(distance > 0.5).astype(np.float32)
        return reward if self._vectorised else float(reward.reshape(-1)[0])

    def _obs(self) -> dict[str, np.ndarray]:
        return {
            "observation": np.array([self._t, -self._t], dtype=np.float64),
            "achieved_goal": np.array([float(self._t)]),
            "desired_goal": np.array([2.0]),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        super().reset(seed=seed)
        self._t = 0
        return self._obs(), {}

    def step(self, action: np.ndarray) -> Any:
        self._t += 1
        obs = self._obs()
        reward = float(
            self.compute_reward(obs["achieved_goal"][None], obs["desired_goal"][None], None)[0]
        )
        return obs, reward, False, self._t >= 5, {"is_success": np.float32(reward == 0.0)}


@pytest.mark.unit
class TestFlattening:
    def test_concat_observation_and_desired_goal_float32(self) -> None:
        obs = {
            "observation": np.arange(4, dtype=np.float64),
            "achieved_goal": np.array([99.0, 99.0]),
            "desired_goal": np.array([7.0, 8.0]),
        }
        flat = flatten_goal_obs(obs)
        assert flat.dtype == np.float32
        assert flat.shape == (6,)
        np.testing.assert_array_equal(flat, [0, 1, 2, 3, 7, 8])
        assert 99.0 not in flat  # achieved_goal never enters the network input

    def test_goal_space_dims(self) -> None:
        assert goal_space_dims(FakeGoalEnv(obs_dim=10, goal_dim=3).observation_space) == (10, 3)
        with pytest.raises(ValueError, match="Dict"):
            goal_space_dims(gym.spaces.Box(-1, 1, (3,)))


@pytest.mark.unit
class TestContractGuard:
    def test_returns_the_unwrapped_reward_oracle(self) -> None:
        env = gym.wrappers.RecordEpisodeStatistics(FakeGoalEnv())
        compute_reward = check_goal_env(env)
        out = compute_reward(np.zeros((3, 1), np.float32), np.ones((3, 1), np.float32), None)
        np.testing.assert_array_equal(out, [-1.0, -1.0, -1.0])

    def test_rejects_flat_observation_space(self) -> None:
        with pytest.raises(ValueError, match="Dict"):
            check_goal_env(gym.make("Pendulum-v1"))

    def test_rejects_wrong_keys(self) -> None:
        env = FakeGoalEnv()
        assert isinstance(env.observation_space, gym.spaces.Dict)
        spaces = dict(env.observation_space.spaces)
        spaces["extra"] = spaces.pop("achieved_goal")
        env.observation_space = gym.spaces.Dict(spaces)
        with pytest.raises(ValueError, match=str(GOAL_OBS_KEYS[1])):
            check_goal_env(env)

    def test_rejects_mismatched_goal_shapes(self) -> None:
        env = FakeGoalEnv()
        assert isinstance(env.observation_space, gym.spaces.Dict)
        spaces = dict(env.observation_space.spaces)
        spaces["achieved_goal"] = gym.spaces.Box(-1, 1, (2,))
        env.observation_space = gym.spaces.Dict(spaces)
        with pytest.raises(ValueError, match="share a shape"):
            check_goal_env(env)

    def test_rejects_unvectorised_compute_reward(self) -> None:
        with pytest.raises(ValueError, match="vectorised"):
            check_goal_env(FakeGoalEnv(vectorised=False))

    def test_rejects_missing_compute_reward(self) -> None:
        env = FakeGoalEnv()
        del FakeGoalEnv.compute_reward
        try:
            with pytest.raises(ValueError, match="compute_reward"):
                check_goal_env(env)
        finally:
            FakeGoalEnv.compute_reward = _compute_reward_backup  # type: ignore[method-assign]

    def test_rejects_discrete_actions(self) -> None:
        env = FakeGoalEnv()
        env.action_space = gym.spaces.Discrete(3)
        with pytest.raises(ValueError, match="Box"):
            check_goal_env(env)


_compute_reward_backup = FakeGoalEnv.compute_reward


@pytest.mark.unit
class TestSuccess:
    def test_reads_final_step_flag_as_float(self) -> None:
        assert episode_success({"is_success": np.float32(1.0)}) == 1.0
        assert episode_success({"is_success": 0.0}) == 0.0
        assert episode_success({"is_success": True}) == 1.0

    def test_missing_flag_raises(self) -> None:
        with pytest.raises(KeyError, match="is_success"):
            episode_success({"episode": {}})

    def test_success_mid_episode_then_leaving_reads_zero(self) -> None:
        # The fake env passes through the goal (t = 2) and keeps going to t = 5:
        # per-step flags are [0, 1, 0, 0, 0]; episode success is the FINAL one.
        env = FakeGoalEnv()
        env.reset()
        flags = []
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(np.zeros(1, np.float32))
            flags.append(episode_success(info))
            done = terminated or truncated
        assert flags == [0.0, 1.0, 0.0, 0.0, 0.0]
        assert episode_success(info) == 0.0
        assert max(flags) == 1.0  # "any step" would wrongly say success


@pytest.fixture(scope="module")
def fetch_env() -> Any:
    pytest.importorskip("gymnasium_robotics", reason="fetch extra not installed")
    from roborl.envs.factory import make_env

    env = make_env("FetchReach-v4", seed=3)()
    yield env
    env.close()


@pytest.mark.unit
class TestRealFetchContract:
    """The installed Fetch env honours every assumption the buffer makes (skips w/o the extra)."""

    def test_spaces_and_flattening(self, fetch_env: gym.Env) -> None:
        env = fetch_env
        assert goal_space_dims(env.observation_space) == (10, 3)
        obs, _ = env.reset()
        flat = flatten_goal_obs(obs)
        assert flat.shape == (13,) and flat.dtype == np.float32
        assert obs["observation"].dtype == np.float64  # the cast is ours to do

    def test_reward_contract_and_never_terminates(self, fetch_env: gym.Env) -> None:
        env = fetch_env
        compute_reward = check_goal_env(env)
        obs, _ = env.reset()
        assert compute_reward.__self__ is env.unwrapped  # type: ignore[attr-defined]
        assert not hasattr(env, "compute_reward")  # wrappers do not forward it (pitfall 18)
        terminated_ever = False
        steps = 0
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            steps += 1
            terminated_ever |= bool(terminated)
            # Stored reward == oracle(post-step achieved goal, desired goal).
            recomputed = compute_reward(obs["achieved_goal"][None], obs["desired_goal"][None], None)
            assert recomputed.shape == (1,) and float(recomputed[0]) == float(reward)
            assert float(reward) in (0.0, -1.0)
            assert "is_success" in info
            done = terminated or truncated
        assert steps == 50 and not terminated_ever

    def test_threshold_is_inclusive_five_centimetres(self, fetch_env: gym.Env) -> None:
        env = fetch_env
        compute_reward = check_goal_env(env)
        goal = np.zeros((1, 3), np.float32)
        assert compute_reward(goal + np.array([0.049, 0, 0]), goal, None)[0] == 0.0
        assert compute_reward(goal + np.array([0.051, 0, 0]), goal, None)[0] == -1.0
