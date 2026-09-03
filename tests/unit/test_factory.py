"""make_env: wrappers applied, seeding deterministic and per-index distinct."""

import gymnasium as gym
import pytest

from roborl.envs.factory import make_env


@pytest.mark.unit
def test_thunk_returns_wrapped_env() -> None:
    env = make_env("CartPole-v1", seed=1)()
    wrappers = set()
    unwrapped = env
    while isinstance(unwrapped, gym.Wrapper):
        wrappers.add(type(unwrapped).__name__)
        unwrapped = unwrapped.env
    assert "RecordEpisodeStatistics" in wrappers
    assert "RecordVideo" not in wrappers  # capture_video defaults off
    env.close()


@pytest.mark.unit
def test_video_wrapper_only_on_env_zero(tmp_path: object) -> None:
    def wrapper_names(idx: int) -> set[str]:
        env = make_env(
            "CartPole-v1", seed=1, idx=idx, capture_video=True, video_dir=str(tmp_path)
        )()
        names = set()
        unwrapped = env
        while isinstance(unwrapped, gym.Wrapper):
            names.add(type(unwrapped).__name__)
            unwrapped = unwrapped.env
        env.close()
        return names

    assert "RecordVideo" in wrapper_names(0)
    assert "RecordVideo" not in wrapper_names(1)


@pytest.mark.unit
def test_same_seed_same_episode() -> None:
    def rollout() -> list[float]:
        env = make_env("CartPole-v1", seed=42)()
        env.reset()
        rewards = []
        for _ in range(50):
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            rewards.append(float(reward))
            if terminated or truncated:
                env.reset()
        env.close()
        return rewards

    assert rollout() == rollout()


@pytest.mark.unit
def test_index_offsets_seed() -> None:
    first = make_env("CartPole-v1", seed=42, idx=0)()
    second = make_env("CartPole-v1", seed=42, idx=1)()
    assert first.action_space.sample() is not None  # spaces seeded without error
    obs_first, _ = first.reset()
    obs_second, _ = second.reset()
    assert (obs_first != obs_second).any()
    first.close()
    second.close()


@pytest.mark.unit
def test_unknown_env_id_still_raises_name_not_found() -> None:
    with pytest.raises(gym.error.NameNotFound):
        make_env("DefinitelyNotAnEnv-v0", seed=0)()


@pytest.mark.unit
def test_fetch_env_registers_lazily_through_the_factory() -> None:
    pytest.importorskip("gymnasium_robotics", reason="fetch extra not installed")
    env = make_env("FetchReach-v4", seed=0)()
    assert isinstance(env.observation_space, gym.spaces.Dict)
    obs, _ = env.reset()
    assert set(obs) == {"observation", "achieved_goal", "desired_goal"}
    env.close()
