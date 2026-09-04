"""Environment factory producing seeded, instrumented Gymnasium envs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym


def make_env(
    env_id: str,
    seed: int,
    idx: int = 0,
    capture_video: bool = False,
    video_dir: str = "videos",
) -> Callable[[], gym.Env]:
    """Build a thunk that creates one fully-wrapped environment.

    Returns a zero-argument callable (CleanRL style) so it can be handed
    directly to ``gym.vector.SyncVectorEnv`` / ``AsyncVectorEnv`` later; a
    single environment is just the thunk called once.

    The created environment:

    - is always wrapped in ``RecordEpisodeStatistics``, which puts episodic
      return and length into ``info["episode"]`` at episode end — the source
      of ``charts/episodic_return`` and ``charts/episodic_length``;
    - gets ``RecordVideo`` on env 0 only (when ``capture_video``), using
      Gymnasium's default capped-cubic episode schedule, writing to
      ``{video_dir}``;
    - has its RNG seeded with ``seed + idx`` (per-env streams stay distinct in
      vector envs) via an initial ``reset(seed=...)``, and its action and
      observation spaces seeded likewise — an unseeded
      ``action_space.sample()`` is a classic reproducibility leak.

    Callers should use plain ``reset()`` afterwards; the seeded RNG stream
    persists across resets.

    Args:
        env_id: Gymnasium environment id, e.g. ``"CartPole-v1"``.
        seed: Base seed; the env uses ``seed + idx``.
        idx: Index of this env within a vector env (0 for a single env).
        capture_video: Record periodic episode videos (env 0 only).
        video_dir: Directory the video files are written to.

    Returns:
        A thunk returning the wrapped, seeded environment.
    """

    def thunk() -> gym.Env:
        render_kwargs = {"render_mode": "rgb_array"} if capture_video and idx == 0 else {}
        env = _make_registered(env_id, **render_kwargs)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, video_folder=video_dir)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed + idx)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


def _make_registered(env_id: str, **kwargs: Any) -> gym.Env:
    """``gym.make`` that lazily registers Gymnasium-Robotics envs on first use.

    Gymnasium 1.x has no entry-point autoloading: ``gym.make("FetchPush-v4")``
    raises ``NameNotFound`` until ``gymnasium_robotics`` has been imported and
    registered. On that error, import + register once and retry; if the
    package itself is missing, re-raise with the install hint. A genuinely
    unknown id still raises ``NameNotFound`` from the retry.
    """
    try:
        return gym.make(env_id, **kwargs)
    except gym.error.NameNotFound as err:
        try:
            import gymnasium_robotics
        except ImportError:
            raise gym.error.NameNotFound(
                f"{err} If this is a Gymnasium-Robotics environment (Fetch*), install the "
                "'fetch' extra: uv sync --extra fetch"
            ) from err
        gym.register_envs(gymnasium_robotics)
        return gym.make(env_id, **kwargs)
