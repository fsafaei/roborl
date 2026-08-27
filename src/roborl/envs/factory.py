"""Environment factory producing seeded, instrumented Gymnasium envs."""

from __future__ import annotations

from collections.abc import Callable

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
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, video_folder=video_dir)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed + idx)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk
