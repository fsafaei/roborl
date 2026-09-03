"""Goal-conditioned environment plumbing: the dict-observation contract, flattening, success.

Gymnasium-Robotics goal envs return ``Dict`` observations with exactly three
pieces — ``observation``, ``achieved_goal``, ``desired_goal`` — expose a
vectorised ``compute_reward(achieved_goal, desired_goal, info)`` on the
*unwrapped* env, and report ``info["is_success"]`` every step. These helpers
pin that contract once, so the training loops stay diffable against
``sac.py`` / ``flashsac.py``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from roborl.algos.her.buffer import RewardFn

GOAL_OBS_KEYS: tuple[str, str, str] = ("observation", "achieved_goal", "desired_goal")
"""The three pieces of a goal-env observation, in the order the contract lists them."""


def flatten_goal_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Build the network input ``concat(observation, desired_goal)`` as float32.

    ``achieved_goal`` is deliberately absent — it is state the buffer stores
    for relabeling, never something the policy or critics see.

    Args:
        obs: One goal-env observation dict.

    Returns:
        A 1-D float32 array of length ``obs_dim + goal_dim``.
    """
    return np.concatenate([obs["observation"], obs["desired_goal"]]).astype(np.float32)


def episode_success(info: dict[str, Any]) -> float:
    """Read the step's ``is_success`` flag as a float (0.0 or 1.0).

    Success of an *episode* is this flag on the episode's **final** step —
    the HER-literature convention — not whether the goal was ever touched
    and not a per-step average. Callers apply this at episode end only.

    Args:
        info: The ``info`` dict returned by ``env.step``.

    Returns:
        ``float(info["is_success"])``.

    Raises:
        KeyError: If the env did not report ``is_success``.
    """
    if "is_success" not in info:
        raise KeyError("Goal env did not report info['is_success']; is this a goal env?")
    return float(info["is_success"])


def goal_space_dims(space: gym.Space) -> tuple[int, int]:
    """Return ``(obs_dim, goal_dim)`` of a goal-env ``Dict`` observation space.

    Args:
        space: The env's observation space.

    Returns:
        Flat sizes of the ``observation`` and ``desired_goal`` pieces.

    Raises:
        ValueError: If the space does not satisfy the goal-env contract.
    """
    _require_goal_dict(space)
    assert isinstance(space, gym.spaces.Dict)
    obs_shape = space["observation"].shape
    goal_shape = space["desired_goal"].shape
    assert obs_shape is not None and goal_shape is not None
    return int(np.prod(obs_shape)), int(np.prod(goal_shape))


def check_goal_env(env: gym.Env) -> RewardFn:
    """Verify the goal-env contract and return the relabeling reward oracle.

    A cheap structural guard for loop start-up (the behavioural checks —
    reward values, ``is_success``, never-terminates — live in tests):

    - ``Dict`` observation space with exactly the three goal keys, each a
      ``Box``; ``achieved_goal`` and ``desired_goal`` share a shape;
    - ``Box`` action space;
    - ``env.unwrapped.compute_reward`` exists, accepts a ``(B, G)`` batch
      with ``info=None`` and returns shape ``(B,)``.

    Args:
        env: The (wrapped) goal environment.

    Returns:
        ``env.unwrapped.compute_reward``, bound — wrappers do not forward it
        reliably, so the loop takes it from here once.

    Raises:
        ValueError: On any contract violation, naming the piece that failed.
    """
    space = env.observation_space
    _require_goal_dict(space)
    assert isinstance(space, gym.spaces.Dict)
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError(f"HER needs a continuous (Box) action space; got {env.action_space}.")
    compute_reward = getattr(env.unwrapped, "compute_reward", None)
    if compute_reward is None:
        raise ValueError("env.unwrapped has no compute_reward(achieved, desired, info) method.")
    goal_shape = space["desired_goal"].shape
    assert goal_shape is not None
    batch = 4
    probe = np.zeros((batch, *goal_shape), dtype=np.float32)
    out = np.asarray(compute_reward(probe, probe, None))
    if out.shape != (batch,):
        raise ValueError(
            f"compute_reward must be vectorised over a leading batch dimension: a {probe.shape} "
            f"batch returned shape {out.shape}, expected {(batch,)}."
        )
    result: RewardFn = compute_reward
    return result


def _require_goal_dict(space: gym.Space) -> None:
    """Raise unless ``space`` is a goal-env ``Dict`` of three matching ``Box`` pieces."""
    if not isinstance(space, gym.spaces.Dict):
        raise ValueError(f"HER needs a Dict observation space; got {space}.")
    keys = tuple(sorted(space.spaces))
    if keys != tuple(sorted(GOAL_OBS_KEYS)):
        raise ValueError(f"Goal env must expose exactly {GOAL_OBS_KEYS}; got {keys}.")
    for key in GOAL_OBS_KEYS:
        if not isinstance(space[key], gym.spaces.Box):
            raise ValueError(f"Observation piece {key!r} must be a Box; got {space[key]}.")
    if space["achieved_goal"].shape != space["desired_goal"].shape:
        raise ValueError(
            "achieved_goal and desired_goal must share a shape; got "
            f"{space['achieved_goal'].shape} vs {space['desired_goal'].shape}."
        )
