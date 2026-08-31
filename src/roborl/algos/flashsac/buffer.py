"""Uniform replay buffer, local to FlashSAC (rule of three — ADR 0003; second copy after SAC's).

A preallocated NumPy ring buffer instead of a framework dependency: the
buffer is itself curriculum. Rewards are stored **raw**: adaptive reward
scaling is applied to the sampled batch at update time (``rewards.py``), so
the buffer never freezes a stale scale. The two details that matter live
here:

- **Everything is float32.** MuJoCo observations arrive as float64, and MPS
  has no float64 — the cast happens once, at this boundary.
- **The stored done flag is ``terminated`` only.** Targets bootstrap through
  time-limit truncation (the episode didn't end, our budget did), never
  through true termination. Callers must therefore pass the *true* final
  observation as ``next_obs`` at episode end — under Gymnasium 1.x that
  means a non-autoresetting env (see ``flashsac.py``), because an autoreset
  step would hand you the next episode's reset observation instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ReplayBatch:
    """One sampled minibatch, as tensors on the training device.

    Attributes:
        observations: ``(batch, *obs_shape)`` float32.
        actions: ``(batch, *action_shape)`` float32.
        next_observations: ``(batch, *obs_shape)`` float32.
        rewards: ``(batch,)`` float32 — deliberately 1-D; a stray
            ``(batch, 1)`` here broadcasts into a ``(batch, batch)`` TD
            target downstream.
        dones: ``(batch,)`` float32, 1.0 only on true termination.
    """

    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Fixed-capacity uniform-sampling transition store.

    Overwrites oldest transitions once full. Sampling uses NumPy's global
    generator, which ``seed_everything`` seeds — same-seed runs replay the
    same minibatch sequence.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
    ) -> None:
        """Preallocate storage.

        Args:
            capacity: Maximum number of transitions held.
            obs_shape: Shape of a single observation.
            action_shape: Shape of a single action.
        """
        self._observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._next_observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._pos = 0
        self._full = False

    def __len__(self) -> int:
        """Number of transitions currently stored."""
        return self._capacity if self._full else self._pos

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        """Store one transition, casting to float32.

        Args:
            obs: Observation the action was taken from.
            action: Action taken.
            reward: Reward received.
            next_obs: Next observation — the *true* final observation at
                episode end, never a reset observation.
            terminated: True termination only; pass False on truncation so
                the target keeps bootstrapping.
        """
        self._observations[self._pos] = obs
        self._next_observations[self._pos] = next_obs
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._dones[self._pos] = float(terminated)
        self._pos += 1
        if self._pos == self._capacity:
            self._pos = 0
            self._full = True

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        """Sample a uniform minibatch (with replacement) onto ``device``.

        Args:
            batch_size: Number of transitions to draw.
            device: Device the returned tensors live on.

        Returns:
            The sampled batch.

        Raises:
            ValueError: If the buffer is empty.
        """
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        idx = np.random.randint(0, len(self), size=batch_size)
        return ReplayBatch(
            observations=torch.as_tensor(self._observations[idx], device=device),
            actions=torch.as_tensor(self._actions[idx], device=device),
            next_observations=torch.as_tensor(self._next_observations[idx], device=device),
            rewards=torch.as_tensor(self._rewards[idx], device=device),
            dones=torch.as_tensor(self._dones[idx], device=device),
        )
