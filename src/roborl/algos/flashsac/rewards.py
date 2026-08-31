"""Adaptive reward scaling on the collected stream (docs/algos/flashsac.md).

Rewards are stored **raw** in the replay buffer and normalised on the
sampled batch at update time — storing normalised rewards would freeze a
stale scale into the buffer. The statistics are maintained on the collected
stream, per transition: a discounted-return accumulator ``G`` (reset on
``terminated OR truncated`` — *both* flags, unlike the TD target, which
uses ``terminated`` only), its running variance (Chan's parallel update),
and a running max ``G_max_seen``.

The denominator ``max(sqrt(var + 1e-8), G_max_seen / G_max)`` keeps rewards
at unit scale while guaranteeing discounted returns fit inside the critic's
fixed support ``[-G_max, +G_max]``. Turning this off while keeping the
fixed support is not a valid configuration — targets leave the support,
get clamped, and the critic saturates at a boundary atom while looking
superficially healthy.
"""

from __future__ import annotations

import math

import numpy as np
import torch


class RunningMeanVar:
    """Streaming mean and population variance via Chan's parallel update."""

    def __init__(self) -> None:
        """Start with no observations."""
        self.count = 0.0
        self.mean = 0.0
        self.var = 0.0

    def update(self, values: np.ndarray) -> None:
        """Fold a batch of values into the running statistics.

        Args:
            values: 1-D array of new observations (one per parallel env).
        """
        batch = float(values.size)
        batch_mean = float(values.mean())
        batch_var = float(values.var())
        total = self.count + batch
        delta = batch_mean - self.mean
        m2 = self.var * self.count + batch_var * batch + delta**2 * self.count * batch / total
        self.mean += delta * batch / total
        self.var = m2 / total
        self.count = total


class RewardNormalizer:
    """Discounted-return statistics on the collected stream; batch normalisation at update time."""

    def __init__(self, gamma: float, num_envs: int = 1, g_max: float = 5.0) -> None:
        """Build the normaliser.

        Args:
            gamma: Discount factor (the TD target's gamma).
            num_envs: Number of parallel environments the stream comes from.
            g_max: Half-width of the critic's support the returns must fit in.
        """
        self.gamma = gamma
        self.g_max = g_max
        self.accumulator = np.zeros(num_envs, dtype=np.float64)
        self.g_max_seen = 0.0
        self.rms = RunningMeanVar()

    def update(
        self,
        rewards: float | np.ndarray,
        terminated: bool | np.ndarray,
        truncated: bool | np.ndarray,
    ) -> None:
        """Fold one collected transition (per env) into the statistics.

        Args:
            rewards: Raw environment reward(s).
            terminated: True-termination flag(s).
            truncated: Time-limit truncation flag(s) — resets the
                accumulator just like termination does.
        """
        rewards = np.asarray(rewards, dtype=np.float64).reshape(self.accumulator.shape)
        done = np.logical_or(terminated, truncated).astype(np.float64)
        done = done.reshape(self.accumulator.shape)
        self.accumulator = self.gamma * (1.0 - done) * self.accumulator + rewards
        self.g_max_seen = max(self.g_max_seen, float(np.abs(self.accumulator).max()))
        self.rms.update(self.accumulator)

    @property
    def denominator(self) -> float:
        """The current normalisation denominator."""
        return max(math.sqrt(self.rms.var + 1e-8), self.g_max_seen / self.g_max)

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalise a sampled batch of raw rewards.

        Args:
            rewards: Raw rewards from the replay buffer, shape ``(B,)``.

        Returns:
            ``rewards / denominator``, same shape.
        """
        return rewards / self.denominator
