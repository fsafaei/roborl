"""Episode-aware replay buffer with hindsight goal relabeling — the HER algorithm itself.

HER changes the data, not the algorithm: the SAC update never sees this
file, only the ``ReplayBatch`` it produces (same shape as SAC's). All the
correctness risk of the method concentrates here, so the buffer is pure
NumPy with the reward function *injected* (``compute_reward`` from the
unwrapped env, or a hand-written fake in tests) and imports nothing from
Gymnasium — every relabeling rule in ``docs/algos/her.md`` is testable
against hand-built arrays.

Storage is episode-major (``(episodes, horizon, ...)``). Transitions are
staged while an episode is in flight and become sampleable only after
``commit_episode()``: a half-written episode has no future to relabel from.
Relabeling happens at sample time — a fixed head of every batch gets its
desired goal replaced by an achieved goal from later in the same episode
and its reward recomputed under that goal; storage is never mutated.

Everything is float32 (MuJoCo emits float64; MPS has none) and the stored
done flag is ``terminated`` only — never truncation, and never a relabeled
success. Sampling draws from NumPy's global generator, which
``seed_everything`` seeds, so same-seed runs replay the same minibatch and
relabeling sequence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

GoalStrategy = Literal["future", "final", "episode"]
"""Which achieved goal substitutes the desired one: from later in the episode
(``future``, includes the transition's own successor), the episode's last
achieved goal (``final``), or any achieved goal of the episode (``episode``)."""

RewardFn = Callable[[np.ndarray, np.ndarray, None], np.ndarray]
"""``compute_reward(achieved_goal, desired_goal, info)`` vectorised over a
leading batch dimension; the Gymnasium-Robotics contract, ``info`` unused."""


@dataclass(frozen=True)
class ReplayBatch:
    """One sampled minibatch, as tensors on the training device.

    The same shape SAC's buffer returns, so the update block of the loop is
    line-for-line ``sac.py``. Observations already carry the (possibly
    substituted) goal: ``concat(observation, goal)``.

    Attributes:
        observations: ``(batch, obs_dim + goal_dim)`` float32.
        actions: ``(batch, *action_shape)`` float32.
        next_observations: ``(batch, obs_dim + goal_dim)`` float32, carrying
            the *same* goal as ``observations``.
        rewards: ``(batch,)`` float32 — deliberately 1-D; a stray
            ``(batch, 1)`` broadcasts into a ``(batch, batch)`` TD target.
        dones: ``(batch,)`` float32, 1.0 only on true termination.
    """

    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


@dataclass(frozen=True)
class RelabeledSample:
    """A sampled minibatch as NumPy arrays, plus the relabeling bookkeeping.

    ``HerReplayBuffer.sample_arrays`` returns this; ``sample`` converts the
    first five fields into a ``ReplayBatch``. The remaining fields exist so
    tests (and diagnostics) can check *which* goal each row got and where
    it came from.

    Attributes:
        observations: ``(batch, obs_dim + goal_dim)``.
        actions: ``(batch, *action_shape)``.
        next_observations: ``(batch, obs_dim + goal_dim)``.
        rewards: ``(batch,)`` — recomputed on virtual rows, stored on real.
        dones: ``(batch,)`` — stored ``terminated`` flags, bitwise.
        goals: ``(batch, goal_dim)`` — the goal used on both sides of each row.
        episode_idx: ``(batch,)`` slot each row was drawn from.
        step_idx: ``(batch,)`` step ``t`` within that episode.
        goal_step_idx: ``(batch,)`` step ``f`` whose *next* achieved goal was
            substituted; ``-1`` on real rows.
        virtual: ``(batch,)`` bool — True on relabeled rows.
    """

    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    goals: np.ndarray
    episode_idx: np.ndarray
    step_idx: np.ndarray
    goal_step_idx: np.ndarray
    virtual: np.ndarray

    @property
    def virtual_fraction(self) -> float:
        """Realized share of relabeled rows (``k / (k + 1)`` by construction)."""
        return float(self.virtual.mean()) if self.virtual.size else 0.0

    @property
    def virtual_reward_zero_fraction(self) -> float:
        """Share of relabeled rows whose recomputed reward is 0; NaN with no virtual rows."""
        if not self.virtual.any():
            return float("nan")
        return float(np.mean(self.rewards[self.virtual] == 0.0))


@dataclass(frozen=True)
class StoredEpisode:
    """One committed episode, sliced to its true length (for tests and audits).

    Attributes:
        observations: ``(length, *obs_shape)``.
        achieved_goals: ``(length, *goal_shape)`` — achieved *before* each step.
        desired_goals: ``(length, *goal_shape)``.
        actions: ``(length, *action_shape)``.
        rewards: ``(length,)``.
        next_observations: ``(length, *obs_shape)``.
        next_achieved_goals: ``(length, *goal_shape)`` — achieved *after* each step.
        terminateds: ``(length,)``.
    """

    observations: np.ndarray
    achieved_goals: np.ndarray
    desired_goals: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    next_achieved_goals: np.ndarray
    terminateds: np.ndarray


class HerReplayBuffer:
    """Episode-major replay buffer with sample-time hindsight relabeling.

    Capacity is ``buffer_size // max_episode_steps`` *episodes*; ``len()``
    counts stored transitions. The oldest episode is overwritten once full.
    """

    def __init__(
        self,
        buffer_size: int,
        max_episode_steps: int,
        obs_shape: tuple[int, ...],
        goal_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        compute_reward: RewardFn,
        *,
        her_k: int = 4,
        strategy: GoalStrategy = "future",
        her_enabled: bool = True,
    ) -> None:
        """Preallocate episode-major storage and one episode of staging.

        Args:
            buffer_size: Capacity in *transitions*, as SAC counts it; the
                episode capacity is ``buffer_size // max_episode_steps``.
            max_episode_steps: Longest episode the env can produce (Fetch: 50).
            obs_shape: Shape of the ``observation`` piece of the dict obs.
            goal_shape: Shape of ``achieved_goal`` / ``desired_goal``.
            action_shape: Shape of a single action.
            compute_reward: The env's goal-conditioned reward function,
                vectorised over a leading batch dimension, ``info`` ignored.
            her_k: SB3's ``n_sampled_goal`` — virtual rows are the fraction
                ``k / (k + 1)`` of every batch (4 → 0.8).
            strategy: Goal-selection strategy for the virtual rows.
            her_enabled: When False the buffer stores identically but never
                relabels (``nb_virtual = 0``) — the ablation's no-HER rungs.

        Raises:
            ValueError: On a capacity of zero episodes, a negative ``her_k``,
                or an unknown strategy.
        """
        capacity = buffer_size // max_episode_steps
        if capacity < 1:
            raise ValueError(
                f"buffer_size {buffer_size} holds no complete episode of {max_episode_steps} steps."
            )
        if her_k < 0:
            raise ValueError(f"her_k must be non-negative, got {her_k}.")
        if strategy not in ("future", "final", "episode"):
            raise ValueError(f"Unknown HER strategy {strategy!r}.")
        horizon = max_episode_steps
        self._capacity = capacity
        self._horizon = horizon
        self._compute_reward = compute_reward
        self._strategy: GoalStrategy = strategy
        self._p_her = her_k / (her_k + 1) if her_enabled else 0.0

        def zeros(*shape: int) -> np.ndarray:
            return np.zeros(shape, dtype=np.float32)

        # Committed storage, episode-major.
        self._observations = zeros(capacity, horizon, *obs_shape)
        self._next_observations = zeros(capacity, horizon, *obs_shape)
        self._achieved_goals = zeros(capacity, horizon, *goal_shape)
        self._next_achieved_goals = zeros(capacity, horizon, *goal_shape)
        self._desired_goals = zeros(capacity, horizon, *goal_shape)
        self._actions = zeros(capacity, horizon, *action_shape)
        self._rewards = zeros(capacity, horizon)
        self._terminateds = zeros(capacity, horizon)
        self._ep_len = np.zeros(capacity, dtype=np.int64)  # 0 = slot empty
        self._pos = 0

        # Staging for the episode in flight — never sampleable.
        self._stage_observations = zeros(horizon, *obs_shape)
        self._stage_next_observations = zeros(horizon, *obs_shape)
        self._stage_achieved_goals = zeros(horizon, *goal_shape)
        self._stage_next_achieved_goals = zeros(horizon, *goal_shape)
        self._stage_desired_goals = zeros(horizon, *goal_shape)
        self._stage_actions = zeros(horizon, *action_shape)
        self._stage_rewards = zeros(horizon)
        self._stage_terminateds = zeros(horizon)
        self._staged = 0

    # ------------------------------------------------------------------ sizes

    def __len__(self) -> int:
        """Number of committed transitions (staged steps do not count)."""
        return int(self._ep_len.sum())

    @property
    def n_episodes(self) -> int:
        """Number of committed episodes."""
        return int(np.count_nonzero(self._ep_len))

    @property
    def capacity_episodes(self) -> int:
        """Episode slots available (``buffer_size // max_episode_steps``)."""
        return self._capacity

    @property
    def staged_steps(self) -> int:
        """Transitions of the in-flight episode awaiting ``commit_episode``."""
        return self._staged

    @property
    def virtual_fraction(self) -> float:
        """Configured share of relabeled rows per batch: ``k / (k + 1)``, 0 when HER is off."""
        return self._p_her

    def committed_lengths(self) -> np.ndarray:
        """Lengths of every committed episode (for the fixed-horizon audit in the loop)."""
        return self._ep_len[self._ep_len > 0].copy()

    # ------------------------------------------------------------------ writes

    def add(
        self,
        obs: np.ndarray,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        next_achieved_goal: np.ndarray,
        terminated: bool,
    ) -> None:
        """Stage one transition of the episode in flight, casting to float32.

        All three dict pieces are stored explicitly — ``achieved_goal`` is
        never reconstructed from ``observation``.

        Args:
            obs: ``observation`` the action was taken from.
            achieved_goal: ``achieved_goal`` *before* the step.
            desired_goal: ``desired_goal`` of the step (stored per step, even
                though Fetch keeps it constant within an episode).
            action: Action taken.
            reward: Env reward received (raw scale).
            next_obs: ``observation`` after the step — the *true* final
                observation at episode end, never a reset observation.
            next_achieved_goal: ``achieved_goal`` after the step; this is what
                relabeling serves as substitute goals.
            terminated: True termination only; pass False on truncation.

        Raises:
            ValueError: If the staging area already holds a full episode —
                the caller forgot ``commit_episode``.
        """
        if self._staged == self._horizon:
            raise ValueError(
                f"Episode in flight already has {self._horizon} steps; call commit_episode()."
            )
        i = self._staged
        self._stage_observations[i] = obs
        self._stage_next_observations[i] = next_obs
        self._stage_achieved_goals[i] = achieved_goal
        self._stage_next_achieved_goals[i] = next_achieved_goal
        self._stage_desired_goals[i] = desired_goal
        self._stage_actions[i] = action
        self._stage_rewards[i] = reward
        self._stage_terminateds[i] = float(terminated)
        self._staged = i + 1

    def commit_episode(self) -> int:
        """Move the staged episode into the next slot and make it sampleable.

        Returns:
            The slot index the episode was written to.

        Raises:
            ValueError: If nothing is staged.
        """
        n = self._staged
        if n == 0:
            raise ValueError("commit_episode() called with no staged transitions.")
        slot = self._pos
        self._observations[slot, :n] = self._stage_observations[:n]
        self._next_observations[slot, :n] = self._stage_next_observations[:n]
        self._achieved_goals[slot, :n] = self._stage_achieved_goals[:n]
        self._next_achieved_goals[slot, :n] = self._stage_next_achieved_goals[:n]
        self._desired_goals[slot, :n] = self._stage_desired_goals[:n]
        self._actions[slot, :n] = self._stage_actions[:n]
        self._rewards[slot, :n] = self._stage_rewards[:n]
        self._terminateds[slot, :n] = self._stage_terminateds[:n]
        self._ep_len[slot] = n
        self._pos = (slot + 1) % self._capacity
        self._staged = 0
        return slot

    def stored_episode(self, slot: int) -> StoredEpisode:
        """Return a copy of one committed episode, sliced to its length.

        Args:
            slot: Episode slot index.

        Returns:
            The stored arrays of that episode.

        Raises:
            ValueError: If the slot is empty.
        """
        n = int(self._ep_len[slot])
        if n == 0:
            raise ValueError(f"Slot {slot} holds no committed episode.")
        return StoredEpisode(
            observations=self._observations[slot, :n].copy(),
            achieved_goals=self._achieved_goals[slot, :n].copy(),
            desired_goals=self._desired_goals[slot, :n].copy(),
            actions=self._actions[slot, :n].copy(),
            rewards=self._rewards[slot, :n].copy(),
            next_observations=self._next_observations[slot, :n].copy(),
            next_achieved_goals=self._next_achieved_goals[slot, :n].copy(),
            terminateds=self._terminateds[slot, :n].copy(),
        )

    # ---------------------------------------------------------------- sampling

    def sample_arrays(self, batch_size: int) -> RelabeledSample:
        """Sample a minibatch with hindsight relabeling, as NumPy arrays.

        The algorithm, exactly (``docs/algos/her.md``):

        1. ``ep`` uniform over committed episodes, ``t`` uniform within each.
        2. A fixed count ``nb_virtual = int(k/(k+1) * batch)`` of rows — the
           head of the batch — is relabeled; the rest are real.
        3. Substitute goals are ``next_achieved[ep, f]`` (the goal achieved
           *after* step ``f``) with ``f`` drawn per strategy — ``future``:
           ``randint(t, L)`` (includes the row's own successor, never empty).
        4. Virtual rewards are ``compute_reward(next_achieved[ep, t], goal)``
           — the post-step achieved goal; real rows keep the stored reward.
        5. The same goal is concatenated onto *both* ``obs`` and ``next_obs``.
        6. Dones are the stored ``terminated`` flags — never relabeled success.

        Args:
            batch_size: Number of rows to draw (with replacement).

        Returns:
            The batch plus its relabeling bookkeeping.

        Raises:
            ValueError: If no episode has been committed yet.
        """
        filled = np.flatnonzero(self._ep_len)
        if filled.size == 0:
            raise ValueError(
                "Cannot sample: no committed episode "
                f"({self._staged} staged step(s) are not sampleable until commit_episode())."
            )
        # 1. which transitions
        ep = np.random.choice(filled, size=batch_size)
        lengths = self._ep_len[ep]
        t = np.random.randint(0, lengths)

        # 2. real / virtual split — a fixed head of the batch, not a coin flip
        nb_virtual = int(self._p_her * batch_size)
        virtual = np.zeros(batch_size, dtype=bool)
        virtual[:nb_virtual] = True

        # 3. substitute goals for the virtual rows, from the SAME episode
        goals = self._desired_goals[ep, t].copy()
        goal_step = np.full(batch_size, -1, dtype=np.int64)
        if nb_virtual > 0:
            ep_v, t_v, len_v = ep[:nb_virtual], t[:nb_virtual], lengths[:nb_virtual]
            if self._strategy == "future":
                f = np.random.randint(t_v, len_v)  # f in [t, L-1]; high-exclusive, never empty
            elif self._strategy == "final":
                f = len_v - 1
            else:  # "episode"
                f = np.random.randint(0, len_v)
            assert np.all((f >= 0) & (f < len_v))
            goal_step[:nb_virtual] = f
            goals[:nb_virtual] = self._next_achieved_goals[ep_v, f]

        # 4. rewards — recomputed on the POST-step achieved goal for virtual rows
        rewards = self._rewards[ep, t].copy()
        if nb_virtual > 0:
            recomputed = np.asarray(
                self._compute_reward(
                    self._next_achieved_goals[ep_v, t_v], goals[:nb_virtual], None
                ),
                dtype=np.float32,
            )
            if recomputed.shape != (nb_virtual,):
                raise ValueError(
                    f"compute_reward returned shape {recomputed.shape}, expected {(nb_virtual,)}."
                )
            rewards[:nb_virtual] = recomputed

        # 5. network inputs — the same goal on both sides of the transition
        observations = np.concatenate([self._observations[ep, t], goals], axis=-1)
        next_observations = np.concatenate([self._next_observations[ep, t], goals], axis=-1)

        # 6. dones — stored terminated only
        dones = self._terminateds[ep, t].copy()

        return RelabeledSample(
            observations=observations,
            actions=self._actions[ep, t].copy(),
            next_observations=next_observations,
            rewards=rewards,
            dones=dones,
            goals=goals,
            episode_idx=ep,
            step_idx=t,
            goal_step_idx=goal_step,
            virtual=virtual,
        )

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        """Sample a relabeled minibatch onto ``device`` (see ``sample_arrays``).

        The relabeling statistics of the most recent call are kept in
        ``last_sample_stats`` for telemetry.

        Args:
            batch_size: Number of rows to draw.
            device: Device the returned tensors live on.

        Returns:
            The sampled batch in SAC's ``ReplayBatch`` shape.
        """
        s = self.sample_arrays(batch_size)
        self.last_sample_stats = (s.virtual_fraction, s.virtual_reward_zero_fraction)
        return ReplayBatch(
            observations=torch.as_tensor(s.observations, device=device),
            actions=torch.as_tensor(s.actions, device=device),
            next_observations=torch.as_tensor(s.next_observations, device=device),
            rewards=torch.as_tensor(s.rewards, device=device),
            dones=torch.as_tensor(s.dones, device=device),
        )

    last_sample_stats: tuple[float, float] = (0.0, float("nan"))
    """``(virtual_fraction, virtual_reward_zero_fraction)`` of the last ``sample`` call."""
