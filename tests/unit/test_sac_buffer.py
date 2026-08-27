"""Replay buffer stores, casts, overwrites, and samples correctly."""

import numpy as np
import pytest
import torch

from roborl.algos.sac.buffer import ReplayBuffer


def _add(buffer: ReplayBuffer, value: float, terminated: bool = False) -> None:
    buffer.add(
        obs=np.full(3, value),
        action=np.full(1, value),
        reward=value,
        next_obs=np.full(3, value + 0.5),
        terminated=terminated,
    )


@pytest.mark.unit
class TestReplayBuffer:
    def test_len_grows_then_caps(self) -> None:
        buffer = ReplayBuffer(capacity=2, obs_shape=(3,), action_shape=(1,))
        assert len(buffer) == 0
        _add(buffer, 1.0)
        assert len(buffer) == 1
        _add(buffer, 2.0)
        _add(buffer, 3.0)
        assert len(buffer) == 2

    def test_wraparound_overwrites_oldest(self) -> None:
        buffer = ReplayBuffer(capacity=2, obs_shape=(3,), action_shape=(1,))
        for value in (1.0, 2.0, 3.0):
            _add(buffer, value)
        batch = buffer.sample(64, torch.device("cpu"))
        stored = set(batch.rewards.tolist())
        assert 1.0 not in stored
        assert stored <= {2.0, 3.0}

    def test_casts_float64_to_float32(self) -> None:
        buffer = ReplayBuffer(capacity=4, obs_shape=(3,), action_shape=(1,))
        buffer.add(
            obs=np.zeros(3, dtype=np.float64),  # MuJoCo emits float64; MPS has none
            action=np.zeros(1, dtype=np.float64),
            reward=1.0,
            next_obs=np.ones(3, dtype=np.float64),
            terminated=False,
        )
        batch = buffer.sample(2, torch.device("cpu"))
        assert batch.observations.dtype == torch.float32
        assert batch.actions.dtype == torch.float32
        assert batch.rewards.dtype == torch.float32
        assert batch.dones.dtype == torch.float32

    def test_shapes_rewards_and_dones_are_1d(self) -> None:
        # A (batch, 1) reward silently broadcasts into a (batch, batch) TD target.
        buffer = ReplayBuffer(capacity=8, obs_shape=(3,), action_shape=(2,))
        for value in (1.0, 2.0, 3.0):
            _add_2d(buffer, value)
        batch = buffer.sample(5, torch.device("cpu"))
        assert batch.observations.shape == (5, 3)
        assert batch.actions.shape == (5, 2)
        assert batch.next_observations.shape == (5, 3)
        assert batch.rewards.shape == (5,)
        assert batch.dones.shape == (5,)

    def test_done_is_terminated_only(self) -> None:
        # The caller stores terminated, never truncated — the truncated
        # transition must keep done=0 so its target bootstraps.
        buffer = ReplayBuffer(capacity=4, obs_shape=(3,), action_shape=(1,))
        _add(buffer, 1.0, terminated=True)
        batch = buffer.sample(8, torch.device("cpu"))
        assert torch.all(batch.dones == 1.0)
        buffer_trunc = ReplayBuffer(capacity=4, obs_shape=(3,), action_shape=(1,))
        _add(buffer_trunc, 1.0, terminated=False)
        batch = buffer_trunc.sample(8, torch.device("cpu"))
        assert torch.all(batch.dones == 0.0)

    def test_transition_integrity(self) -> None:
        # What goes in comes out paired: obs with its own next_obs and reward.
        buffer = ReplayBuffer(capacity=4, obs_shape=(3,), action_shape=(1,))
        _add(buffer, 1.0)
        _add(buffer, 2.0)
        batch = buffer.sample(32, torch.device("cpu"))
        for i in range(32):
            value = batch.rewards[i].item()
            assert torch.all(batch.observations[i] == value)
            assert torch.all(batch.next_observations[i] == value + 0.5)
            assert torch.all(batch.actions[i] == value)

    def test_empty_sample_raises(self) -> None:
        buffer = ReplayBuffer(capacity=4, obs_shape=(3,), action_shape=(1,))
        with pytest.raises(ValueError, match="empty"):
            buffer.sample(1, torch.device("cpu"))


def _add_2d(buffer: ReplayBuffer, value: float) -> None:
    buffer.add(
        obs=np.full(3, value),
        action=np.full(2, value),
        reward=value,
        next_obs=np.full(3, value + 0.5),
        terminated=False,
    )
