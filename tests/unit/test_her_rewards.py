"""Reward recomputation: consistency with storage, argument order, and the oracle's contract."""

from __future__ import annotations

import numpy as np
import pytest

from roborl.algos.her.buffer import HerReplayBuffer

GOAL_DIM = 3
OBS_DIM = 5
T = 7


def sparse_reward(achieved: np.ndarray, desired: np.ndarray, _info: None) -> np.ndarray:
    """A Fetch-shaped oracle: 0 within 0.05, else -1, on the last axis."""
    distance = np.linalg.norm(achieved - desired, axis=-1)
    reward: np.ndarray = -(distance > 0.05).astype(np.float32)
    return reward


def random_walk_episode(rng: np.random.Generator) -> list[dict[str, np.ndarray]]:
    """A random goal-env episode whose stored reward is the oracle on the post-step ag."""
    desired = rng.normal(size=GOAL_DIM)
    ag = rng.normal(size=GOAL_DIM)
    steps = []
    for _ in range(T):
        next_ag = ag + rng.normal(scale=0.04, size=GOAL_DIM)  # small steps: some rewards hit 0
        steps.append(
            {
                "obs": rng.normal(size=OBS_DIM),
                "achieved_goal": ag,
                "desired_goal": desired,
                "action": rng.normal(size=2),
                "next_obs": rng.normal(size=OBS_DIM),
                "next_achieved_goal": next_ag,
            }
        )
        ag = next_ag
    return steps


@pytest.mark.unit
def test_recompute_consistency_with_stored_env_rewards() -> None:
    # compute_reward(next_achieved[t], desired[t], None) must equal the stored env
    # reward for every transition — the check that catches wrong-argument bugs.
    rng = np.random.default_rng(0)
    buffer = HerReplayBuffer(10 * T, T, (OBS_DIM,), (GOAL_DIM,), (2,), sparse_reward)
    for _ in range(10):
        for step in random_walk_episode(rng):
            reward = float(
                sparse_reward(step["next_achieved_goal"][None], step["desired_goal"][None], None)[0]
            )
            buffer.add(
                obs=step["obs"],
                achieved_goal=step["achieved_goal"],
                desired_goal=step["desired_goal"],
                action=step["action"],
                reward=reward,
                next_obs=step["next_obs"],
                next_achieved_goal=step["next_achieved_goal"],
                terminated=False,
            )
        buffer.commit_episode()
    for slot in range(10):
        episode = buffer.stored_episode(slot)
        recomputed = sparse_reward(episode.next_achieved_goals, episode.desired_goals, None)
        np.testing.assert_array_equal(recomputed, episode.rewards)
        # And the pre-step achieved goal would NOT reproduce them in general.
        assert episode.rewards.shape == (T,)


@pytest.mark.unit
def test_oracle_receives_post_step_goals_batched_with_info_none() -> None:
    calls: list[tuple[tuple[int, ...], tuple[int, ...], object]] = []

    def recording_reward(achieved: np.ndarray, desired: np.ndarray, info: None) -> np.ndarray:
        calls.append((achieved.shape, desired.shape, info))
        assert achieved.dtype == np.float32 and desired.dtype == np.float32
        return sparse_reward(achieved, desired, info)

    rng = np.random.default_rng(1)
    buffer = HerReplayBuffer(T, T, (OBS_DIM,), (GOAL_DIM,), (2,), recording_reward, her_k=4)
    steps = random_walk_episode(rng)
    for step in steps:
        buffer.add(
            step["obs"], step["achieved_goal"], step["desired_goal"], step["action"], -1.0,
            step["next_obs"], step["next_achieved_goal"], False,
        )  # fmt: skip
    buffer.commit_episode()
    np.random.seed(0)
    sample = buffer.sample_arrays(10)
    assert calls == [((8, GOAL_DIM), (8, GOAL_DIM), None)]
    # Own-successor rows (f == t) get the floor reward 0 under the post-step rule.
    own = sample.virtual & (sample.goal_step_idx == sample.step_idx)
    assert np.all(sample.rewards[own] == 0.0)


@pytest.mark.unit
def test_relabeled_rewards_are_raw_env_scale() -> None:
    # Nothing rescales recomputed rewards: the value set is exactly the oracle's {-1, 0}.
    rng = np.random.default_rng(2)
    buffer = HerReplayBuffer(T, T, (OBS_DIM,), (GOAL_DIM,), (2,), sparse_reward)
    for step in random_walk_episode(rng):
        buffer.add(
            step["obs"], step["achieved_goal"], step["desired_goal"], step["action"], -1.0,
            step["next_obs"], step["next_achieved_goal"], False,
        )  # fmt: skip
    buffer.commit_episode()
    np.random.seed(0)
    sample = buffer.sample_arrays(500)
    assert set(np.unique(sample.rewards).tolist()) <= {-1.0, 0.0}
    assert sample.rewards.dtype == np.float32
