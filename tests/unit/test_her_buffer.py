"""HER buffer relabeling math on the hand-computed micro-episode of docs/algos/her.md.

The worked episode (T = 3, 1-D goals, fake reward ``0 if |ag - g| <= 0.5 else -1``):

    ag:      s0 = 0.0 -> s1 = 1.0 -> s2 = 2.0 -> s3 = 3.0   (next_achieved = [1, 2, 3])
    desired: 10.0 every step  ->  stored rewards [-1, -1, -1], terminated [0, 0, 0]

Observations are ``[t, t]`` so a row's step is readable from its inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from roborl.algos.her.buffer import HerReplayBuffer, RelabeledSample

T = 3
OBS_DIM = 2


def fake_reward(achieved: np.ndarray, desired: np.ndarray, _info: None) -> np.ndarray:
    distance = np.linalg.norm(achieved - desired, axis=-1)
    return np.where(distance <= 0.5, 0.0, -1.0)


def make_buffer(
    buffer_size: int = 30, her_k: int = 4, strategy: str = "future", her_enabled: bool = True
) -> HerReplayBuffer:
    return HerReplayBuffer(
        buffer_size=buffer_size,
        max_episode_steps=T,
        obs_shape=(OBS_DIM,),
        goal_shape=(1,),
        action_shape=(1,),
        compute_reward=fake_reward,
        her_k=her_k,
        strategy=strategy,  # type: ignore[arg-type]  # tests pass plain strings deliberately
        her_enabled=her_enabled,
    )


def stage_micro_episode(
    buffer: HerReplayBuffer,
    ag_offset: float = 0.0,
    desired: float = 10.0,
    terminated_last: bool = False,
) -> None:
    """Stage the worked episode with achieved goals ``offset + [0, 1, 2, 3]`` (no commit)."""
    for t in range(T):
        ag, next_ag = ag_offset + t, ag_offset + t + 1
        buffer.add(
            obs=np.array([t, t], dtype=np.float64),
            achieved_goal=np.array([ag]),
            desired_goal=np.array([desired]),
            action=np.array([0.1 * t]),
            reward=float(fake_reward(np.array([next_ag]), np.array([desired]), None)),
            next_obs=np.array([t + 1, t + 1], dtype=np.float64),
            next_achieved_goal=np.array([next_ag]),
            terminated=terminated_last and t == T - 1,
        )


def committed_micro_buffer(**kwargs: object) -> HerReplayBuffer:
    buffer = make_buffer(**kwargs)  # type: ignore[arg-type]
    stage_micro_episode(buffer)
    buffer.commit_episode()
    return buffer


def rows_at(sample: RelabeledSample, t: int, virtual: bool = True) -> np.ndarray:
    return np.flatnonzero((sample.step_idx == t) & (sample.virtual == virtual))


@pytest.mark.unit
class TestFutureCandidateSets:
    def test_t0_draws_from_all_three_successors(self) -> None:
        np.random.seed(0)
        sample = committed_micro_buffer().sample_arrays(3000)
        goals_t0 = set(sample.goals[rows_at(sample, 0), 0].tolist())
        assert goals_t0 == {1.0, 2.0, 3.0}

    def test_last_step_draws_only_its_own_successor(self) -> None:
        # randint(t, L) at t = L-1 is the single index L-1 — never an empty range.
        np.random.seed(0)
        sample = committed_micro_buffer().sample_arrays(3000)
        rows = rows_at(sample, 2)
        assert rows.size > 0
        assert set(sample.goals[rows, 0].tolist()) == {3.0}
        assert set(sample.goal_step_idx[rows].tolist()) == {2}

    def test_future_index_never_precedes_its_step(self) -> None:
        np.random.seed(1)
        sample = committed_micro_buffer().sample_arrays(5000)
        v = sample.virtual
        assert np.all(sample.goal_step_idx[v] >= sample.step_idx[v])
        assert np.all(sample.goal_step_idx[v] < T)
        assert np.all(sample.goal_step_idx[~v] == -1)

    def test_future_index_uniform_over_candidates(self) -> None:
        # 30k draws: at t = 0 each of the 3 future indices gets ~1/3; at t = 2 all mass on 2.
        np.random.seed(2)
        sample = committed_micro_buffer().sample_arrays(30_000)
        f_t0 = sample.goal_step_idx[rows_at(sample, 0)]
        counts = np.bincount(f_t0, minlength=T) / f_t0.size
        assert np.allclose(counts, 1 / 3, atol=0.03), counts
        assert np.all(sample.goal_step_idx[rows_at(sample, 2)] == 2)


@pytest.mark.unit
class TestRecomputedRewards:
    def test_own_successor_gives_the_reward_floor_of_zero(self) -> None:
        # t = 0 relabeled with g' = 1.0 (= ag_1): |1.0 - 1.0| <= 0.5 -> 0.0
        np.random.seed(0)
        sample = committed_micro_buffer().sample_arrays(3000)
        rows = rows_at(sample, 0)
        own = rows[sample.goals[rows, 0] == 1.0]
        assert own.size > 0
        assert np.all(sample.rewards[own] == 0.0)

    def test_distant_future_goal_gives_minus_one(self) -> None:
        # t = 0 relabeled with g' = 3.0: |1.0 - 3.0| > 0.5 -> -1.0
        np.random.seed(0)
        sample = committed_micro_buffer().sample_arrays(3000)
        rows = rows_at(sample, 0)
        far = rows[sample.goals[rows, 0] == 3.0]
        assert far.size > 0
        assert np.all(sample.rewards[far] == -1.0)

    def test_final_strategy_rewards_by_step(self) -> None:
        # g' = 3.0 for every virtual row: rewards [-1, -1, 0] at t = 0, 1, 2.
        np.random.seed(0)
        sample = committed_micro_buffer(strategy="final").sample_arrays(3000)
        assert np.all(sample.goals[sample.virtual, 0] == 3.0)
        assert np.all(sample.goal_step_idx[sample.virtual] == 2)
        for t, expected in ((0, -1.0), (1, -1.0), (2, 0.0)):
            rows = rows_at(sample, t)
            assert rows.size > 0
            assert np.all(sample.rewards[rows] == expected), t

    def test_recompute_uses_post_step_achieved_goal(self) -> None:
        # With g' = 2.0 at t = 1: post-step ag_2 = 2.0 -> 0.0. Recomputing on the
        # PRE-step ag_1 = 1.0 would give -1.0 — the silent one-step-off bug.
        np.random.seed(0)
        sample = committed_micro_buffer().sample_arrays(3000)
        rows = rows_at(sample, 1)
        own = rows[sample.goals[rows, 0] == 2.0]
        assert own.size > 0
        assert np.all(sample.rewards[own] == 0.0)

    def test_episode_strategy_can_look_backwards(self) -> None:
        np.random.seed(0)
        sample = committed_micro_buffer(strategy="episode").sample_arrays(3000)
        rows = rows_at(sample, 2)
        assert set(sample.goals[rows, 0].tolist()) == {1.0, 2.0, 3.0}


@pytest.mark.unit
class TestBothSidesAndStorage:
    def test_goal_substituted_on_both_obs_and_next_obs(self) -> None:
        np.random.seed(0)
        buffer = committed_micro_buffer()
        sample = buffer.sample_arrays(3000)
        rows = rows_at(sample, 0)
        own = rows[sample.goals[rows, 0] == 1.0]
        assert own.size > 0
        np.testing.assert_array_equal(sample.observations[own], [[0.0, 0.0, 1.0]] * own.size)
        np.testing.assert_array_equal(sample.next_observations[own], [[1.0, 1.0, 1.0]] * own.size)
        # Every row, real or virtual: the goal on the next side equals the goal on the obs side.
        np.testing.assert_array_equal(sample.observations[:, -1], sample.next_observations[:, -1])
        np.testing.assert_array_equal(sample.observations[:, -1], sample.goals[:, 0])

    def test_relabeling_never_mutates_storage(self) -> None:
        np.random.seed(0)
        buffer = committed_micro_buffer()
        buffer.sample_arrays(3000)
        stored = buffer.stored_episode(0)
        np.testing.assert_array_equal(stored.desired_goals, [[10.0]] * T)
        np.testing.assert_array_equal(stored.rewards, [-1.0] * T)
        np.testing.assert_array_equal(stored.next_achieved_goals, [[1.0], [2.0], [3.0]])

    def test_real_rows_are_bitwise_storage(self) -> None:
        np.random.seed(0)
        sample = committed_micro_buffer().sample_arrays(3000)
        real = ~sample.virtual
        assert real.sum() == 3000 - 2400
        assert np.all(sample.goals[real, 0] == 10.0)
        assert np.all(sample.rewards[real] == -1.0)
        assert np.all(sample.observations[real, -1] == 10.0)


@pytest.mark.unit
class TestDones:
    def test_relabeled_success_does_not_manufacture_a_termination(self) -> None:
        np.random.seed(0)
        sample = committed_micro_buffer().sample_arrays(3000)
        assert np.any(sample.rewards == 0.0)  # some rows hit their substituted goal
        assert np.all(sample.dones == 0.0)

    def test_stored_terminated_survives_relabeling_bitwise(self) -> None:
        buffer = make_buffer()
        stage_micro_episode(buffer, terminated_last=True)
        buffer.commit_episode()
        np.random.seed(0)
        sample = buffer.sample_arrays(3000)
        np.testing.assert_array_equal(sample.dones, (sample.step_idx == T - 1).astype(np.float32))


@pytest.mark.unit
class TestSplitArithmetic:
    @pytest.mark.parametrize(
        ("her_k", "batch", "expected"),
        [(4, 10, 8), (4, 2048, 1638), (1, 10, 5), (8, 90, 80), (0, 10, 0)],
    )
    def test_fixed_virtual_count(self, her_k: int, batch: int, expected: int) -> None:
        buffer = committed_micro_buffer(her_k=her_k)
        sample = buffer.sample_arrays(batch)
        assert int(sample.virtual.sum()) == expected  # a count, not a proportion
        assert np.all(sample.virtual[:expected]) and not np.any(sample.virtual[expected:])

    def test_her_disabled_relabels_nothing(self) -> None:
        buffer = committed_micro_buffer(her_enabled=False)
        assert buffer.virtual_fraction == 0.0
        sample = buffer.sample_arrays(64)
        assert not sample.virtual.any()
        assert np.all(sample.goals == 10.0)
        assert np.all(sample.rewards == -1.0)
        assert np.isnan(sample.virtual_reward_zero_fraction)

    def test_stats_properties(self) -> None:
        np.random.seed(0)
        buffer = committed_micro_buffer()
        sample = buffer.sample_arrays(10)
        assert sample.virtual_fraction == 0.8
        zero_frac = sample.virtual_reward_zero_fraction
        assert zero_frac == np.mean(sample.rewards[:8] == 0.0)
        buffer.sample(10, torch.device("cpu"))
        assert buffer.last_sample_stats[0] == 0.8


@pytest.mark.unit
class TestCrossEpisodeLeakage:
    def test_substituted_goals_stay_inside_their_source_episode(self) -> None:
        # Episode i has achieved goals in [10 i, 10 i + 3]; 10k relabels never cross.
        buffer = make_buffer(buffer_size=5 * T)
        for i in range(5):
            stage_micro_episode(buffer, ag_offset=10.0 * i)
            buffer.commit_episode()
        np.random.seed(3)
        sample = buffer.sample_arrays(10_000)
        v = sample.virtual
        lo = 10.0 * sample.episode_idx[v]
        goals = sample.goals[v, 0]
        assert np.all((goals >= lo + 1.0) & (goals <= lo + 3.0))
        assert len(set(sample.episode_idx.tolist())) == 5


@pytest.mark.unit
class TestStagingAndCapacity:
    def test_sample_raises_with_only_staged_steps(self) -> None:
        buffer = make_buffer()
        stage_micro_episode(buffer)
        assert len(buffer) == 0
        assert buffer.staged_steps == T
        with pytest.raises(ValueError, match="commit_episode"):
            buffer.sample_arrays(4)
        buffer.commit_episode()
        assert len(buffer) == T
        assert buffer.sample_arrays(4).observations.shape == (4, OBS_DIM + 1)

    def test_staged_half_episode_never_contributes_goals(self) -> None:
        buffer = committed_micro_buffer(buffer_size=4 * T)
        buffer.add(  # a distinctive in-flight step
            obs=np.zeros(OBS_DIM),
            achieved_goal=np.array([100.0]),
            desired_goal=np.array([10.0]),
            action=np.zeros(1),
            reward=-1.0,
            next_obs=np.zeros(OBS_DIM),
            next_achieved_goal=np.array([100.0]),
            terminated=False,
        )
        np.random.seed(0)
        sample = buffer.sample_arrays(5000)
        assert not np.any(sample.goals == 100.0)
        assert not np.any(sample.observations[:, -1] == 100.0)

    def test_add_beyond_horizon_raises(self) -> None:
        buffer = make_buffer()
        stage_micro_episode(buffer)
        with pytest.raises(ValueError, match="commit_episode"):
            buffer.add(
                np.zeros(OBS_DIM), np.zeros(1), np.zeros(1), np.zeros(1), 0.0,
                np.zeros(OBS_DIM), np.zeros(1), False,
            )  # fmt: skip

    def test_commit_with_nothing_staged_raises(self) -> None:
        with pytest.raises(ValueError, match="no staged"):
            make_buffer().commit_episode()

    def test_capacity_is_in_episodes_and_overwrites_per_slot(self) -> None:
        buffer = make_buffer(buffer_size=2 * T)  # N = 2 slots
        assert buffer.capacity_episodes == 2
        for i in range(3):
            stage_micro_episode(buffer, ag_offset=10.0 * i)
            assert buffer.commit_episode() == i % 2
        assert len(buffer) == 2 * T
        assert buffer.n_episodes == 2
        np.testing.assert_array_equal(buffer.stored_episode(0).achieved_goals[:, 0], [20, 21, 22])
        np.testing.assert_array_equal(buffer.stored_episode(1).achieved_goals[:, 0], [10, 11, 12])

    def test_capacity_arithmetic(self) -> None:
        buffer = HerReplayBuffer(1_000_000, 50, (25,), (3,), (4,), fake_reward)
        assert buffer.capacity_episodes == 20_000
        with pytest.raises(ValueError, match="no complete episode"):
            HerReplayBuffer(10, 50, (25,), (3,), (4,), fake_reward)

    def test_shorter_episode_stores_true_length(self) -> None:
        buffer = make_buffer()
        buffer.add(
            np.zeros(OBS_DIM), np.zeros(1), np.array([10.0]), np.zeros(1), -1.0,
            np.ones(OBS_DIM), np.ones(1), True,
        )  # fmt: skip
        buffer.commit_episode()
        assert len(buffer) == 1
        np.testing.assert_array_equal(buffer.committed_lengths(), [1])
        sample = buffer.sample_arrays(16)
        assert np.all(sample.step_idx == 0) and np.all(sample.goal_step_idx[sample.virtual] == 0)
        assert np.all(sample.dones == 1.0)


@pytest.mark.unit
class TestBatchContract:
    def test_shapes_dtypes_and_float32_cast(self) -> None:
        buffer = committed_micro_buffer()
        batch = buffer.sample(5, torch.device("cpu"))
        assert batch.observations.shape == (5, OBS_DIM + 1)
        assert batch.next_observations.shape == (5, OBS_DIM + 1)
        assert batch.actions.shape == (5, 1)
        assert batch.rewards.shape == (5,)
        assert batch.dones.shape == (5,)
        for tensor in (
            batch.observations,
            batch.actions,
            batch.next_observations,
            batch.rewards,
            batch.dones,
        ):
            assert tensor.dtype == torch.float32  # inputs above were float64

    def test_same_seed_same_sample(self) -> None:
        buffer = committed_micro_buffer()
        np.random.seed(11)
        first = buffer.sample_arrays(64)
        np.random.seed(11)
        second = buffer.sample_arrays(64)
        np.testing.assert_array_equal(first.observations, second.observations)
        np.testing.assert_array_equal(first.goal_step_idx, second.goal_step_idx)
        np.testing.assert_array_equal(first.rewards, second.rewards)

    def test_rejects_bad_configuration(self) -> None:
        with pytest.raises(ValueError, match="strategy"):
            make_buffer(strategy="past")
        with pytest.raises(ValueError, match="her_k"):
            make_buffer(her_k=-1)

    def test_rejects_unvectorised_reward_function(self) -> None:
        def scalar_reward(achieved: np.ndarray, desired: np.ndarray, _info: None) -> np.ndarray:
            return np.asarray(-1.0, dtype=np.float32)  # 0-d: not vectorised over the batch

        buffer = HerReplayBuffer(30, T, (OBS_DIM,), (1,), (1,), scalar_reward)
        stage_micro_episode(buffer)
        buffer.commit_episode()
        with pytest.raises(ValueError, match="compute_reward returned shape"):
            buffer.sample_arrays(8)


@pytest.mark.unit
class TestTransitionUniformSampling:
    def test_variable_length_episodes_are_weighted_by_length(self) -> None:
        # Pass B alignment with SB3: draws are uniform over valid TRANSITIONS, so a
        # 1-step episode next to a 3-step one gets ~1/4 of the rows, not 1/2.
        buffer = make_buffer(buffer_size=3 * T)
        buffer.add(
            np.zeros(OBS_DIM), np.zeros(1), np.array([10.0]), np.zeros(1), -1.0,
            np.ones(OBS_DIM), np.ones(1), False,
        )  # fmt: skip
        buffer.commit_episode()  # slot 0, length 1
        stage_micro_episode(buffer, ag_offset=10.0)
        buffer.commit_episode()  # slot 1, length 3
        np.random.seed(5)
        sample = buffer.sample_arrays(20_000)
        share_short = float(np.mean(sample.episode_idx == 0))
        assert abs(share_short - 0.25) < 0.02, share_short
        assert np.all(sample.step_idx[sample.episode_idx == 0] == 0)
        assert set(sample.step_idx[sample.episode_idx == 1].tolist()) == {0, 1, 2}

    def test_empty_slots_between_committed_ones_are_never_drawn(self) -> None:
        # Overwrite leaves the ring with a mix of slot ages; only non-empty slots appear.
        buffer = make_buffer(buffer_size=4 * T)
        stage_micro_episode(buffer, ag_offset=0.0)
        buffer.commit_episode()  # slot 0
        stage_micro_episode(buffer, ag_offset=10.0)
        buffer.commit_episode()  # slot 1; slots 2, 3 empty
        np.random.seed(6)
        sample = buffer.sample_arrays(2000)
        assert set(sample.episode_idx.tolist()) == {0, 1}
