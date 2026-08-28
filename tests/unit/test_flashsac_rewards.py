"""Reward normaliser reproduces a hand-run stream; Chan update matches numpy."""

import math

import numpy as np
import pytest
import torch

from roborl.algos.flashsac.rewards import RewardNormalizer, RunningMeanVar


@pytest.mark.unit
class TestRewardNormalizerHandRun:
    def test_three_steps_with_mid_termination(self) -> None:
        # gamma = 0.9. Step 1: r=1.0 -> G=1.0. Step 2: r=2.0 terminated ->
        # accumulator reset, G=2.0. Step 3: r=0.5 -> G=0.9*2+0.5=2.3.
        # Stream of G values [1, 2, 2.3]: population var = 0.3088889.
        norm = RewardNormalizer(gamma=0.9)
        norm.update(1.0, terminated=False, truncated=False)
        assert math.isclose(float(norm.accumulator[0]), 1.0)
        norm.update(2.0, terminated=True, truncated=False)
        assert math.isclose(float(norm.accumulator[0]), 2.0)
        norm.update(0.5, terminated=False, truncated=False)
        assert math.isclose(float(norm.accumulator[0]), 2.3)
        assert math.isclose(norm.g_max_seen, 2.3)
        expected_var = np.var([1.0, 2.0, 2.3])
        assert math.isclose(norm.rms.var, float(expected_var), rel_tol=1e-9)
        # sqrt(0.3088889 + 1e-8) = 0.5558 beats G_max_seen / 5 = 0.46.
        assert math.isclose(norm.denominator, math.sqrt(expected_var + 1e-8), rel_tol=1e-9)

    def test_truncation_also_resets_accumulator(self) -> None:
        # BOTH flags reset here — unlike the TD target, which uses terminated only.
        norm = RewardNormalizer(gamma=0.9)
        norm.update(1.0, terminated=False, truncated=False)
        norm.update(1.0, terminated=False, truncated=True)
        assert math.isclose(float(norm.accumulator[0]), 1.0)  # not 1.9

    def test_g_max_branch_wins_when_returns_are_large(self) -> None:
        # A single huge constant return: var ~ 0, so G_max_seen / 5 governs
        # and guarantees the scaled returns fit inside [-5, 5].
        norm = RewardNormalizer(gamma=0.0)
        norm.update(100.0, terminated=False, truncated=False)
        assert math.isclose(norm.denominator, 20.0, rel_tol=1e-6)

    def test_normalize_divides_by_denominator(self) -> None:
        norm = RewardNormalizer(gamma=0.9)
        for r in (1.0, 3.0, -2.0):
            norm.update(r, terminated=False, truncated=False)
        rewards = torch.tensor([1.0, -4.0])
        out = norm.normalize(rewards)
        assert torch.allclose(out, rewards / norm.denominator)


@pytest.mark.unit
class TestRunningMeanVar:
    def test_matches_numpy_over_batched_stream(self) -> None:
        rng = np.random.default_rng(0)
        rms = RunningMeanVar()
        chunks = [rng.normal(size=2) * 3 + 1 for _ in range(50)]
        for chunk in chunks:
            rms.update(chunk)
        everything = np.concatenate(chunks)
        assert math.isclose(rms.mean, float(everything.mean()), rel_tol=1e-10)
        assert math.isclose(rms.var, float(everything.var()), rel_tol=1e-10)
        assert rms.count == 100.0

    def test_vectorised_stream_matches_scalar_stream(self) -> None:
        # Two parallel envs folded per step == the same values one at a time.
        values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        batched = RunningMeanVar()
        for row in values:
            batched.update(row)
        scalar = RunningMeanVar()
        for v in values.flatten():
            scalar.update(np.array([v]))
        assert math.isclose(batched.mean, scalar.mean, rel_tol=1e-12)
        assert math.isclose(batched.var, scalar.var, rel_tol=1e-12)
