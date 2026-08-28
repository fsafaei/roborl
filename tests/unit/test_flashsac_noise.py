"""Truncated Zeta sampler and noise repetition semantics."""

import math

import pytest
import torch

from roborl.algos.flashsac.noise import NoiseRepeater, truncated_zeta_pmf


@pytest.mark.unit
class TestTruncatedZetaPmf:
    def test_normalised_and_power_law(self) -> None:
        pmf = truncated_zeta_pmf(mu=2.0, k_max=16)
        assert pmf.shape == (16,)
        assert math.isclose(pmf.sum().item(), 1.0, rel_tol=1e-12)
        # pmf(k) ~ k^-2 -> p(1)/p(2) = 4 exactly.
        assert math.isclose((pmf[0] / pmf[1]).item(), 4.0, rel_tol=1e-12)

    def test_draws_match_pmf(self) -> None:
        torch.manual_seed(0)
        repeater = NoiseRepeater(1, 1)
        draws = torch.tensor([repeater.sample_run_length() for _ in range(100_000)])
        assert draws.min().item() >= 1
        assert draws.max().item() <= 16
        pmf = truncated_zeta_pmf()
        for k in range(1, 17):
            empirical = (draws == k).float().mean().item()
            assert abs(empirical - pmf[k - 1].item()) < 0.01


@pytest.mark.unit
class TestNoiseRepeater:
    def test_noise_held_constant_for_the_sampled_run_length(self) -> None:
        torch.manual_seed(1)
        repeater = NoiseRepeater(1, 4)
        first = repeater.next().clone()
        run_length = repeater.run_length
        assert 1 <= run_length <= 16
        for _ in range(run_length - 1):
            assert torch.equal(repeater.next(), first)
        if run_length < 16:  # a fresh draw is almost surely different
            assert not torch.equal(repeater.next(), first)

    def test_run_lengths_stay_in_bounds_over_many_steps(self) -> None:
        torch.manual_seed(2)
        repeater = NoiseRepeater(2, 3)
        previous = repeater.next().clone()
        run = 1
        for _ in range(500):
            current = repeater.next()
            if torch.equal(current, previous):
                run += 1
            else:
                assert 1 <= run <= 16
                run = 1
                previous = current.clone()
        assert repeater.noise.shape == (2, 3)

    def test_per_env_noise_is_independent(self) -> None:
        torch.manual_seed(3)
        repeater = NoiseRepeater(4, 2)
        noise = repeater.next()
        assert not torch.allclose(noise[0], noise[1])
