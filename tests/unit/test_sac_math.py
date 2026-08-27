"""SAC math reproduces hand-computed values on tiny fixtures."""

import math

import numpy as np
import pytest
import torch

from roborl.algos.sac.sac import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    Actor,
    soft_td_target,
    squashed_gaussian_log_prob,
)


def _hand_log_prob(x: float, mean: float, std: float, scale: float) -> float:
    # log N(x; mean, std) - log(scale * (1 - tanh(x)^2) + 1e-6), by hand with math.
    gaussian = -0.5 * ((x - mean) / std) ** 2 - math.log(std) - 0.5 * math.log(2 * math.pi)
    correction = math.log(scale * (1 - math.tanh(x) ** 2) + 1e-6)
    return gaussian - correction


@pytest.mark.unit
class TestSquashedGaussianLogProb:
    def test_hand_computed_single_dim(self) -> None:
        # x_t=0.5, mean=0, std=1, scale=2 -> -1.0439385 - 0.4529188 = -1.4968573
        out = squashed_gaussian_log_prob(
            x_t=torch.tensor([[0.5]]),
            mean=torch.tensor([[0.0]]),
            log_std=torch.tensor([[0.0]]),
            action_scale=torch.tensor([2.0]),
        )
        assert out.shape == (1, 1)
        expected = _hand_log_prob(0.5, 0.0, 1.0, 2.0)
        assert math.isclose(out.item(), expected, rel_tol=1e-6)
        assert math.isclose(expected, -1.4968573, rel_tol=1e-6)

    def test_dimensions_sum(self) -> None:
        # Two independent dims: log-probs add. std = e^0.5 on the second dim.
        out = squashed_gaussian_log_prob(
            x_t=torch.tensor([[0.5, -0.3]]),
            mean=torch.tensor([[0.0, 0.1]]),
            log_std=torch.tensor([[0.0, 0.5]]),
            action_scale=torch.tensor([2.0, 1.0]),
        )
        expected = _hand_log_prob(0.5, 0.0, 1.0, 2.0) + _hand_log_prob(
            -0.3, 0.1, math.exp(0.5), 1.0
        )
        assert out.shape == (1, 1)
        assert math.isclose(out.item(), expected, rel_tol=1e-5)

    def test_scale_enters_correction(self) -> None:
        # Doubling the action scale must subtract exactly log(2) per dim —
        # the classic bug is dropping the scale from inside the correction.
        args = {
            "x_t": torch.tensor([[0.0]]),
            "mean": torch.tensor([[0.0]]),
            "log_std": torch.tensor([[0.0]]),
        }
        narrow = squashed_gaussian_log_prob(**args, action_scale=torch.tensor([1.0]))
        wide = squashed_gaussian_log_prob(**args, action_scale=torch.tensor([2.0]))
        assert math.isclose((narrow - wide).item(), math.log(2.0), abs_tol=1e-5)

    def test_saturation_is_finite(self) -> None:
        # Deep in tanh saturation the 1e-6 floor keeps the value finite.
        out = squashed_gaussian_log_prob(
            x_t=torch.tensor([[20.0]]),
            mean=torch.tensor([[0.0]]),
            log_std=torch.tensor([[0.0]]),
            action_scale=torch.tensor([1.0]),
        )
        assert torch.isfinite(out).all()


@pytest.mark.unit
class TestSoftTdTarget:
    def test_hand_computed(self) -> None:
        # y0 = 1 + 0.9*(10 - 0.1*(-1)) = 10.09; y1 terminated -> reward only.
        out = soft_td_target(
            rewards=torch.tensor([1.0, 2.0]),
            dones=torch.tensor([0.0, 1.0]),
            gamma=0.9,
            alpha=0.1,
            min_q_next=torch.tensor([[10.0], [20.0]]),
            next_log_pi=torch.tensor([[-1.0], [-2.0]]),
        )
        assert out.shape == (2,)
        assert torch.allclose(out, torch.tensor([10.09, 2.0]), atol=1e-6)

    def test_truncation_bootstraps(self) -> None:
        # A truncated transition is stored with done=0 and must keep the
        # bootstrap term (bootstrap through time limits, not termination).
        out = soft_td_target(
            rewards=torch.tensor([0.0]),
            dones=torch.tensor([0.0]),
            gamma=0.99,
            alpha=0.0,
            min_q_next=torch.tensor([[5.0]]),
            next_log_pi=torch.tensor([[0.0]]),
        )
        assert math.isclose(out.item(), 4.95, rel_tol=1e-6)

    def test_rejects_2d_rewards(self) -> None:
        # (batch, 1) rewards would silently broadcast to a (batch, batch) target.
        with pytest.raises(ValueError, match="1-D"):
            soft_td_target(
                rewards=torch.zeros(2, 1),
                dones=torch.zeros(2),
                gamma=0.99,
                alpha=0.1,
                min_q_next=torch.zeros(2, 1),
                next_log_pi=torch.zeros(2, 1),
            )


@pytest.mark.unit
class TestActor:
    def test_actions_respect_asymmetric_bounds(self) -> None:
        # Bounds [0, 4]: scale=2, bias=2. A missing bias or scale shows up here.
        torch.manual_seed(0)
        actor = Actor(3, 1, action_low=np.array([0.0]), action_high=np.array([4.0]))
        action, log_prob, mean_action = actor.get_action(torch.randn(64, 3))
        assert action.min() >= 0.0 and action.max() <= 4.0
        assert mean_action.min() >= 0.0 and mean_action.max() <= 4.0
        assert log_prob.shape == (64, 1)

    def test_log_std_stays_in_bounds(self) -> None:
        actor = Actor(3, 2, action_low=np.array([-1.0, -1.0]), action_high=np.array([1.0, 1.0]))
        _, log_std = actor(torch.randn(32, 3) * 100)  # extreme inputs
        assert log_std.min() >= LOG_STD_MIN
        assert log_std.max() <= LOG_STD_MAX
