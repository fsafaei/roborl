"""HER+SAC loop math and plumbing on hand-computed fixtures (no Fetch needed)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from roborl.algos.her.her_sac import (
    Actor,
    HerSacConfig,
    SoftQNetwork,
    evaluate_policy,
    q_lower_bound_violation,
    soft_td_target,
)
from tests.unit.test_her_goals import FakeGoalEnv


@pytest.mark.unit
def test_gamma_0_95_reaches_the_td_target() -> None:
    # reward -1, done 0, min_q_next -10, log_pi 0 -> -1 + 0.95 * (-10) = -10.5
    config = HerSacConfig()
    assert config.gamma == 0.95
    target = soft_td_target(
        rewards=torch.tensor([-1.0]),
        dones=torch.tensor([0.0]),
        gamma=config.gamma,
        alpha=0.0,
        min_q_next=torch.tensor([[-10.0]]),
        next_log_pi=torch.tensor([[0.0]]),
    )
    assert target.shape == (1,)
    assert math.isclose(target.item(), -10.5, rel_tol=1e-6)


@pytest.mark.unit
def test_td_target_rejects_2d_rewards() -> None:
    with pytest.raises(ValueError, match="1-D"):
        soft_td_target(
            torch.zeros(2, 1), torch.zeros(2), 0.95, 0.0, torch.zeros(2, 1), torch.zeros(2, 1)
        )


@pytest.mark.unit
def test_q_lower_bound_violation_fraction() -> None:
    # gamma 0.95 -> floor -20, threshold -21: [-25, -15, -20.9] -> 1/3 violate.
    fraction = q_lower_bound_violation(torch.tensor([-25.0, -15.0, -20.9]), gamma=0.95)
    assert math.isclose(fraction, 1 / 3, rel_tol=1e-6)
    assert q_lower_bound_violation(torch.tensor([-20.0, 0.0]), gamma=0.95) == 0.0


@pytest.mark.unit
def test_recipe_defaults_match_the_spec_table() -> None:
    config = HerSacConfig()
    assert (config.tau, config.batch_size, config.learning_starts) == (0.05, 2048, 1_000)
    assert config.policy_lr == config.q_lr == 1e-3
    assert config.net_arch == (512, 512, 512)
    assert (config.her_enabled, config.her_strategy, config.her_k) == (True, "future", 4)
    assert config.buffer_size // 50 == 20_000
    assert config.policy_frequency == 2 and config.target_network_frequency == 1


@pytest.mark.unit
def test_networks_follow_net_arch() -> None:
    actor = Actor(13, 4, -np.ones(4), np.ones(4), hidden_sizes=(64, 32))
    critic = SoftQNetwork(13, 4, hidden_sizes=(64, 32))
    linears = [m for m in actor.trunk if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [(13, 64), (64, 32)]
    assert actor.fc_mean.in_features == 32
    x, a = torch.zeros(5, 13), torch.zeros(5, 4)
    assert critic(x, a).shape == (5, 1)
    action, log_prob, mean_action = actor.get_action(x)
    assert action.shape == (5, 4) and log_prob.shape == (5, 1) and mean_action.shape == (5, 4)
    assert torch.all(action.abs() <= 1.0)


@pytest.mark.unit
def test_evaluate_policy_reads_final_step_success_only() -> None:
    # The fake env passes through its goal at t = 2 and leaves: final-step success is 0.
    env = FakeGoalEnv()
    env.reset(seed=0)
    actor = Actor(3, 1, -np.ones(1), np.ones(1), hidden_sizes=(8,))
    result = evaluate_policy(
        actor, _with_episode_stats(env), episodes=3, device=torch.device("cpu")
    )
    assert result.success_rate == 0.0
    assert result.episodic_length == 5.0
    assert result.episodic_return == -4.0  # rewards [-1, 0, -1, -1, -1]


def _with_episode_stats(env: FakeGoalEnv) -> FakeGoalEnv:
    import gymnasium as gym

    wrapped = gym.wrappers.RecordEpisodeStatistics(env)
    wrapped.reset(seed=0)
    return wrapped  # type: ignore[return-value]
