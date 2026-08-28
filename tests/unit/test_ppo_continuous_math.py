"""Continuous-PPO policy math reproduces hand-computed Gaussian values.

GAE, the clipped losses, and explained variance are shared with the
discrete variant and covered by ``test_ppo_math.py``; what is new — and
tested here — is the diagonal-Gaussian agent (details C1-C4).
"""

import math

import pytest
import torch

from roborl.algos.ppo.ppo_continuous import Agent

LOG_2PI = math.log(2 * math.pi)


def _zero_mean_agent(obs_dim: int, action_dim: int) -> Agent:
    """An agent whose policy is exactly N(0, 1) per action dimension."""
    agent = Agent(obs_dim, action_dim)
    with torch.no_grad():
        head = agent.actor_mean[-1]
        assert isinstance(head, torch.nn.Linear)
        head.weight.zero_()
        head.bias.zero_()
    return agent


@pytest.mark.unit
class TestAgent:
    def test_shapes_and_action_passthrough(self) -> None:
        torch.manual_seed(0)
        agent = Agent(obs_dim=4, action_dim=3)
        x = torch.randn(8, 4)
        action, logprob, entropy, value = agent.get_action_and_value(x)
        assert action.shape == (8, 3)
        assert logprob.shape == entropy.shape == (8,)
        assert value.shape == (8, 1)
        # Scoring given actions must return those actions, not fresh samples.
        same_action, logprob2, _, _ = agent.get_action_and_value(x, action)
        assert torch.equal(same_action, action)
        assert torch.allclose(logprob, logprob2)

    def test_state_independent_unit_std_at_init(self) -> None:
        # Detail C2: log std is a free zero-initialized parameter, so the
        # initial policy has sigma = 1 in every dimension and every state.
        agent = Agent(obs_dim=4, action_dim=3)
        assert torch.equal(agent.actor_logstd, torch.zeros(1, 3))

    def test_hand_computed_logprob_unit_gaussian(self) -> None:
        # Zeroed mean head + zero log std -> N(0, 1) per dim; the joint
        # log-prob of an action is the per-dim Gaussian log-density summed
        # (detail C3): sum(-a_i^2/2) - dim/2 * log(2*pi).
        agent = _zero_mean_agent(obs_dim=2, action_dim=2)
        action = torch.tensor([[0.5, -1.0]])
        _, logprob, _, _ = agent.get_action_and_value(torch.randn(1, 2), action)
        expected = -0.5 * (0.5**2 + 1.0**2) - LOG_2PI
        assert math.isclose(logprob.item(), expected, rel_tol=1e-6)

    def test_hand_computed_logprob_scaled_std(self) -> None:
        # sigma = 2 per dim: log N(a|0,2) = -a^2/8 - log 2 - log(2*pi)/2.
        agent = _zero_mean_agent(obs_dim=2, action_dim=1)
        with torch.no_grad():
            agent.actor_logstd.fill_(math.log(2.0))
        _, logprob, _, _ = agent.get_action_and_value(torch.randn(1, 2), torch.tensor([[1.0]]))
        expected = -1.0 / 8 - math.log(2.0) - LOG_2PI / 2
        assert math.isclose(logprob.item(), expected, rel_tol=1e-6)

    def test_hand_computed_entropy(self) -> None:
        # Gaussian entropy per dim is (1 + log(2*pi))/2 + log sigma, summed
        # over dims — state-independent, so identical across the batch.
        agent = _zero_mean_agent(obs_dim=3, action_dim=2)
        with torch.no_grad():
            agent.actor_logstd[0, 1] = math.log(2.0)
        _, _, entropy, _ = agent.get_action_and_value(torch.randn(5, 3))
        expected = 2 * (0.5 + LOG_2PI / 2) + math.log(2.0)
        assert torch.allclose(entropy, torch.full((5,), expected), rtol=1e-6)

    def test_orthogonal_init_gains(self) -> None:
        # Detail 2 carried over: mean head gain 0.01, value head gain 1.0,
        # hidden sqrt(2), all biases zero.
        agent = Agent(obs_dim=4, action_dim=2)
        mean_head = agent.actor_mean[-1]
        critic_head = agent.critic[-1]
        assert isinstance(mean_head, torch.nn.Linear)
        assert isinstance(critic_head, torch.nn.Linear)
        assert torch.linalg.svdvals(mean_head.weight).max() < 0.02
        assert math.isclose(
            torch.linalg.svdvals(critic_head.weight).max().item(), 1.0, rel_tol=1e-5
        )
        for module in [*agent.actor_mean, *agent.critic]:
            if isinstance(module, torch.nn.Linear):
                assert torch.equal(module.bias, torch.zeros_like(module.bias))

    def test_unclipped_actions_score_finite(self) -> None:
        # Detail C5: rollout storage keeps the raw Gaussian sample even when
        # it lies outside the action bounds (ClipAction clips only what the
        # env executes), so scoring far-out actions must stay finite.
        agent = _zero_mean_agent(obs_dim=2, action_dim=1)
        _, logprob, _, _ = agent.get_action_and_value(torch.randn(1, 2), torch.tensor([[6.0]]))
        assert math.isclose(logprob.item(), -18.0 - LOG_2PI / 2, rel_tol=1e-6)
