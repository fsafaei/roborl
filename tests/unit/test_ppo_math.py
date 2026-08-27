"""PPO math reproduces hand-computed values on tiny fixtures."""

import math

import numpy as np
import pytest
import torch

from roborl.algos.ppo.ppo import (
    Agent,
    clipped_policy_loss,
    clipped_value_loss,
    compute_gae,
    explained_variance,
)


@pytest.mark.unit
class TestComputeGae:
    def test_hand_computed_no_dones(self) -> None:
        # gamma=0.9, lambda=0.5; deltas are all 9 by construction:
        # delta_t = r_t + 0.9 * V(s_{t+1}) - V(s_t) = 9 for every t.
        # A2 = 9; A1 = 9 + 0.45*9 = 13.05; A0 = 9 + 0.45*13.05 = 14.8725.
        advantages, returns = compute_gae(
            rewards=torch.tensor([[1.0], [2.0], [3.0]]),
            values=torch.tensor([[10.0], [20.0], [30.0]]),
            dones=torch.zeros(3, 1),
            next_value=torch.tensor([40.0]),
            next_done=torch.tensor([0.0]),
            gamma=0.9,
            gae_lambda=0.5,
        )
        assert advantages.shape == (3, 1)
        assert torch.allclose(advantages, torch.tensor([[14.8725], [13.05], [9.0]]), atol=1e-6)
        # Value targets are advantages + old values, not Monte-Carlo returns.
        assert torch.allclose(returns, torch.tensor([[24.8725], [33.05], [39.0]]), atol=1e-6)

    def test_done_masks_bootstrap_and_recursion(self) -> None:
        # dones[2] = 1: obs[2] began a new episode, so step 1 must neither
        # bootstrap from V(s_2) nor accumulate A_2 (the off-by-one detail:
        # step t masks with the *next* row's flag).
        advantages, _ = compute_gae(
            rewards=torch.tensor([[1.0], [2.0], [3.0]]),
            values=torch.tensor([[10.0], [20.0], [30.0]]),
            dones=torch.tensor([[0.0], [0.0], [1.0]]),
            next_value=torch.tensor([40.0]),
            next_done=torch.tensor([0.0]),
            gamma=0.9,
            gae_lambda=0.5,
        )
        # A1 = delta1 = 2 - 20 = -18 (no bootstrap); A0 = 9 + 0.45*(-18) = 0.9.
        assert torch.allclose(advantages, torch.tensor([[0.9], [-18.0], [9.0]]), atol=1e-6)

    def test_next_done_masks_last_row(self) -> None:
        # next_done = 1: the last step must ignore next_value entirely.
        advantages, _ = compute_gae(
            rewards=torch.tensor([[5.0]]),
            values=torch.tensor([[2.0]]),
            dones=torch.zeros(1, 1),
            next_value=torch.tensor([100.0]),
            next_done=torch.tensor([1.0]),
            gamma=0.9,
            gae_lambda=0.5,
        )
        assert math.isclose(advantages.item(), 3.0, rel_tol=1e-6)  # 5 - 2, no bootstrap

    def test_independent_envs_stay_independent(self) -> None:
        # Two envs with different data: each column must match its own
        # single-env computation (a broadcast bug shows up here).
        rewards = torch.tensor([[1.0, 0.0], [2.0, 1.0]])
        values = torch.tensor([[10.0, 5.0], [20.0, 6.0]])
        dones = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        next_value = torch.tensor([40.0, 7.0])
        next_done = torch.tensor([0.0, 0.0])
        both, _ = compute_gae(rewards, values, dones, next_value, next_done, 0.9, 0.5)
        for env in range(2):
            single, _ = compute_gae(
                rewards[:, env : env + 1],
                values[:, env : env + 1],
                dones[:, env : env + 1],
                next_value[env : env + 1],
                next_done[env : env + 1],
                0.9,
                0.5,
            )
            assert torch.allclose(both[:, env : env + 1], single)

    def test_rejects_flat_tensors(self) -> None:
        # Flattened rollouts would silently compute GAE across env boundaries.
        with pytest.raises(ValueError, match="num_steps, num_envs"):
            compute_gae(
                rewards=torch.zeros(6),
                values=torch.zeros(6),
                dones=torch.zeros(6),
                next_value=torch.zeros(1),
                next_done=torch.zeros(1),
                gamma=0.99,
                gae_lambda=0.95,
            )


@pytest.mark.unit
class TestClippedPolicyLoss:
    def test_hand_computed(self) -> None:
        # ratios [1.5, 0.9], advantages [2, -1], eps = 0.2:
        # item0: A>0, ratio clips to 1.2 -> contribution max(-3, -2.4) = -2.4
        # item1: A<0, ratio 0.9 strictly inside the band -> both branches 0.9
        # pg_loss = (-2.4 + 0.9) / 2 = -0.75; clipfrac = 1/2.
        old = torch.tensor([math.log(0.4), math.log(0.5)])
        new = old + torch.tensor([math.log(1.5), math.log(0.9)])
        pg_loss, approx_kl, old_approx_kl, clipfrac = clipped_policy_loss(
            new, old, advantages=torch.tensor([2.0, -1.0]), clip_coef=0.2
        )
        assert math.isclose(pg_loss.item(), -0.75, rel_tol=1e-6)
        assert math.isclose(clipfrac.item(), 0.5, rel_tol=1e-6)
        # approx_kl = mean((r-1) - log r); old_approx_kl = mean(-log r).
        expected_kl = ((1.5 - 1) - math.log(1.5) + (0.9 - 1) - math.log(0.9)) / 2
        expected_old_kl = (-math.log(1.5) - math.log(0.9)) / 2
        assert math.isclose(approx_kl.item(), expected_kl, rel_tol=1e-5)
        assert math.isclose(old_approx_kl.item(), expected_old_kl, rel_tol=1e-5)

    def test_identical_policies_first_minibatch_invariant(self) -> None:
        # new == old: ratio = 1 everywhere, so the loss is -mean(A) and every
        # diagnostic is exactly zero (detail 14's cheap invariant).
        logprob = torch.tensor([-0.7, -1.2, -0.1])
        advantages = torch.tensor([1.0, -2.0, 4.0])
        pg_loss, approx_kl, old_approx_kl, clipfrac = clipped_policy_loss(
            logprob.clone(), logprob, advantages, clip_coef=0.2
        )
        assert math.isclose(pg_loss.item(), -1.0, rel_tol=1e-6)  # -mean([1,-2,4])
        assert approx_kl.item() == 0.0
        assert old_approx_kl.item() == 0.0
        assert clipfrac.item() == 0.0

    def test_gradient_dies_outside_clip_band(self) -> None:
        # For A > 0 and ratio above 1+eps the clipped branch wins and the
        # gradient through new_logprob must vanish — the whole point of PPO.
        old = torch.tensor([0.0])
        new = torch.tensor([math.log(2.0)], requires_grad=True)  # ratio 2 > 1.2
        pg_loss, *_ = clipped_policy_loss(new, old, torch.tensor([1.0]), clip_coef=0.2)
        pg_loss.backward()
        assert new.grad is not None
        assert torch.allclose(new.grad, torch.zeros(1))


@pytest.mark.unit
class TestClippedValueLoss:
    def test_hand_computed(self) -> None:
        # eps=0.5, old=0: item0 clips 1->0.5, (0.5-2)^2=2.25 beats (1-2)^2=1;
        # item1 clips 3->0.5, (0.5-0.5)^2=0 loses to (3-0.5)^2=6.25.
        # loss = 0.5 * mean([2.25, 6.25]) = 2.125.
        loss = clipped_value_loss(
            new_values=torch.tensor([1.0, 3.0]),
            old_values=torch.tensor([0.0, 0.0]),
            returns=torch.tensor([2.0, 0.5]),
            clip_coef=0.5,
        )
        assert math.isclose(loss.item(), 2.125, rel_tol=1e-6)

    def test_reduces_to_mse_inside_band(self) -> None:
        # Predictions within eps of their rollout values: identical to 0.5*MSE.
        new = torch.tensor([1.05, 2.9])
        old = torch.tensor([1.0, 3.0])
        returns = torch.tensor([2.0, 2.0])
        loss = clipped_value_loss(new, old, returns, clip_coef=0.2)
        assert torch.allclose(loss, 0.5 * ((new - returns) ** 2).mean())


@pytest.mark.unit
class TestExplainedVariance:
    def test_hand_computed(self) -> None:
        # Var(R)=8/3, Var(R-V)=2/3 -> EV = 0.75.
        ev = explained_variance(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
        assert math.isclose(ev, 0.75, rel_tol=1e-9)

    def test_perfect_critic_is_one(self) -> None:
        returns = np.array([1.0, -2.0, 5.0])
        assert explained_variance(returns.copy(), returns) == 1.0

    def test_zero_variance_returns_nan(self) -> None:
        assert math.isnan(explained_variance(np.array([1.0, 2.0]), np.array([3.0, 3.0])))


@pytest.mark.unit
class TestAgent:
    def test_shapes_and_action_passthrough(self) -> None:
        torch.manual_seed(0)
        agent = Agent(obs_dim=4, n_actions=3)
        x = torch.randn(8, 4)
        action, logprob, entropy, value = agent.get_action_and_value(x)
        assert action.shape == logprob.shape == entropy.shape == (8,)
        assert value.shape == (8, 1)
        # Scoring given actions must return those actions, not fresh samples.
        same_action, logprob2, _, _ = agent.get_action_and_value(x, action)
        assert torch.equal(same_action, action)
        assert torch.allclose(logprob, logprob2)

    def test_orthogonal_init_gains(self) -> None:
        # Detail 2: policy head gain 0.01, value head gain 1.0, hidden sqrt(2),
        # all biases zero. Gain is recoverable as the singular value of W.
        agent = Agent(obs_dim=4, n_actions=2)
        actor_head = agent.actor[-1]
        critic_head = agent.critic[-1]
        assert isinstance(actor_head, torch.nn.Linear)
        assert isinstance(critic_head, torch.nn.Linear)
        assert torch.linalg.svdvals(actor_head.weight).max() < 0.02
        assert math.isclose(
            torch.linalg.svdvals(critic_head.weight).max().item(), 1.0, rel_tol=1e-5
        )
        for module in [*agent.actor, *agent.critic]:
            if isinstance(module, torch.nn.Linear):
                assert torch.equal(module.bias, torch.zeros_like(module.bias))

    def test_near_uniform_initial_policy(self) -> None:
        # The 0.01 policy head gain exists to make the initial action
        # distribution near-uniform — assert the property, not just the gain.
        torch.manual_seed(0)
        agent = Agent(obs_dim=4, n_actions=5)
        _, _, entropy, _ = agent.get_action_and_value(torch.randn(64, 4))
        assert entropy.min() > 0.999 * math.log(5)
