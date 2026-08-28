"""Gradient isolation, target EMA, temperature direction, LR schedule.

These replicate the training loop's update fragments exactly (same call
pattern, same freeze/detach discipline) on tiny networks, so a change to
the pattern that couples the objectives fails here.
"""

import copy
import itertools
import math

import pytest
import torch

from roborl.algos.flashsac.distrib import categorical_td_target, select_min_member
from roborl.algos.flashsac.flashsac import cosine_lr
from roborl.algos.flashsac.networks import (
    FlashSACActor,
    FlashSACDoubleCritic,
    Temperature,
    entropy_target,
)

BATCH = 16


def _setup() -> tuple[FlashSACActor, FlashSACDoubleCritic, FlashSACDoubleCritic, Temperature]:
    torch.manual_seed(0)
    actor = FlashSACActor(3, 2, hidden=8, num_blocks=1)
    critic = FlashSACDoubleCritic(3, 2, hidden=8, num_blocks=1, n_atoms=11)
    return actor, critic, copy.deepcopy(critic), Temperature(0.01)


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = torch.randn(BATCH, 3)
    act = torch.rand(BATCH, 2) * 2 - 1
    next_obs = torch.randn(BATCH, 3)
    rewards = torch.randn(BATCH)
    dones = torch.zeros(BATCH)
    return obs, act, next_obs, rewards, dones


def _critic_update(
    actor: FlashSACActor,
    critic: FlashSACDoubleCritic,
    target_critic: FlashSACDoubleCritic,
    temperature: Temperature,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One critic update exactly as the loop performs it; returns (loss, obs_all, act_all)."""
    obs, act, next_obs, rewards, dones = _batch()
    with torch.no_grad():
        a_next, logp_next = actor(next_obs, training=False)
        alpha = temperature()
        ent_term = alpha * logp_next
        obs_all = torch.cat([obs, next_obs], dim=0)
        act_all = torch.cat([act, a_next], dim=0)
        q_all, logp_all = target_critic(obs_all, act_all, training=True)
        log_p = select_min_member(q_all.chunk(2, dim=1)[1], logp_all.chunk(2, dim=1)[1])
        m, _ = categorical_td_target(
            log_p, rewards, dones, ent_term, critic.bin_values.view(-1), gamma=0.99
        )
    _, logp_pred_all = critic(obs_all, act_all, training=True)
    ce = -(m.unsqueeze(0) * logp_pred_all.chunk(2, dim=1)[0]).sum(dim=-1)
    return ce.mean(), obs_all, act_all


@pytest.mark.unit
class TestGradientIsolation:
    def test_critic_loss_touches_only_the_critic(self) -> None:
        # Pitfall 7: alpha read inside no_grad — after critic backward,
        # log_temp and the actor must have no gradient; the critic must.
        actor, critic, target_critic, temperature = _setup()
        critic_loss, _, _ = _critic_update(actor, critic, target_critic, temperature)
        critic_loss.backward()
        assert temperature.log_temp.grad is None
        assert all(p.grad is None for p in actor.parameters())
        assert all(p.grad is None for p in target_critic.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in critic.parameters())

    def test_actor_backward_leaves_critic_grads_untouched(self) -> None:
        # Pitfall 8: critic PARAMETERS frozen around the actor-loss forward,
        # not no_grad — actor grads must be non-zero (gradient flows through
        # the action into the critic), critic grads must stay exactly the
        # ones the critic update produced.
        actor, critic, target_critic, temperature = _setup()
        critic_loss, obs_all, _ = _critic_update(actor, critic, target_critic, temperature)
        opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
        opt.zero_grad()
        critic_loss.backward()
        opt.step()
        critic.normalize_parameters()
        grads_before = [p.grad.clone() for p in critic.parameters() if p.grad is not None]
        assert len(grads_before) == len(list(critic.parameters()))

        with torch.no_grad():
            alpha = temperature()
        a_all, logp_all = actor(obs_all, training=True)
        a_pi = a_all.chunk(2, dim=0)[0]
        logp = logp_all.chunk(2, dim=0)[0]
        critic.requires_grad_(False)
        q_pi, _ = critic(obs_all.chunk(2, dim=0)[0], a_pi, training=False)
        critic.requires_grad_(True)
        actor_loss = (alpha * logp - q_pi.min(dim=0).values).mean()
        actor_loss.backward()

        for before, param in zip(grads_before, critic.parameters(), strict=True):
            assert param.grad is not None
            assert torch.equal(before, param.grad)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters())

    def test_temperature_loss_touches_only_log_temp(self) -> None:
        # Pitfall 9: the entropy is detached in the temperature loss.
        actor, _, _, temperature = _setup()
        obs = torch.randn(BATCH, 3)
        _, logp = actor(obs, training=True)
        entropy = -logp.mean().detach()
        alpha_loss = temperature() * (entropy - entropy_target(2))
        alpha_loss.backward()
        assert all(p.grad is None for p in actor.parameters())
        assert temperature.log_temp.grad is not None
        assert temperature.log_temp.grad.abs().item() > 0


@pytest.mark.unit
class TestTemperatureDirection:
    def test_too_deterministic_policy_raises_alpha(self) -> None:
        # H < H_target must INCREASE alpha after one step (pitfall 11),
        # tested numerically rather than by reading the formula.
        temperature = Temperature(0.01)
        opt = torch.optim.Adam(temperature.parameters(), lr=1e-2)
        alpha_loss = temperature() * torch.tensor(-1.0)  # H - H_target = -1
        opt.zero_grad()
        alpha_loss.backward()
        opt.step()
        assert temperature().item() > 0.01

    def test_too_stochastic_policy_lowers_alpha(self) -> None:
        temperature = Temperature(0.01)
        opt = torch.optim.Adam(temperature.parameters(), lr=1e-2)
        alpha_loss = temperature() * torch.tensor(1.0)  # H - H_target = +1
        opt.zero_grad()
        alpha_loss.backward()
        opt.step()
        assert temperature().item() < 0.01


@pytest.mark.unit
class TestTargetEma:
    def test_lerp_moves_parameters_and_not_buffers(self) -> None:
        # Target 1.0, source 2.0, tau = 0.01 -> 1.01 exactly; BatchNorm
        # running statistics must NOT be touched (pitfall 15).
        _, critic, target_critic, _ = _setup()
        with torch.no_grad():
            for p in critic.parameters():
                p.fill_(2.0)
            for p_t in target_critic.parameters():
                p_t.fill_(1.0)
            target_critic.embedder.norm.running_mean.fill_(0.7)
        with torch.no_grad():
            for p_t, p in zip(target_critic.parameters(), critic.parameters(), strict=True):
                p_t.lerp_(p, 0.01)
        for p_t in target_critic.parameters():
            assert torch.allclose(p_t, torch.full_like(p_t, 1.01))
        assert torch.allclose(
            target_critic.embedder.norm.running_mean,
            torch.full_like(target_critic.embedder.norm.running_mean, 0.7),
        )


@pytest.mark.unit
class TestCosineLr:
    def test_endpoints_and_midpoint(self) -> None:
        kwargs = {"init": 3e-4, "peak": 3e-4, "end": 1.5e-4, "warmup_rate": 1e-6}
        assert math.isclose(cosine_lr(0, 1000, **kwargs), 3e-4, rel_tol=1e-6)
        assert math.isclose(cosine_lr(1000, 1000, **kwargs), 1.5e-4, rel_tol=1e-9)
        assert math.isclose(cosine_lr(500, 1000, **kwargs), 2.25e-4, rel_tol=1e-4)

    def test_monotone_decay_after_warmup(self) -> None:
        kwargs = {"init": 3e-4, "peak": 3e-4, "end": 1.5e-4, "warmup_rate": 1e-6}
        values = [cosine_lr(step, 100, **kwargs) for step in range(101)]
        assert all(a >= b for a, b in itertools.pairwise(values))
