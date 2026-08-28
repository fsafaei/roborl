"""FlashSAC layers and networks: normalisation invariants, shapes, hand-computed values."""

import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from roborl.algos.flashsac.layers import (
    Block,
    Embedder,
    EnsembleBlock,
    EnsembleUnitBatchNorm,
    EnsembleUnitLinear,
    EnsembleUnitRMSNorm,
    UnitBatchNorm,
    UnitLinear,
    UnitRMSNorm,
)
from roborl.algos.flashsac.networks import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    FlashSACActor,
    FlashSACDoubleCritic,
    Temperature,
    entropy_target,
    safe_tanh_log_det_jacobian,
)


@pytest.mark.unit
class TestUnitLinear:
    def test_no_bias_and_forward_is_matmul(self) -> None:
        layer = UnitLinear(3, 5)
        assert layer.w.bias is None
        x = torch.randn(4, 3)
        assert torch.allclose(layer(x), x @ layer.weight.T)

    def test_normalize_gives_unit_output_rows(self) -> None:
        layer = UnitLinear(7, 4)
        with torch.no_grad():
            layer.w.weight.mul_(3.7)
        layer.normalize_parameters()
        row_norms = layer.weight.norm(dim=-1)
        assert torch.allclose(row_norms, torch.ones(4), atol=1e-5)


@pytest.mark.unit
class TestUnitBatchNorm:
    def test_running_mean_moves_with_pytorch_momentum_convention(self) -> None:
        # new = 0.99 * old + 0.01 * batch — the JAX convention would give 0.99 * batch.
        norm = UnitBatchNorm(2)
        x = torch.tensor([[1.0, 10.0], [3.0, 30.0]])  # batch mean (2, 20)
        norm(x, training=True)
        assert torch.allclose(norm.running_mean, torch.tensor([0.02, 0.20]), atol=1e-6)
        assert not torch.allclose(norm.running_mean, torch.zeros(2))

    def test_train_mode_uses_batch_statistics(self) -> None:
        norm = UnitBatchNorm(1)
        x = torch.tensor([[2.0], [4.0]])  # mean 3, biased var 1
        out = norm(x, training=True)
        expected = (x - 3.0) / math.sqrt(1.0 + norm.eps)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_eval_mode_uses_running_statistics(self) -> None:
        norm = UnitBatchNorm(1)
        with torch.no_grad():
            norm.running_mean.fill_(1.0)
            norm.running_var.fill_(4.0)
        out = norm(torch.tensor([[3.0]]), training=False)
        assert torch.allclose(out, torch.tensor([[2.0 / math.sqrt(4.0 + norm.eps)]]), atol=1e-6)

    def test_normalize_is_joint_over_gamma_and_beta(self) -> None:
        # d = 4, gamma = 2s, beta = 1s: joint sq-sum 20, factor sqrt(4/20).
        norm = UnitBatchNorm(4)
        with torch.no_grad():
            norm.weight.fill_(2.0)
            norm.bias.fill_(1.0)
        norm.normalize_parameters()
        factor = math.sqrt(4.0 / 20.0)
        assert torch.allclose(norm.weight, torch.full((4,), 2.0 * factor), atol=1e-4)
        assert torch.allclose(norm.bias, torch.full((4,), 1.0 * factor), atol=1e-4)
        joint = torch.cat([norm.weight, norm.bias]).norm()
        assert math.isclose(joint.item(), math.sqrt(4.0), rel_tol=1e-4)


@pytest.mark.unit
class TestUnitRMSNorm:
    def test_forward_matches_manual(self) -> None:
        norm = UnitRMSNorm(3)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0]))
        x = torch.tensor([[1.0, -2.0, 2.0]])  # mean square 3
        expected = x / math.sqrt(3.0 + norm.eps) * norm.weight
        assert torch.allclose(norm(x), expected, atol=1e-5)

    def test_normalize_gives_sqrt_d_norm(self) -> None:
        norm = UnitRMSNorm(9)
        with torch.no_grad():
            norm.weight.mul_(5.0)
        norm.normalize_parameters()
        assert math.isclose(norm.weight.norm().item(), 3.0, rel_tol=1e-5)


@pytest.mark.unit
class TestBlocks:
    def test_embedder_normalises_before_linear(self) -> None:
        embedder = Embedder(3, 8)
        x = torch.randn(5, 3)
        expected = embedder.w(embedder.norm(x, False))
        assert torch.allclose(embedder(x, training=False), expected)

    def test_block_is_residual_with_no_post_add_activation(self) -> None:
        block = Block(4, expansion=2)
        x = torch.randn(6, 4)
        path = F.relu(block.n2(block.w2(F.relu(block.n1(block.w1(x), False))), False))
        out = block(x, training=False)
        assert torch.allclose(out, x + path, atol=1e-6)
        # The relu path is non-negative, so out < x detects the residual add
        # surviving un-activated wherever x is negative.
        assert (out >= x - 1e-6).all()


@pytest.mark.unit
class TestEnsembleLayers:
    def test_linear_matches_per_member_matmul(self) -> None:
        layer = EnsembleUnitLinear(2, 3, 5)
        x = torch.randn(2, 4, 3)
        out = layer(x)
        assert out.shape == (2, 4, 5)
        for member in range(2):
            assert torch.allclose(out[member], x[member] @ layer.weight[member].T, atol=1e-6)

    def test_linear_normalize_per_member_rows(self) -> None:
        layer = EnsembleUnitLinear(2, 6, 4)
        with torch.no_grad():
            layer.weight.mul_(2.5)
        layer.normalize_parameters()
        assert torch.allclose(layer.weight.norm(dim=-1), torch.ones(2, 4), atol=1e-5)

    def test_batchnorm_statistics_are_per_member(self) -> None:
        norm = EnsembleUnitBatchNorm(2, 3)
        x = torch.stack([torch.full((8, 3), 1.0), torch.full((8, 3), 5.0)])
        norm(x, training=True)
        assert torch.allclose(norm.running_mean[0], torch.full((3,), 0.01), atol=1e-6)
        assert torch.allclose(norm.running_mean[1], torch.full((3,), 0.05), atol=1e-6)

    def test_batchnorm_forward_matches_single_member(self) -> None:
        ens = EnsembleUnitBatchNorm(2, 3)
        single = UnitBatchNorm(3)
        x = torch.randn(2, 16, 3)
        out = ens(x, training=True)
        for member in range(2):
            assert torch.allclose(out[member], single(x[member], training=True), atol=1e-5)
            with torch.no_grad():
                single.running_mean.zero_()
                single.running_var.fill_(1.0)

    def test_rmsnorm_matches_per_member(self) -> None:
        ens = EnsembleUnitRMSNorm(2, 4)
        with torch.no_grad():
            ens.weight[1].mul_(3.0)
        x = torch.randn(2, 5, 4)
        out = ens(x)
        for member in range(2):
            single = UnitRMSNorm(4)
            with torch.no_grad():
                single.weight.copy_(ens.weight[member])
            assert torch.allclose(out[member], single(x[member]), atol=1e-5)

    def test_ensemble_block_residual(self) -> None:
        block = EnsembleBlock(2, 4, expansion=2)
        x = torch.randn(2, 6, 4)
        out = block(x, training=False)
        assert out.shape == x.shape
        assert (out >= x - 1e-6).all()


def _assert_norm_invariants(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, (UnitLinear, EnsembleUnitLinear)):
            row_norms = child.weight.norm(dim=-1)
            assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-5)
        elif isinstance(child, (UnitBatchNorm, EnsembleUnitBatchNorm)):
            joint = (child.weight.pow(2) + child.bias.pow(2)).sum(dim=-1).sqrt()
            expected = torch.full_like(joint, math.sqrt(child.num_features))
            assert torch.allclose(joint, expected, atol=1e-4)
        elif isinstance(child, (UnitRMSNorm, EnsembleUnitRMSNorm)):
            norms = child.weight.norm(dim=-1) if child.weight.ndim > 1 else child.weight.norm()
            expected = torch.full_like(norms, math.sqrt(child.num_features))
            assert torch.allclose(norms, expected, atol=1e-4)


@pytest.mark.unit
class TestNormalizationAfterOptimizerStep:
    def test_invariants_hold_after_adam_step(self) -> None:
        torch.manual_seed(0)
        actor = FlashSACActor(5, 2, hidden=16, num_blocks=2)
        critic = FlashSACDoubleCritic(5, 2, hidden=16, num_blocks=2, n_atoms=11)
        opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-2)
        obs, act = torch.randn(32, 5), torch.rand(32, 2) * 2 - 1
        a, logp = actor(obs, training=True)
        q, log_prob = critic(obs, act, training=True)
        loss = a.sum() + logp.sum() + q.sum() + log_prob.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        actor.normalize_parameters()
        critic.normalize_parameters()
        _assert_norm_invariants(actor)
        _assert_norm_invariants(critic)

    def test_normalize_does_not_touch_running_stats_or_free_biases(self) -> None:
        critic = FlashSACDoubleCritic(5, 2, hidden=16, num_blocks=1, n_atoms=11)
        critic(torch.randn(16, 5), torch.rand(16, 2), training=True)
        stats_before = critic.embedder.norm.running_mean.clone()
        with torch.no_grad():
            critic.head.bias.fill_(0.3)
        critic.normalize_parameters()
        assert torch.equal(critic.embedder.norm.running_mean, stats_before)
        assert torch.allclose(critic.head.bias, torch.full_like(critic.head.bias, 0.3))


@pytest.mark.unit
class TestActor:
    def test_shapes_and_action_range(self) -> None:
        actor = FlashSACActor(17, 6)
        action, log_prob = actor(torch.randn(8, 17), training=False)
        assert action.shape == (8, 6)
        assert log_prob.shape == (8,)
        # tanh(u) can round to exactly 1.0 in float32 for large |u|.
        assert (action.abs() <= 1.0).all()

    def test_log_std_bounds_are_minus_ten_to_two(self) -> None:
        assert LOG_STD_MIN == -10.0
        assert LOG_STD_MAX == 2.0
        actor = FlashSACActor(3, 2, hidden=8, num_blocks=1)
        _, std = actor.get_mean_and_std(torch.randn(4, 3) * 100, training=False)
        assert (std >= math.exp(-10.0)).all()
        assert (std <= math.exp(2.0)).all()

    def test_eval_action_is_tanh_mean(self) -> None:
        actor = FlashSACActor(3, 2, hidden=8, num_blocks=1)
        obs = torch.randn(4, 3)
        mean, _ = actor.get_mean_and_std(obs, training=False)
        assert torch.allclose(actor.eval_action(obs), torch.tanh(mean))


@pytest.mark.unit
class TestSafeTanhLogDetJacobian:
    def test_matches_naive_form_in_float64(self) -> None:
        u = torch.linspace(-5.0, 5.0, 101, dtype=torch.float64)
        naive = torch.log(1.0 - torch.tanh(u) ** 2)
        assert torch.allclose(safe_tanh_log_det_jacobian(u), naive, atol=1e-12)

    def test_stable_where_naive_underflows(self) -> None:
        # tanh(20)^2 rounds to exactly 1 in float64: the naive form is -inf.
        u = torch.tensor([-20.0, 0.0, 20.0], dtype=torch.float64)
        out = safe_tanh_log_det_jacobian(u)
        assert torch.isfinite(out).all()
        # For |u| = 20, log(1 - tanh^2) = 2*(log 2 - |u|) - 2*log1p(exp(-2|u|)) ~ 2*(log 2 - 20).
        expected = 2.0 * (math.log(2.0) - 20.0)
        assert math.isclose(out[0].item(), expected, rel_tol=1e-12)
        assert math.isclose(out[2].item(), expected, rel_tol=1e-12)
        assert math.isclose(out[1].item(), 0.0, abs_tol=1e-12)

    def test_summed_log_prob_matches_hand_computed(self) -> None:
        # u = 0.5, mean 0, std 1: log N = -0.5*0.25 - 0.5*log(2*pi);
        # correction = log(1 - tanh(0.5)^2).
        actor = FlashSACActor(3, 1, hidden=8, num_blocks=1)
        mean = torch.tensor([[0.0]])
        std = torch.tensor([[1.0]])
        u = torch.tensor([[0.5]])
        dist = torch.distributions.Normal(mean, std)
        log_prob = (dist.log_prob(u) - safe_tanh_log_det_jacobian(u)).sum(dim=-1)
        expected = (-0.125 - 0.5 * math.log(2 * math.pi)) - math.log(1 - math.tanh(0.5) ** 2)
        assert math.isclose(log_prob.item(), expected, rel_tol=1e-5)
        del actor


@pytest.mark.unit
class TestCritic:
    def test_shapes_and_probability_normalisation(self) -> None:
        critic = FlashSACDoubleCritic(17, 6)
        q, log_prob = critic(torch.randn(8, 17), torch.rand(8, 6), training=True)
        assert q.shape == (2, 8)
        assert log_prob.shape == (2, 8, 101)
        assert torch.allclose(log_prob.exp().sum(dim=-1), torch.ones(2, 8), atol=1e-5)

    def test_q_is_expectation_over_support(self) -> None:
        critic = FlashSACDoubleCritic(3, 2, hidden=8, num_blocks=1, n_atoms=5)
        q, log_prob = critic(torch.randn(4, 3), torch.rand(4, 2), training=True)
        manual = (log_prob.exp() * critic.bin_values).sum(dim=-1)
        assert torch.allclose(q, manual)

    def test_support_is_linspace(self) -> None:
        critic = FlashSACDoubleCritic(3, 2, hidden=8, num_blocks=1)
        bins = critic.bin_values.view(-1)
        assert bins.shape == (101,)
        assert bins[0].item() == -5.0
        assert bins[-1].item() == 5.0
        assert math.isclose((bins[1] - bins[0]).item(), 0.1, rel_tol=1e-6)


@pytest.mark.unit
def test_parameter_count_matches_paper() -> None:
    # HalfCheetah-v4: S=17, A=6 -> ~2.44M total (the paper's "2.5M, 6-layer" network).
    actor = FlashSACActor(17, 6)
    critic = FlashSACDoubleCritic(17, 6)
    n_actor = sum(p.numel() for p in actor.parameters())
    n_critic = sum(p.numel() for p in critic.parameters())
    total = n_actor + n_critic
    assert abs(total - 2_440_000) / 2_440_000 < 0.03
    assert abs(n_critic - 2_170_000) / 2_170_000 < 0.03


@pytest.mark.unit
class TestTemperatureAndEntropyTarget:
    def test_temperature_initial_value(self) -> None:
        temp = Temperature(alpha_init=0.01)
        assert math.isclose(temp().item(), 0.01, rel_tol=1e-6)
        assert math.isclose(temp.log_temp.item(), math.log(0.01), rel_tol=1e-6)

    def test_entropy_target_hand_computed(self) -> None:
        # 0.5 * log(2*pi*e*0.15^2) = -0.4782 per dimension (4 decimals).
        per_dim = 0.5 * math.log(2 * math.pi * math.e * 0.15**2)
        assert math.isclose(per_dim, -0.4782, abs_tol=5e-5)
        assert math.isclose(entropy_target(1), per_dim, rel_tol=1e-12)
        assert math.isclose(entropy_target(6), 6 * per_dim, rel_tol=1e-12)
        assert math.isclose(entropy_target(6), -2.8691, abs_tol=5e-4)


@pytest.mark.unit
def test_orthogonal_init_needs_the_init_normalization() -> None:
    # Where out_features > in_features, orthogonal init gives orthonormal
    # COLUMNS, not unit rows — so the reference (and our loop) normalize at
    # construction, before any optimiser step.
    actor = FlashSACActor(17, 6)
    rows = actor.embedder.w.weight.norm(dim=-1)
    assert not torch.allclose(rows, torch.ones_like(rows), atol=1e-3)
    actor.normalize_parameters()
    _assert_norm_invariants(actor)
