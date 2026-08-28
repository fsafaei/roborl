"""Distributional target math reproduces hand-computed fixtures (5 atoms on [-1, 1])."""

import pytest
import torch

from roborl.algos.flashsac.distrib import categorical_td_target, select_min_member

# Support [-1, 1] with 5 atoms: [-1, -0.5, 0, 0.5, 1], bin_width 0.5.
BINS = torch.linspace(-1.0, 1.0, 5)


def _point_mass(atom: int) -> torch.Tensor:
    p = torch.zeros(1, 5)
    p[0, atom] = 1.0
    return p.log()


def _target(
    log_prob: torch.Tensor, r: float, d: float, e: float, gamma: float = 0.9
) -> torch.Tensor:
    return categorical_td_target(
        log_prob,
        rewards=torch.tensor([r]),
        terminated=torch.tensor([d]),
        ent_term=torch.tensor([e]),
        bin_values=BINS,
        gamma=gamma,
    )


@pytest.mark.unit
class TestProjectionFixtures:
    def test_hand_computed_projection(self) -> None:
        # Point mass on atom 3 (z = 0.5); r = 0.1, gamma = 0.9, d = 0, e = 0.
        # z_target = 0.55, b = 3.1, l = 3, u = 4, f = 0.1 -> m = [0, 0, 0, 0.9, 0.1].
        m = _target(_point_mass(3), r=0.1, d=0.0, e=0.0)
        assert torch.allclose(m, torch.tensor([[0.0, 0.0, 0.0, 0.9, 0.1]]), atol=1e-6)

    def test_termination_collapses_to_reward(self) -> None:
        # d = 1: every atom becomes r = 0.1 -> b = 2.2 -> m = [0, 0, 0.8, 0.2, 0],
        # regardless of the next-state distribution.
        spread = torch.full((1, 5), 0.2).log()
        m = _target(spread, r=0.1, d=1.0, e=0.0)
        assert torch.allclose(m, torch.tensor([[0.0, 0.0, 0.8, 0.2, 0.0]]), atol=1e-6)

    def test_clamped_target_keeps_all_mass_on_last_atom(self) -> None:
        # r = 10 pushes every atom above v_max; clamp puts b exactly on the
        # last atom, where l == u — the naive C51 split would lose the mass.
        m = _target(_point_mass(3), r=10.0, d=0.0, e=0.0)
        assert torch.allclose(m, torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]]), atol=1e-6)
        assert torch.allclose(m.sum(dim=-1), torch.ones(1), atol=1e-6)

    def test_entropy_term_shifts_atoms_down(self) -> None:
        # e = alpha*logp = 0.2, gamma = 0.9: every atom drops by 0.18.
        # z_target = 0.1 + 0.9*(0.5 - 0.2) = 0.37 -> b = 2.74 -> [0, 0, 0.26, 0.74, 0].
        m = _target(_point_mass(3), r=0.1, d=0.0, e=0.2)
        assert torch.allclose(m, torch.tensor([[0.0, 0.0, 0.26, 0.74, 0.0]]), atol=1e-6)

    def test_exact_atom_hit_conserves_mass(self) -> None:
        # gamma = 0.5, r = 0.25, z = 0.5: z_target = 0.5 exactly (all values
        # binary-exact), b = 3.0, l = u - 1, f = 0 -> point mass stays put.
        m = _target(_point_mass(3), r=0.25, d=0.0, e=0.0, gamma=0.5)
        assert torch.allclose(m, torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]]), atol=1e-7)

    def test_larger_alpha_raises_expected_target_value(self) -> None:
        # ent_term = alpha * log pi is NEGATIVE for a stochastic policy, so a
        # larger alpha must RAISE the target — a sign flip here trains an
        # entropy-minimising agent that still produces plausible curves.
        log_pi = -2.0
        values = []
        for alpha in (0.0, 0.1, 1.0):
            m = _target(_point_mass(2), r=0.0, d=0.0, e=alpha * log_pi)
            values.append(float((m * BINS).sum()))
        assert values[0] < values[1] < values[2]

    def test_mass_conservation_fuzz(self) -> None:
        # Random targets including clamping on both sides and exact-atom hits.
        torch.manual_seed(3)
        bins = torch.linspace(-5.0, 5.0, 101)
        batch = 256
        log_prob = torch.log_softmax(torch.randn(batch, 101) * 3, dim=-1)
        rewards = torch.empty(batch).uniform_(-20.0, 20.0)
        rewards[:101] = bins  # exact-atom hits via terminated rows
        terminated = (torch.rand(batch) < 0.5).float()
        terminated[:101] = 1.0
        ent = torch.randn(batch)
        m = categorical_td_target(log_prob, rewards, terminated, ent, bins, gamma=0.99)
        assert torch.allclose(m.sum(dim=-1), torch.ones(batch), atol=1e-5)
        assert (m >= 0).all()


@pytest.mark.unit
class TestShapeGuards:
    def test_column_reward_raises(self) -> None:
        # (B, 1) inputs must raise: against (B, n) they broadcast into a
        # transposed target exactly when B == n_atoms, silently.
        log_prob = torch.log_softmax(torch.randn(4, 5), dim=-1)
        with pytest.raises(ValueError, match="1-D"):
            categorical_td_target(
                log_prob,
                rewards=torch.zeros(4, 1),
                terminated=torch.zeros(4),
                ent_term=torch.zeros(4),
                bin_values=BINS,
                gamma=0.9,
            )

    def test_column_terminated_raises(self) -> None:
        log_prob = torch.log_softmax(torch.randn(4, 5), dim=-1)
        with pytest.raises(ValueError, match="1-D"):
            categorical_td_target(
                log_prob,
                rewards=torch.zeros(4),
                terminated=torch.zeros(4, 1),
                ent_term=torch.zeros(4),
                bin_values=BINS,
                gamma=0.9,
            )


@pytest.mark.unit
class TestMinSelection:
    def test_selects_argmin_members_whole_distribution(self) -> None:
        # Member 0 wins sample 0, member 1 wins sample 1; the returned rows
        # must be bitwise the argmin member's rows.
        q = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
        log_prob = torch.log_softmax(torch.randn(2, 2, 4), dim=-1)
        selected = select_min_member(q, log_prob)
        assert selected.shape == (2, 4)
        assert torch.equal(selected[0], log_prob[0, 0])
        assert torch.equal(selected[1], log_prob[1, 1])

    def test_selected_row_is_a_probability_vector(self) -> None:
        q = torch.randn(2, 8)
        log_prob = torch.log_softmax(torch.randn(2, 8, 11), dim=-1)
        selected = select_min_member(q, log_prob)
        assert torch.allclose(selected.exp().sum(dim=-1), torch.ones(8), atol=1e-5)
