"""Distributional TD target construction: min-selection, atom shift, projection.

Pure tensor functions (docs/algos/flashsac.md §"Update equations"), separated
from the training loop so each step can be unit-tested against hand-computed
fixtures. Callers run everything here under ``torch.no_grad()``.

The projection uses the mass-conserving two-point split ``m_l += p*(1-f)``,
``m_u += p*f`` with ``u = min(l+1, n-1)`` — total probability is preserved
even when the target lands exactly on an atom or is clamped to the support
boundary. The classic C51 form ``m_l = p*(u-b)``, ``m_u = p*(b-l)`` silently
*loses* that mass when ``l == u``; do not swap it in.
"""

from __future__ import annotations

import torch


def select_min_member(q: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
    """Clipped double-Q in distributional form.

    Argmin over the ensemble's **scalar expected values**, then gather that
    member's **whole distribution** — an elementwise per-atom min of two
    probability vectors is not a probability vector.

    Args:
        q: Expected values per member, shape ``(E, B)``.
        log_prob: Atom log-probabilities per member, shape ``(E, B, n)``.

    Returns:
        The argmin member's log-probabilities per sample, shape ``(B, n)``.
    """
    ensemble_size, batch = q.shape
    assert log_prob.shape[:2] == (ensemble_size, batch)
    j = q.argmin(dim=0)
    idx = j[None, :, None].expand(1, batch, log_prob.shape[-1])
    selected: torch.Tensor = log_prob.gather(0, idx)[0]
    assert selected.shape == log_prob.shape[1:]
    return selected


def categorical_td_target(
    log_prob: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    ent_term: torch.Tensor,
    bin_values: torch.Tensor,
    gamma: float,
    n_step: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift every atom by the soft Bellman backup and project onto the support.

    ``z_target = r + gamma**n_step * (z - ent_term) * (1 - d)``, clamped to
    the support, then two-point projected. The entropy bonus enters as the
    per-atom shift ``-ent_term`` where ``ent_term = alpha * log pi(a'|s')``
    (negative entropy — the shift *raises* the target for a stochastic
    policy). ``terminated`` is true termination only — truncation
    bootstraps; at a true termination the target collapses to a point mass
    at ``r``. Rewards must already be reward-normalised.

    Args:
        log_prob: Selected member's next-state atom log-probabilities ``(B, n)``.
        rewards: Normalised rewards, shape ``(B,)`` — 1-D enforced; a
            ``(B, 1)`` here broadcasts into a transposed target whenever
            ``B == n`` and raises otherwise.
        terminated: True-termination flags, shape ``(B,)``.
        ent_term: ``alpha * log pi(a'|s')``, shape ``(B,)``.
        bin_values: The support atoms, shape ``(n,)`` (ascending, uniform).
        gamma: Discount factor.
        n_step: Return horizon (1 in this phase).

    Returns:
        ``(m, clamp_fraction)``: the projected target distribution, shape
        ``(B, n)`` with every row summing to 1, and the scalar fraction of
        ``z_target`` entries at or beyond the support bounds before
        clamping — ``diagnostics/target_clamp_fraction``, the single best
        FlashSAC health signal (a climb above a few percent means the
        reward scaling is not keeping returns inside the support).

    Raises:
        ValueError: If ``rewards``, ``terminated`` or ``ent_term`` is not 1-D.
    """
    if rewards.ndim != 1 or terminated.ndim != 1 or ent_term.ndim != 1:
        raise ValueError(
            "rewards, terminated and ent_term must be 1-D, got shapes "
            f"{tuple(rewards.shape)}, {tuple(terminated.shape)} and {tuple(ent_term.shape)}."
        )
    batch = rewards.shape[0]
    n_atoms = bin_values.shape[-1]
    assert log_prob.shape == (batch, n_atoms)

    z = bin_values.view(1, n_atoms)
    v_min = float(bin_values[0])
    v_max = float(bin_values[-1])
    bin_width = (v_max - v_min) / (n_atoms - 1)

    r = rewards.view(batch, 1)
    d = terminated.view(batch, 1)
    e = ent_term.view(batch, 1)
    z_target = r + (gamma**n_step) * (z - e) * (1.0 - d)
    clamp_fraction = ((z_target <= v_min) | (z_target >= v_max)).float().mean()
    z_target = z_target.clamp(v_min, v_max)

    b = (z_target - v_min) / bin_width
    lower = b.floor().long().clamp(0, n_atoms - 1)
    upper = (lower + 1).clamp(0, n_atoms - 1)
    frac = b - lower.float()
    p = log_prob.exp()
    m = torch.zeros_like(p)
    m.scatter_add_(1, lower, p * (1.0 - frac))
    m.scatter_add_(1, upper, p * frac)
    return m, clamp_fraction
