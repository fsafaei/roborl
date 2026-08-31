"""Temporally-correlated exploration noise (docs/algos/flashsac.md).

Exploration noise is held constant for a run of ``k`` action selections,
with ``k`` drawn from a Zeta (power-law) distribution ``pmf(k) ∝ k**-mu``
truncated to ``{1..k_max}``, sampled by inverse-CDF on a precomputed
cumulative table. The noise tensor is per-environment; the run length is
shared across environments (reference semantics — moot with one env, kept
so the vectorised phase matches). This noise is used **only for acting**:
the log-probabilities in the losses come from fresh ``rsample()`` draws
inside the update, unaffected by repetition.
"""

from __future__ import annotations

import torch


def truncated_zeta_pmf(mu: float = 2.0, k_max: int = 16) -> torch.Tensor:
    """The normalised pmf of the truncated Zeta distribution over ``1..k_max``.

    Args:
        mu: Power-law exponent.
        k_max: Truncation point (inclusive).

    Returns:
        Probabilities for ``k = 1..k_max``, shape ``(k_max,)``, summing to 1.
    """
    k = torch.arange(1, k_max + 1, dtype=torch.float64)
    pmf = k**-mu
    return pmf / pmf.sum()


class NoiseRepeater:
    """Per-env Gaussian noise, redrawn every Zeta-distributed number of steps.

    Uses torch's global RNG (seeded by ``seed_everything``) for both the
    run-length draw and the noise itself, so same-seed runs repeat exactly.
    """

    def __init__(self, num_envs: int, act_dim: int, mu: float = 2.0, k_max: int = 16) -> None:
        """Build the repeater.

        Args:
            num_envs: Number of parallel environments.
            act_dim: Action dimensionality.
            mu: Zeta exponent.
            k_max: Zeta truncation point.
        """
        self._cdf = truncated_zeta_pmf(mu, k_max).cumsum(dim=0)
        self.noise = torch.zeros(num_envs, act_dim)
        self.count = 0
        self.run_length = 0

    def sample_run_length(self) -> int:
        """Draw one run length by inverse-CDF on the precomputed table."""
        u = torch.rand((), dtype=torch.float64)
        return int(torch.searchsorted(self._cdf, u).item()) + 1

    def next(self) -> torch.Tensor:
        """Return the noise for one action selection, redrawing when the run ends.

        Returns:
            The current noise, shape ``(num_envs, act_dim)``.
        """
        if self.count == 0 or self.count >= self.run_length:
            self.noise = torch.randn_like(self.noise)
            self.run_length = self.sample_run_length()
            self.count = 0
        self.count += 1
        return self.noise
