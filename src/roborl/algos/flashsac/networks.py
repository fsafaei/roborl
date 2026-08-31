"""FlashSAC networks: tanh-Gaussian actor, ensemble categorical critic, temperature.

Shapes follow docs/algos/flashsac.md and are asserted, not trusted: actor
outputs ``(B, A)`` actions and ``(B,)`` log-probs; the critic outputs
``(E, B)`` expected values and ``(E, B, n_atoms)`` log-probabilities. All
forward passes take an explicit ``training`` flag for BatchNorm mode —
rollout and evaluation paths must pass ``False``.

The actor's log-prob carries **no** ``action_scale`` term, so the
environment's action space must be ``[-1, 1]`` (``RescaleAction`` at env
construction; asserted there). Do not copy SAC's rescaling helper.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812 — torch's universal alias
from torch import nn

from roborl.algos.flashsac.layers import (
    Block,
    Embedder,
    EnsembleBlock,
    EnsembleEmbedder,
    EnsembleUnitLinear,
    EnsembleUnitRMSNorm,
    UnitLinear,
    UnitRMSNorm,
    normalize_parameters,
)

LOG_STD_MAX = 2.0
LOG_STD_MIN = -10.0  # not SAC's -5


def safe_tanh_log_det_jacobian(u: torch.Tensor) -> torch.Tensor:
    """Per-dimension ``log(1 - tanh(u)^2)`` in a numerically stable form.

    ``2 * (log 2 - u - softplus(-2u))`` — exact, no epsilon, no clamp; the
    naive form underflows to ``log(0)`` where tanh saturates.

    Args:
        u: Pre-squash Gaussian samples, any shape.

    Returns:
        The log-det-Jacobian of tanh, elementwise, same shape.
    """
    return 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))


def entropy_target(act_dim: int, sigma_tgt: float = 0.15) -> float:
    """Entropy target from a fixed per-dimension action standard deviation.

    Differential entropy of a diagonal Gaussian:
    ``0.5 * A * log(2 * pi * e * sigma_tgt**2)`` — about ``-0.4782 * A`` at
    the default, roughly half as negative as SAC's ``-A``, and the one
    number transfers across action dimensionalities.

    Args:
        act_dim: Action dimensionality ``A``.
        sigma_tgt: Target per-dimension action standard deviation.

    Returns:
        The entropy target.
    """
    return 0.5 * act_dim * math.log(2.0 * math.pi * math.e * sigma_tgt**2)


class FlashSACActor(nn.Module):
    """Residual-BN trunk with a tanh-Gaussian head on unscaled ``[-1, 1]`` actions."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 128,
        num_blocks: int = 2,
        use_rmsnorm: bool = True,
    ) -> None:
        """Build the actor.

        Args:
            obs_dim: Flat observation size.
            act_dim: Flat action size.
            hidden: Trunk width.
            num_blocks: Residual block count.
            use_rmsnorm: Terminal RMSNorm on the trunk (False only on
                ablation-ladder rung 2).
        """
        super().__init__()
        self.act_dim = act_dim
        self.embedder = Embedder(obs_dim, hidden)
        self.blocks = nn.ModuleList(Block(hidden) for _ in range(num_blocks))
        self.out_norm: nn.Module = UnitRMSNorm(hidden) if use_rmsnorm else nn.Identity()
        self.mean_w = UnitLinear(hidden, act_dim)
        self.mean_bias = nn.Parameter(torch.zeros(act_dim))
        self.std_w = UnitLinear(hidden, act_dim)
        self.std_bias = nn.Parameter(torch.zeros(act_dim))

    def _trunk(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        h = self.embedder(x, training)
        for block in self.blocks:
            h = block(h, training)
        out: torch.Tensor = self.out_norm(h)
        return out

    def get_mean_and_std(
        self, x: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the pre-squash Gaussian's ``(mean, std)``, each ``(B, A)``.

        log_std is tanh-squashed into ``[LOG_STD_MIN, LOG_STD_MAX]``, not
        clamped, keeping gradients alive at the bounds.

        Args:
            x: Observations ``(B, obs_dim)``.
            training: BatchNorm mode; False on every rollout/eval path.
        """
        h = self._trunk(x, training)
        mean = self.mean_w(h) + self.mean_bias
        raw = self.std_w(h) + self.std_bias
        log_std = LOG_STD_MIN + (LOG_STD_MAX - LOG_STD_MIN) * 0.5 * (1.0 + torch.tanh(raw))
        assert mean.shape == (x.shape[0], self.act_dim)
        return mean, log_std.exp()

    def forward(self, x: torch.Tensor, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a reparameterised squashed action.

        Args:
            x: Observations ``(B, obs_dim)``.
            training: BatchNorm mode; False on every rollout/eval path.

        Returns:
            ``(action, log_prob)``: ``tanh(u)`` with **no** rescaling,
            shape ``(B, A)``, and its log-density, shape ``(B,)``.
        """
        mean, std = self.get_mean_and_std(x, training)
        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = (dist.log_prob(u) - safe_tanh_log_det_jacobian(u)).sum(dim=-1)
        assert log_prob.shape == (x.shape[0],)
        return action, log_prob

    def eval_action(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic evaluation action ``tanh(mean)``, no sampling."""
        mean, _ = self.get_mean_and_std(x, training=False)
        return torch.tanh(mean)

    def normalize_parameters(self) -> None:
        """Re-project every unit-normalised layer; call after each optimiser step."""
        normalize_parameters(self)


class EnsembleCategoricalValue(nn.Module):
    """Categorical value head on a fixed support; scalar Q is the expectation."""

    bin_values: torch.Tensor

    def __init__(
        self, ensemble_size: int, hidden: int, n_atoms: int, v_min: float, v_max: float
    ) -> None:
        """Build the head.

        Args:
            ensemble_size: Number of members ``E``.
            hidden: Trunk width.
            n_atoms: Number of support atoms ``n``.
            v_min: Lowest atom.
            v_max: Highest atom.
        """
        super().__init__()
        self.w = EnsembleUnitLinear(ensemble_size, hidden, n_atoms)
        self.bias = nn.Parameter(torch.zeros(ensemble_size, n_atoms))
        self.register_buffer("bin_values", torch.linspace(v_min, v_max, n_atoms).view(1, 1, -1))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map features ``(E, B, hidden)`` to ``(q, log_prob)``.

        Returns:
            ``q``: expected values ``(E, B)``; ``log_prob``: atom
            log-probabilities ``(E, B, n_atoms)``.
        """
        logits = self.w(h) + self.bias.unsqueeze(1)
        log_prob = F.log_softmax(logits, dim=-1)
        q = (log_prob.exp() * self.bin_values).sum(dim=-1)
        return q, log_prob


class EnsembleScalarValue(nn.Module):
    """Scalar value head (ablation-ladder rungs 2-3: SAC-style critic + MSE)."""

    def __init__(self, ensemble_size: int, hidden: int) -> None:
        """Build the head.

        Args:
            ensemble_size: Number of members ``E``.
            hidden: Trunk width.
        """
        super().__init__()
        self.w = EnsembleUnitLinear(ensemble_size, hidden, 1)
        self.bias = nn.Parameter(torch.zeros(ensemble_size, 1))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Map features ``(E, B, hidden)`` to ``(q, None)``, ``q`` of shape ``(E, B)``."""
        q = (self.w(h) + self.bias.unsqueeze(1)).squeeze(-1)
        return q, None


class FlashSACDoubleCritic(nn.Module):
    """Twin categorical critics as one ensemble module with a leading member dim."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 256,
        num_blocks: int = 2,
        ensemble_size: int = 2,
        n_atoms: int = 101,
        v_min: float = -5.0,
        v_max: float = 5.0,
        use_rmsnorm: bool = True,
        distributional: bool = True,
    ) -> None:
        """Build the critic ensemble.

        Args:
            obs_dim: Flat observation size.
            act_dim: Flat action size.
            hidden: Trunk width.
            num_blocks: Residual block count.
            ensemble_size: Number of members ``E``.
            n_atoms: Number of support atoms.
            v_min: Lowest atom.
            v_max: Highest atom.
            use_rmsnorm: Terminal RMSNorm on the trunk (False only on
                ablation-ladder rung 2).
            distributional: Categorical head on the fixed support; False
                (ladder rungs 2-3) swaps in a scalar head, and ``forward``
                then returns ``log_prob=None``.
        """
        super().__init__()
        self.ensemble_size = ensemble_size
        self.n_atoms = n_atoms
        self.distributional = distributional
        in_features = obs_dim + act_dim
        self.embedder = EnsembleEmbedder(ensemble_size, in_features, hidden)
        self.blocks = nn.ModuleList(EnsembleBlock(ensemble_size, hidden) for _ in range(num_blocks))
        self.out_norm: nn.Module = (
            EnsembleUnitRMSNorm(ensemble_size, hidden) if use_rmsnorm else nn.Identity()
        )
        self.head: nn.Module = (
            EnsembleCategoricalValue(ensemble_size, hidden, n_atoms, v_min, v_max)
            if distributional
            else EnsembleScalarValue(ensemble_size, hidden)
        )

    @property
    def bin_values(self) -> torch.Tensor:
        """The support atoms, shape ``(1, 1, n_atoms)`` (distributional head only)."""
        assert isinstance(self.head, EnsembleCategoricalValue)
        return self.head.bin_values

    def features(self, obs: torch.Tensor, act: torch.Tensor, training: bool) -> torch.Tensor:
        """Trunk features after the terminal RMSNorm, shape ``(E, B, hidden)``.

        Args:
            obs: Observations ``(B, obs_dim)``.
            act: Actions ``(B, act_dim)``.
            training: BatchNorm mode.
        """
        x = torch.cat([obs, act], dim=-1)
        assert x.ndim == 2
        x = x.unsqueeze(0).expand(self.ensemble_size, *x.shape)
        h = self.embedder(x, training)
        for block in self.blocks:
            h = block(h, training)
        out: torch.Tensor = self.out_norm(h)
        return out

    def forward(
        self, obs: torch.Tensor, act: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Evaluate both members on the same ``(obs, act)`` batch.

        Args:
            obs: Observations ``(B, obs_dim)``.
            act: Actions ``(B, act_dim)``.
            training: BatchNorm mode. The critic loss and target
                construction use True (as a *single* concatenated
                cross-batch call); the critic inside the actor loss uses
                False — that asymmetry is deliberate.

        Returns:
            ``(q, log_prob)`` with shapes ``(E, B)`` and ``(E, B, n_atoms)``;
            ``log_prob`` is None with the scalar (non-distributional) head.
        """
        q, log_prob = self.head(self.features(obs, act, training))
        assert q.shape == (self.ensemble_size, obs.shape[0])
        if self.distributional:
            assert log_prob is not None
            assert log_prob.shape == (self.ensemble_size, obs.shape[0], self.n_atoms)
        return q, log_prob

    def normalize_parameters(self) -> None:
        """Re-project every unit-normalised layer; call after each optimiser step."""
        normalize_parameters(self)


class Temperature(nn.Module):
    """Learned entropy temperature ``alpha = exp(log_temp)``."""

    def __init__(self, alpha_init: float = 0.01) -> None:
        """Build the module.

        Args:
            alpha_init: Initial temperature; ``log_temp`` starts at its log.
        """
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(math.log(alpha_init)))

    def forward(self) -> torch.Tensor:
        """Return the scalar temperature ``alpha``."""
        alpha: torch.Tensor = self.log_temp.exp()
        return alpha
