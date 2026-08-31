"""FlashSAC primitive layers and blocks (docs/algos/flashsac.md).

Three unit-normalised primitives — bias-free linear, BatchNorm, RMSNorm —
each with a ``normalize_parameters()`` hook that re-projects its parameters
onto a fixed-norm manifold *after* every optimiser step (in-place, under
``no_grad``; inside the graph it would corrupt Adam's state). Ensemble
variants carry a leading member dimension so the twin critics are one
module, never two. The norm constraints are the paper's mechanism for
bounding critic error accumulation; they are load-bearing, not decoration.

BatchNorm caveats that matter here (see the pitfall catalogue in the algo
doc): momentum 0.01 is *PyTorch's* convention (``new = 0.99*old +
0.01*batch``) — a value ported from JAX/Flax means the complementary
convention and an almost-frozen normaliser. ``[gamma; beta]`` are
normalised *jointly* to L2 norm ``sqrt(d)``, not separately.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812 — torch's universal alias
from torch import nn

if not hasattr(F, "rms_norm"):  # pragma: no cover — torch >= 2.4 is pinned
    raise ImportError(
        "FlashSAC requires torch >= 2.4 for torch.nn.functional.rms_norm; "
        f"found torch {torch.__version__}. Upgrade torch rather than falling "
        "back to a hand-rolled RMSNorm."
    )

_NORM_EPS = 1e-8


class UnitNormalizedLayer(nn.Module):
    """Base for layers that re-project their parameters after each optimiser step."""

    def normalize_parameters(self) -> None:
        """Re-project parameters onto the layer's fixed-norm manifold, in place."""
        raise NotImplementedError


def normalize_parameters(module: nn.Module) -> None:
    """Call ``normalize_parameters()`` on every unit-normalised layer in ``module``.

    Must run after ``optimizer.step()``; each hook works in place under
    ``no_grad`` so Adam's state is untouched.

    Args:
        module: Root module whose tree is walked.
    """
    for child in module.modules():
        if isinstance(child, UnitNormalizedLayer):
            child.normalize_parameters()


class UnitLinear(UnitNormalizedLayer):
    """Bias-free linear with orthogonal init and unit-norm weight rows.

    ``normalize_parameters()`` rescales every *output* row to unit L2 norm
    over its input features.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """Build the layer.

        Args:
            in_features: Input feature count.
            out_features: Output feature count.
        """
        super().__init__()
        self.w = nn.Linear(in_features, out_features, bias=False)
        nn.init.orthogonal_(self.w.weight, gain=1)

    @property
    def weight(self) -> torch.Tensor:
        """The ``(out_features, in_features)`` weight matrix."""
        weight: torch.Tensor = self.w.weight
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear map."""
        out: torch.Tensor = self.w(x)
        return out

    def normalize_parameters(self) -> None:
        """Rescale each output row to unit L2 norm."""
        with torch.no_grad():
            self.w.weight.copy_(F.normalize(self.w.weight, dim=-1, eps=_NORM_EPS))


class UnitBatchNorm(UnitNormalizedLayer):
    """BatchNorm1d semantics with jointly-normalised affine parameters.

    Momentum 0.01 in PyTorch's convention; ``normalize_parameters()``
    rescales the concatenation ``[gamma; beta]`` to L2 norm ``sqrt(d)``.
    """

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, num_features: int, momentum: float = 0.01, eps: float = 1e-5) -> None:
        """Build the layer.

        Args:
            num_features: Feature count ``d``.
            momentum: Running-stat update weight on the batch statistic.
            eps: Variance floor inside the normalisation.
        """
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Normalise ``x`` of shape ``(B, d)``.

        Args:
            x: Input batch.
            training: True uses (and updates) batch statistics; False uses
                the running statistics. Rollout and evaluation paths must
                pass False — a train-mode BatchNorm over a batch of one has
                zero variance and poisons the running statistics.

        Returns:
            The normalised batch, same shape.
        """
        assert x.shape[-1] == self.num_features
        out: torch.Tensor = F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training,
            self.momentum,
            self.eps,
        )
        return out

    def normalize_parameters(self) -> None:
        """Rescale ``[gamma; beta]`` jointly to L2 norm ``sqrt(d)``."""
        with torch.no_grad():
            _joint_affine_rescale(self.weight, self.bias, self.num_features)


class UnitRMSNorm(UnitNormalizedLayer):
    """RMSNorm whose weight is held at L2 norm ``sqrt(d)``."""

    def __init__(self, num_features: int, eps: float = 1e-6) -> None:
        """Build the layer.

        Args:
            num_features: Feature count ``d``.
            eps: RMS floor.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalisation over the last dimension."""
        out: torch.Tensor = F.rms_norm(x, self.weight.shape, self.weight, self.eps)
        return out

    def normalize_parameters(self) -> None:
        """Rescale the weight to L2 norm ``sqrt(d)``."""
        with torch.no_grad():
            sqsum = self.weight.pow(2).sum()
            self.weight.mul_(math.sqrt(self.num_features) * torch.rsqrt(sqsum + _NORM_EPS))


def _joint_affine_rescale(weight: torch.Tensor, bias: torch.Tensor, num_features: int) -> None:
    """Rescale ``[gamma; beta]`` jointly (per ensemble member) to norm ``sqrt(d)``.

    The sum of squares runs over *both* vectors together — beta starts at
    zeros, so after the first update it is rescaled to a fixed joint norm
    regardless of its own magnitude; that is genuinely what the reference
    implementation does.
    """
    sqsum = (weight.pow(2) + bias.pow(2)).sum(dim=-1, keepdim=True)
    factor = math.sqrt(num_features) * torch.rsqrt(sqsum + _NORM_EPS)
    weight.mul_(factor)
    bias.mul_(factor)


class Embedder(nn.Module):
    """Input embedding: BatchNorm *first*, then a unit linear.

    The leading BatchNorm over the raw input is FlashSAC's observation
    normaliser — there is no separate running obs-normalisation wrapper.
    """

    def __init__(self, in_features: int, hidden: int) -> None:
        """Build the block.

        Args:
            in_features: Raw input size.
            hidden: Embedding width.
        """
        super().__init__()
        self.norm = UnitBatchNorm(in_features)
        self.w = UnitLinear(in_features, hidden)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Embed ``(B, in_features)`` to ``(B, hidden)``."""
        out: torch.Tensor = self.w(self.norm(x, training))
        return out


class Block(nn.Module):
    """Inverted-bottleneck residual block: ``x + relu(BN(w2 relu(BN(w1 x))))``.

    No activation after the residual add.
    """

    def __init__(self, hidden: int, expansion: int = 4) -> None:
        """Build the block.

        Args:
            hidden: Block width.
            expansion: Bottleneck expansion factor.
        """
        super().__init__()
        self.w1 = UnitLinear(hidden, hidden * expansion)
        self.n1 = UnitBatchNorm(hidden * expansion)
        self.w2 = UnitLinear(hidden * expansion, hidden)
        self.n2 = UnitBatchNorm(hidden)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Apply the residual block, preserving shape ``(B, hidden)``."""
        residual = x
        x = F.relu(self.n1(self.w1(x), training))
        x = F.relu(self.n2(self.w2(x), training))
        return x + residual


class EnsembleUnitLinear(UnitNormalizedLayer):
    """Bias-free linear over a leading ensemble dimension, per-member orthogonal init."""

    def __init__(self, ensemble_size: int, in_features: int, out_features: int) -> None:
        """Build the layer.

        Args:
            ensemble_size: Number of members ``E``.
            in_features: Input feature count.
            out_features: Output feature count.
        """
        super().__init__()
        weight = torch.empty(ensemble_size, out_features, in_features)
        for member in range(ensemble_size):
            nn.init.orthogonal_(weight[member], gain=1)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(E, B, in)`` to ``(E, B, out)``, each member with its own weight."""
        assert x.ndim == 3 and x.shape[0] == self.weight.shape[0]
        return torch.einsum("nbi,noi->nbo", x, self.weight)

    def normalize_parameters(self) -> None:
        """Rescale each member's output rows to unit L2 norm."""
        with torch.no_grad():
            self.weight.copy_(F.normalize(self.weight, dim=-1, eps=_NORM_EPS))


class EnsembleUnitBatchNorm(UnitNormalizedLayer):
    """Per-member BatchNorm: weight/bias/running stats all ``(E, d)``.

    Implemented as one flattened ``F.batch_norm`` over ``E * d`` channels,
    which computes exactly per-member per-feature batch statistics.
    """

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(
        self, ensemble_size: int, num_features: int, momentum: float = 0.01, eps: float = 1e-5
    ) -> None:
        """Build the layer.

        Args:
            ensemble_size: Number of members ``E``.
            num_features: Feature count ``d``.
            momentum: Running-stat update weight on the batch statistic.
            eps: Variance floor inside the normalisation.
        """
        super().__init__()
        self.ensemble_size = ensemble_size
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ensemble_size, num_features))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, num_features))
        self.register_buffer("running_mean", torch.zeros(ensemble_size, num_features))
        self.register_buffer("running_var", torch.ones(ensemble_size, num_features))

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Normalise ``(E, B, d)`` with member-independent statistics."""
        ensemble_size, batch, num_features = x.shape
        assert ensemble_size == self.ensemble_size and num_features == self.num_features
        flat = x.transpose(0, 1).reshape(batch, ensemble_size * num_features)
        out = F.batch_norm(
            flat,
            self.running_mean.view(-1),  # views share storage: in-place stat updates land
            self.running_var.view(-1),
            self.weight.view(-1),
            self.bias.view(-1),
            training,
            self.momentum,
            self.eps,
        )
        return out.view(batch, ensemble_size, num_features).transpose(0, 1)

    def normalize_parameters(self) -> None:
        """Rescale each member's ``[gamma; beta]`` jointly to L2 norm ``sqrt(d)``."""
        with torch.no_grad():
            _joint_affine_rescale(self.weight, self.bias, self.num_features)


class EnsembleUnitRMSNorm(UnitNormalizedLayer):
    """Per-member RMSNorm; each member's weight held at L2 norm ``sqrt(d)``."""

    def __init__(self, ensemble_size: int, num_features: int, eps: float = 1e-6) -> None:
        """Build the layer.

        Args:
            ensemble_size: Number of members ``E``.
            num_features: Feature count ``d``.
            eps: RMS floor.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ensemble_size, num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalisation over the last dim of ``(E, B, d)``, per member."""
        assert x.ndim == 3 and x.shape[0] == self.weight.shape[0]
        normalized = F.rms_norm(x, (self.num_features,), None, self.eps)
        return normalized * self.weight.unsqueeze(1)

    def normalize_parameters(self) -> None:
        """Rescale each member's weight to L2 norm ``sqrt(d)``."""
        with torch.no_grad():
            sqsum = self.weight.pow(2).sum(dim=-1, keepdim=True)
            self.weight.mul_(math.sqrt(self.num_features) * torch.rsqrt(sqsum + _NORM_EPS))


class EnsembleEmbedder(nn.Module):
    """Ensemble input embedding: per-member BatchNorm first, then a unit linear."""

    def __init__(self, ensemble_size: int, in_features: int, hidden: int) -> None:
        """Build the block.

        Args:
            ensemble_size: Number of members ``E``.
            in_features: Raw input size.
            hidden: Embedding width.
        """
        super().__init__()
        self.norm = EnsembleUnitBatchNorm(ensemble_size, in_features)
        self.w = EnsembleUnitLinear(ensemble_size, in_features, hidden)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Embed ``(E, B, in_features)`` to ``(E, B, hidden)``."""
        out: torch.Tensor = self.w(self.norm(x, training))
        return out


class EnsembleBlock(nn.Module):
    """Ensemble inverted-bottleneck residual block; see :class:`Block`."""

    def __init__(self, ensemble_size: int, hidden: int, expansion: int = 4) -> None:
        """Build the block.

        Args:
            ensemble_size: Number of members ``E``.
            hidden: Block width.
            expansion: Bottleneck expansion factor.
        """
        super().__init__()
        self.w1 = EnsembleUnitLinear(ensemble_size, hidden, hidden * expansion)
        self.n1 = EnsembleUnitBatchNorm(ensemble_size, hidden * expansion)
        self.w2 = EnsembleUnitLinear(ensemble_size, hidden * expansion, hidden)
        self.n2 = EnsembleUnitBatchNorm(ensemble_size, hidden)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Apply the residual block, preserving shape ``(E, B, hidden)``."""
        residual = x
        x = F.relu(self.n1(self.w1(x), training))
        x = F.relu(self.n2(self.w2(x), training))
        return x + residual
