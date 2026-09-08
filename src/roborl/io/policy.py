"""Policy checkpoints: one documented, versioned format (ADR 0009).

A checkpoint is a *policy artefact*: the trained actor's weights plus what
is needed to rebuild it and act deterministically. It is not a
resume-training snapshot (no critics, optimisers, or replay buffers).
:func:`save_policy` writes one ``.pt`` file; :func:`load_policy` rebuilds
the actor from the spec stored inside and returns a :class:`LoadedPolicy`
whose ``act`` maps observations to actions in the environment's own units.
concerto and robolab only ever call ``load_policy``.

The file is a plain dict of tensors and primitives, read with
``torch.load(weights_only=True)``: a checkpoint is data, never code.

Deterministic action per actor, mirroring each loop's evaluation path:

* ``sac``: ``tanh(mean) * action_scale + action_bias``; the
  rescaling buffers ship inside the state dict.
* ``flashsac``: ``eval_action`` gives ``tanh(mean)`` in ``[-1, 1]``; the
  loop trains behind ``RescaleAction``, so the same affine map onto the
  true bounds is applied here.
* ``ppo_continuous``: observations pass through the frozen
  :class:`ObsNormalizer` (gymnasium's ``NormalizeObservation`` statistics
  at save time, then the clip to ``[-10, 10]``); ``actor_mean`` is clipped
  to the bounds the way ``ClipAction`` does.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from roborl import __version__
from roborl.config import ExperimentConfig
from roborl.telemetry.logger import git_provenance
from roborl.utils.device import resolve_device

FORMAT = "roborl.policy"
FORMAT_VERSION = 1

Algo = Literal["sac", "flashsac", "ppo_continuous"]
ALGOS: tuple[str, ...] = ("sac", "flashsac", "ppo_continuous")
ARCH_KEYS: dict[str, frozenset[str]] = {
    "sac": frozenset(),
    "flashsac": frozenset({"hidden", "num_blocks", "use_rmsnorm"}),
    "ppo_continuous": frozenset(),
}
"""Constructor arguments beyond ``(obs_dim, act_dim)`` that each actor needs."""


@dataclass(frozen=True)
class PolicySpec:
    """What :func:`load_policy` needs to rebuild an actor and interpret its output.

    Attributes:
        algo: Which actor class and which deterministic-action rule apply.
        obs_dim: Flat observation size the actor was trained on.
        act_dim: Flat action size.
        action_low: Lower action bounds in environment units, ``act_dim`` long.
        action_high: Upper action bounds, likewise. ``act()`` returns actions
            inside ``[action_low, action_high]``.
        arch: Constructor arguments beyond the two sizes, exactly the keys
            ``ARCH_KEYS[algo]`` (e.g. ``hidden`` for ``flashsac``).
    """

    algo: Algo
    obs_dim: int
    act_dim: int
    action_low: tuple[float, ...]
    action_high: tuple[float, ...]
    arch: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate sizes, bounds and arch keys; normalise bounds and arch to tuples."""
        if self.algo not in ALGOS:
            raise ValueError(f"unknown algo {self.algo!r}; known: {ALGOS}")
        if self.obs_dim <= 0 or self.act_dim <= 0:
            raise ValueError(f"obs_dim/act_dim must be positive, got {self.obs_dim}/{self.act_dim}")
        low = np.asarray(self.action_low, dtype=np.float64)
        high = np.asarray(self.action_high, dtype=np.float64)
        if low.shape != (self.act_dim,) or high.shape != (self.act_dim,):
            raise ValueError(f"action bounds must have length act_dim={self.act_dim}")
        if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high)) and np.all(low < high)):
            raise ValueError("action bounds must be finite with low < high on every dimension")
        expected = ARCH_KEYS[self.algo]
        if set(self.arch) != expected:
            raise ValueError(
                f"arch for {self.algo!r} needs exactly the keys {sorted(expected)}, "
                f"got {sorted(self.arch)}"
            )
        object.__setattr__(self, "action_low", tuple(float(x) for x in low))
        object.__setattr__(self, "action_high", tuple(float(x) for x in high))
        arch = {k: (tuple(v) if isinstance(v, list | tuple) else v) for k, v in self.arch.items()}
        object.__setattr__(self, "arch", arch)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe dict (tuples become lists) for the checkpoint payload."""
        return {
            "algo": self.algo,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "action_low": list(self.action_low),
            "action_high": list(self.action_high),
            "arch": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.arch.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PolicySpec:
        """Inverse of :meth:`to_dict`; ``ValueError`` on anything malformed."""
        try:
            return cls(
                algo=d["algo"],
                obs_dim=int(d["obs_dim"]),
                act_dim=int(d["act_dim"]),
                action_low=tuple(d["action_low"]),
                action_high=tuple(d["action_high"]),
                arch=dict(d["arch"]),
            )
        except (KeyError, TypeError) as e:
            raise ValueError(f"malformed policy spec: {e}") from e


@dataclass(frozen=True, eq=False)
class ObsNormalizer:
    """Frozen observation normalisation: ``(obs - mean) / sqrt(var + epsilon)``, then a clip.

    ``ppo_continuous`` trains behind gymnasium's ``NormalizeObservation``
    (running statistics kept by the env wrapper, not by the model) followed
    by a ``TransformObservation`` clip. The statistics at save time are
    frozen here so a deployed policy sees inputs the way the network did.

    Attributes:
        mean: Per-feature running mean, ``(obs_dim,)``.
        var: Per-feature running variance, ``(obs_dim,)``.
        epsilon: The wrapper's stability constant.
        clip: Symmetric clip applied after normalisation, or None.
    """

    mean: np.ndarray
    var: np.ndarray
    epsilon: float
    clip: float | None = None

    def __post_init__(self) -> None:
        """Validate shapes and positivity; store float64 copies."""
        mean = np.array(self.mean, dtype=np.float64)
        var = np.array(self.var, dtype=np.float64)
        if mean.ndim != 1 or mean.shape != var.shape:
            raise ValueError("mean and var must be 1-D arrays of the same shape")
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(var)) and np.all(var >= 0)):
            raise ValueError("mean must be finite and var finite and non-negative")
        if not self.epsilon > 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if self.clip is not None and not self.clip > 0:
            raise ValueError(f"clip must be > 0 or None, got {self.clip}")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "var", var)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Normalise (and clip) observations; broadcasts over a leading batch axis."""
        out = (np.asarray(obs, dtype=np.float64) - self.mean) / np.sqrt(self.var + self.epsilon)
        if self.clip is not None:
            out = np.clip(out, -self.clip, self.clip)
        result: np.ndarray = out.astype(np.float32)
        return result


@dataclass(eq=False)
class LoadedPolicy:
    """A checkpoint rebuilt into an actor that acts deterministically.

    Attributes:
        spec: The stored :class:`PolicySpec`.
        metadata: Provenance written by :func:`policy_metadata` (plus anything
            the saver added).
        sha256: Hex digest of the file that was loaded.
        actor: The rebuilt actor, in eval mode, gradients off, on ``device``.
        obs_normalizer: Frozen input normalisation, or None.
        device: Where the actor lives.
    """

    spec: PolicySpec
    metadata: dict[str, Any]
    sha256: str
    actor: nn.Module
    obs_normalizer: ObsNormalizer | None
    device: torch.device

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action(s) in environment units.

        Args:
            obs: One observation ``(obs_dim,)`` or a batch ``(B, obs_dim)``.

        Returns:
            ``(act_dim,)`` or ``(B, act_dim)`` float32, inside the spec's bounds.

        Raises:
            ValueError: On a wrong observation size or non-finite values.
        """
        x = np.asarray(obs, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.spec.obs_dim:
            raise ValueError(f"expected observations of size {self.spec.obs_dim}, got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("observation contains NaN or inf")
        if self.obs_normalizer is not None:
            x = self.obs_normalizer(x)
        with torch.inference_mode():
            a = _deterministic_action(self.spec, self.actor, torch.as_tensor(x, device=self.device))
        out = a.detach().cpu().numpy().astype(np.float32)
        return out[0] if single else out

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Alias for :meth:`act`."""
        return self.act(obs)


# --- helpers for the training loops ------------------------------------------


def action_bounds(space: gym.Space[Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """``(low, high)`` of a flat ``Box`` action space as tuples of floats."""
    if not isinstance(space, gym.spaces.Box):
        raise TypeError(f"policy checkpoints need a Box action space, got {space}")
    low = tuple(float(x) for x in np.asarray(space.low).reshape(-1))
    high = tuple(float(x) for x in np.asarray(space.high).reshape(-1))
    return low, high


def policy_metadata(
    config: ExperimentConfig, *, global_step: int, resolved_device: str
) -> dict[str, Any]:
    """Provenance stored beside the weights: what trained them, where, and when."""
    return {
        "roborl_version": __version__,
        **git_provenance(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "gymnasium_version": gym.__version__,
        "resolved_device": resolved_device,
        "env_id": config.env_id,
        "seed": config.seed,
        "global_step": int(global_step),
        "exp_name": config.exp_name,
        "run_name": config.run_name,
        "config": config.to_dict(),
    }


def sha256_of(path: str | Path) -> str:
    """SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- rebuild and act ---------------------------------------------------------


def build_actor(spec: PolicySpec) -> nn.Module:
    """A freshly initialised actor of the class and shape ``spec`` describes.

    Imports the algorithm package lazily: ``roborl.io`` never depends on an
    algorithm at import time, while every loop imports ``roborl.io``.
    """
    low = np.asarray(spec.action_low, dtype=np.float32)
    high = np.asarray(spec.action_high, dtype=np.float32)
    if spec.algo == "sac":
        from roborl.algos.sac.sac import Actor as SacActor

        return SacActor(spec.obs_dim, spec.act_dim, low, high)
    if spec.algo == "flashsac":
        from roborl.algos.flashsac.networks import FlashSACActor

        return FlashSACActor(
            spec.obs_dim,
            spec.act_dim,
            hidden=int(spec.arch["hidden"]),
            num_blocks=int(spec.arch["num_blocks"]),
            use_rmsnorm=bool(spec.arch["use_rmsnorm"]),
        )
    from roborl.algos.ppo.ppo_continuous import Agent

    return Agent(spec.obs_dim, spec.act_dim)


def _check_actor_bounds(spec: PolicySpec, actor: nn.Module) -> None:
    """SAC-style actors carry rescaling buffers; they must agree with the spec's bounds."""
    if spec.algo != "sac":
        return
    scale = actor.get_buffer("action_scale").detach().cpu().numpy()
    bias = actor.get_buffer("action_bias").detach().cpu().numpy()
    low, high = np.asarray(spec.action_low), np.asarray(spec.action_high)
    if not (
        np.allclose(scale, (high - low) / 2.0, atol=1e-6)
        and np.allclose(bias, (high + low) / 2.0, atol=1e-6)
    ):
        raise ValueError(
            "the actor's action_scale/action_bias buffers disagree with the spec's action bounds"
        )


def _deterministic_action(spec: PolicySpec, actor: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The evaluation-time action of each actor family (see the module docstring)."""
    low = torch.as_tensor(spec.action_low, dtype=torch.float32, device=x.device)
    high = torch.as_tensor(spec.action_high, dtype=torch.float32, device=x.device)
    if spec.algo == "sac":
        mean, _log_std = actor(x)
        squashed: torch.Tensor = torch.tanh(mean) * actor.get_buffer("action_scale")
        return squashed + actor.get_buffer("action_bias")
    if spec.algo == "flashsac":
        from roborl.algos.flashsac.networks import FlashSACActor

        assert isinstance(actor, FlashSACActor)
        unit = actor.eval_action(x)
        return low + (unit + 1.0) * 0.5 * (high - low)
    from roborl.algos.ppo.ppo_continuous import Agent

    assert isinstance(actor, Agent)
    mean_action: torch.Tensor = actor.actor_mean(x)
    return torch.maximum(torch.minimum(mean_action, high), low)


# --- save / load ---------------------------------------------------------------


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set | frozenset):
        return sorted(o)
    return str(o)


def _jsonable(obj: Mapping[str, Any]) -> dict[str, Any]:
    """Round-trip through JSON so only primitives reach the file (``weights_only``-safe)."""
    out: dict[str, Any] = json.loads(json.dumps(dict(obj), default=_json_default))
    return out


def save_policy(
    path: str | Path,
    actor: nn.Module,
    spec: PolicySpec,
    *,
    metadata: Mapping[str, Any],
    obs_normalizer: ObsNormalizer | None = None,
) -> str:
    """Write a policy checkpoint and return its SHA-256 hex digest.

    The weights are test-loaded into an actor rebuilt from ``spec`` before
    anything is written: a spec/actor mismatch surfaces here, not on a robot.

    Args:
        path: Destination file (parents are created). ``.pt`` by convention.
        actor: The trained actor (any device).
        spec: Its :class:`PolicySpec`.
        metadata: Provenance; see :func:`policy_metadata`. Reduced to JSON
            primitives, so anything goes but only primitives come back.
        obs_normalizer: Required for ``ppo_continuous``, forbidden otherwise.

    Raises:
        ValueError: If the weights do not fit ``spec``, the SAC-style
            rescaling buffers disagree with the bounds, or the normaliser is
            missing/unexpected or the wrong size.
    """
    if (obs_normalizer is not None) != (spec.algo == "ppo_continuous"):
        raise ValueError("obs_normalizer is required for ppo_continuous and must be None otherwise")
    if obs_normalizer is not None and obs_normalizer.mean.shape != (spec.obs_dim,):
        raise ValueError(
            f"obs_normalizer has {obs_normalizer.mean.shape}, spec needs ({spec.obs_dim},)"
        )
    state = {k: v.detach().to("cpu").clone() for k, v in actor.state_dict().items()}
    probe = build_actor(spec)
    try:
        probe.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise ValueError(f"actor weights do not fit an actor rebuilt from {spec}: {e}") from e
    _check_actor_bounds(spec, probe)
    normalizer_payload = (
        None
        if obs_normalizer is None
        else {
            "mean": torch.as_tensor(obs_normalizer.mean, dtype=torch.float64),
            "var": torch.as_tensor(obs_normalizer.var, dtype=torch.float64),
            "epsilon": float(obs_normalizer.epsilon),
            "clip": None if obs_normalizer.clip is None else float(obs_normalizer.clip),
        }
    )
    payload: dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "spec": spec.to_dict(),
        "metadata": _jsonable(metadata),
        "actor_state_dict": state,
        "obs_normalizer": normalizer_payload,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)
    return sha256_of(p)


def load_policy(path: str | Path, *, device: str = "cpu") -> LoadedPolicy:
    """Read a checkpoint written by :func:`save_policy`.

    Args:
        path: The ``.pt`` file.
        device: ``"cpu"`` (default), ``"cuda"``, ``"mps"`` or ``"auto"``; an
            unavailable explicit device raises rather than falling back.

    Returns:
        A :class:`LoadedPolicy` ready to ``act``.

    Raises:
        ValueError: If the file is not a roborl policy checkpoint, has a
            format version this roborl does not read, or is internally
            inconsistent (spec vs. weights, bounds, normaliser).
    """
    p = Path(path)
    try:
        payload = torch.load(p, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, zipfile.BadZipFile) as e:
        raise ValueError(f"{p}: not a roborl policy checkpoint ({type(e).__name__}: {e})") from e
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError(f"{p}: not a roborl policy checkpoint")
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{p}: format version {version!r}; this roborl reads version {FORMAT_VERSION}"
        )
    spec = PolicySpec.from_dict(payload["spec"])
    actor = build_actor(spec)
    try:
        actor.load_state_dict(payload["actor_state_dict"], strict=True)
    except (RuntimeError, KeyError, TypeError) as e:
        raise ValueError(f"{p}: weights do not fit spec {spec}: {e}") from e
    _check_actor_bounds(spec, actor)
    raw = payload.get("obs_normalizer")
    if (raw is not None) != (spec.algo == "ppo_continuous"):
        raise ValueError(f"{p}: obs_normalizer presence does not match algo {spec.algo!r}")
    normalizer = (
        None
        if raw is None
        else ObsNormalizer(
            mean=raw["mean"].numpy(),
            var=raw["var"].numpy(),
            epsilon=float(raw["epsilon"]),
            clip=None if raw["clip"] is None else float(raw["clip"]),
        )
    )
    if normalizer is not None and normalizer.mean.shape != (spec.obs_dim,):
        raise ValueError(f"{p}: obs_normalizer size {normalizer.mean.shape} != ({spec.obs_dim},)")
    dev = resolve_device(device)
    actor.to(dev).eval()
    for param in actor.parameters():
        param.requires_grad_(False)
    return LoadedPolicy(
        spec=spec,
        metadata=dict(payload.get("metadata") or {}),
        sha256=sha256_of(p),
        actor=actor,
        obs_normalizer=normalizer,
        device=dev,
    )
