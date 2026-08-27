"""Experiment configuration: one frozen dataclass drives a whole run."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import cached_property
from typing import Any


def _utc_stamp() -> str:
    """Return a filesystem-safe UTC timestamp like ``20260827T093015``."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


@dataclass(frozen=True)
class ExperimentConfig:
    """Base configuration shared by every roborl entry point.

    A full experiment is one frozen dataclass: tyro turns it into a CLI, the
    telemetry logger records it verbatim, and the run name is derived from it.
    Future training scripts subclass this and add algorithm hyperparameters.

    Attributes:
        exp_name: Short experiment name, e.g. ``"demo"`` or ``"dqn"``. First
            component of the run name and W&B group.
        env_id: Gymnasium environment id, e.g. ``"CartPole-v1"``.
        seed: Seed for all RNGs (Python, NumPy, torch, environment).
        total_timesteps: Environment steps to run.
        device: ``"auto"`` (cuda > mps > cpu), or an explicit device name.
        track: Whether to log to Weights & Biases. When False, telemetry is a
            no-op and no W&B account is needed.
        wandb_project: W&B project name.
        wandb_entity: W&B entity. ``None`` defers to the ``WANDB_ENTITY``
            environment variable, then to the account default.
        capture_video: Record periodic episode videos (env 0 only).
        video_dir: Where video files are written before upload.
    """

    exp_name: str = "exp"
    env_id: str = "CartPole-v1"
    seed: int = 1
    total_timesteps: int = 10_000
    device: str = "auto"
    track: bool = False
    wandb_project: str = "roborl"
    wandb_entity: str | None = None
    capture_video: bool = False
    video_dir: str = "videos"

    @cached_property
    def run_name(self) -> str:
        """Unique run name: ``{exp_name}-{env_id}-s{seed}-{UTC timestamp}``.

        The timestamp is taken on first access and cached, so one config
        instance keeps one stable run name for its lifetime.
        """
        return f"{self.exp_name}-{self.env_id}-s{self.seed}-{_utc_stamp()}"

    @property
    def group(self) -> str:
        """W&B group: seeds of one experiment share ``{exp_name}-{env_id}``."""
        return f"{self.exp_name}-{self.env_id}"

    @property
    def resolved_wandb_entity(self) -> str | None:
        """Entity to log under: explicit value, else ``WANDB_ENTITY``, else None."""
        return self.wandb_entity or os.environ.get("WANDB_ENTITY") or None

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain dict (for W&B and reports)."""
        d = asdict(self)
        d["run_name"] = self.run_name
        return d
