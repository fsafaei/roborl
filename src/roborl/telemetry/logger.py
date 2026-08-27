"""A thin Weights & Biases wrapper with offline and disabled modes.

Three modes, chosen by config and environment:

- **online** — ``track=True``: metrics stream to W&B.
- **offline** — ``track=True`` with ``WANDB_MODE=offline``: metrics are
  written locally under ``wandb/`` and synced later with ``wandb sync``.
- **disabled** — ``track=False``: every call is a no-op; no W&B import cost,
  no account needed, nothing written. Tests and CI always run in this mode.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any

from roborl.config import ExperimentConfig
from roborl.telemetry import metrics


def git_provenance() -> dict[str, Any]:
    """Return the current git SHA and dirty flag, or placeholders outside a repo."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = (
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
            != ""
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": "unknown", "git_dirty": None}
    return {"git_sha": sha, "git_dirty": dirty}


def _provenance(resolved_device: str) -> dict[str, Any]:
    """Collect the metadata that makes a run traceable to code and machine."""
    import gymnasium
    import torch

    return {
        **git_provenance(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "gymnasium_version": gymnasium.__version__,
        "resolved_device": resolved_device,
        "hostname": platform.node(),
        "platform": sys.platform,
    }


class RunLogger:
    """Run-scoped telemetry: ``start`` once, ``log`` per step, ``finish`` once.

    When ``config.track`` is False the logger is a pure no-op, so callers
    never branch on tracking themselves.
    """

    def __init__(self, config: ExperimentConfig, resolved_device: str) -> None:
        """Store the config; nothing happens until :meth:`start`.

        Args:
            config: The experiment configuration to record.
            resolved_device: The device the run actually uses (post
                ``resolve_device``), recorded as provenance.
        """
        self._config = config
        self._resolved_device = resolved_device
        self._run: Any = None

    @property
    def url(self) -> str | None:
        """The W&B run URL, or None when not tracking / offline."""
        if self._run is None:
            return None
        url: str | None = self._run.get_url()
        return url

    def start(self) -> None:
        """Initialize the W&B run (no-op when ``track`` is False).

        Records the full config plus provenance (git SHA + dirty flag,
        Python/torch/gymnasium versions, resolved device, hostname) and
        declares ``global_step`` as the x-axis for every metric namespace.
        """
        if not self._config.track:
            return
        import wandb

        self._run = wandb.init(
            project=self._config.wandb_project,
            entity=self._config.resolved_wandb_entity,
            group=self._config.group,
            name=self._config.run_name,
            config={**self._config.to_dict(), **_provenance(self._resolved_device)},
        )
        self._run.define_metric(metrics.GLOBAL_STEP)
        for prefix in ("charts", "losses", "diagnostics", "eval"):
            self._run.define_metric(f"{prefix}/*", step_metric=metrics.GLOBAL_STEP)

    def log(self, data: dict[str, Any], step: int) -> None:
        """Log a metric dict at ``global_step == step`` (no-op when disabled).

        Args:
            data: Metric name → value. Use the constants in
                :mod:`roborl.telemetry.metrics`; never hand-type names.
            step: The environment step the values belong to.
        """
        if self._run is None:
            return
        self._run.log({**data, metrics.GLOBAL_STEP: step})

    def finish(self) -> None:
        """Close the W&B run (no-op when disabled). Safe to call twice."""
        if self._run is not None:
            self._run.finish()
            self._run = None
