"""Reference-curve sources for verification.

Reference sources are pluggable adapters that all produce the same shape of
data: a DataFrame with one row per logged episode and columns ``run_id``,
``global_step``, ``episodic_return``. The first adapter pulls CleanRL's
public benchmark runs from the ``openrlbenchmark`` W&B entity; later adapters
(re-run CleanRL locally, SB3 zoo, published tables) implement the same
protocol.

Requires the ``benchmark`` extra (``uv sync --extra benchmark``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from roborl.benchmark.stats import FloatArray

COLUMNS = ("run_id", "global_step", "episodic_return")
DEFAULT_CACHE_DIR = Path(".cache/benchref")


class ReferenceAdapter(Protocol):
    """A source of reference learning curves for one algorithm on one env."""

    def fetch(self, algo: str, env_id: str, force: bool = False) -> pd.DataFrame:
        """Return reference curves as a ``(run_id, global_step, episodic_return)`` frame.

        Args:
            algo: Algorithm name as the source spells it (e.g. ``"sac_continuous_action"``).
            env_id: Environment id as the source ran it (env-version parity:
                verify on the version the reference used).
            force: Bypass any local cache and re-download.
        """
        ...


class OpenRLBenchmarkAdapter:
    """CleanRL's public benchmark runs, via the W&B public API.

    Runs live under the ``openrlbenchmark/cleanrl`` W&B project; an
    algorithm's runs are identified by ``config.exp_name`` (CleanRL's script
    name, e.g. ``"sac_continuous_action"``) and ``config.env_id``. Downloads are cached as
    parquet under ``.cache/benchref/`` so every later use is offline.
    """

    def __init__(
        self,
        entity: str = "openrlbenchmark",
        project: str = "cleanrl",
        cache_dir: Path = DEFAULT_CACHE_DIR,
    ) -> None:
        """Configure the source project and local cache location.

        Args:
            entity: W&B entity holding the reference runs.
            project: W&B project name.
            cache_dir: Root directory for the parquet cache.
        """
        self._entity = entity
        self._project = project
        self._cache_dir = cache_dir

    def cache_path(self, algo: str, env_id: str) -> Path:
        """Return where curves for ``algo`` on ``env_id`` are cached."""
        return self._cache_dir / self._entity / algo / f"{env_id}.parquet"

    def fetch(self, algo: str, env_id: str, force: bool = False) -> pd.DataFrame:
        """Fetch reference curves, from cache when possible (see class docs).

        The network path (a cache miss) needs W&B access; everything
        downstream of the cache is offline. Tests only ever exercise the
        cache-hit path via committed fixtures.

        Args:
            algo: CleanRL exp_name, e.g. ``"ppo"``.
            env_id: Environment id the reference ran.
            force: Re-download even if a cache file exists.

        Returns:
            Curves frame with columns ``run_id, global_step, episodic_return``.

        Raises:
            RuntimeError: If the source has no matching runs.
        """
        cached = self.cache_path(algo, env_id)
        if cached.exists() and not force:
            return pd.read_parquet(cached)

        import wandb

        api = wandb.Api()
        runs = api.runs(
            f"{self._entity}/{self._project}",
            filters={"config.env_id": env_id, "config.exp_name": algo},
        )
        frames = []
        for run in runs:
            history = run.history(keys=["global_step", "charts/episodic_return"])
            if history.empty:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "run_id": run.id,
                        "global_step": history["global_step"],
                        "episodic_return": history["charts/episodic_return"],
                    }
                )
            )
        if not frames:
            raise RuntimeError(
                f"No runs found in {self._entity}/{self._project} for "
                f"exp_name={algo!r}, env_id={env_id!r}."
            )
        frame = pd.concat(frames, ignore_index=True)
        cached.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cached, index=False)
        return frame


def load_runs(paths: list[Path]) -> pd.DataFrame:
    """Load run curves from local CSV or parquet files.

    Each file must contain the columns ``run_id``, ``global_step``,
    ``episodic_return`` (the format ``roborl demo --save-episodes`` writes).

    Args:
        paths: Files to load and concatenate.

    Returns:
        A single curves frame.

    Raises:
        ValueError: If a file is missing required columns.
    """
    frames = []
    for path in paths:
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        missing = set(COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        frames.append(frame[list(COLUMNS)])
    return pd.concat(frames, ignore_index=True)


def to_curves(frame: pd.DataFrame) -> list[tuple[FloatArray, FloatArray]]:
    """Split a curves frame into per-run ``(steps, values)`` arrays.

    Args:
        frame: Curves frame with the standard columns.

    Returns:
        One ``(global_step, episodic_return)`` array pair per ``run_id``,
        sorted by step.
    """
    curves = []
    for _, group in frame.groupby("run_id", sort=True):
        ordered = group.sort_values("global_step")
        curves.append(
            (
                ordered["global_step"].to_numpy(dtype=float),
                ordered["episodic_return"].to_numpy(dtype=float),
            )
        )
    return curves
