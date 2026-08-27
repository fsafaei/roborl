"""Evaluation statistics: IQM and stratified bootstrap CIs.

Implements the methodology of Agarwal et al., *Deep RL at the Edge of the
Statistical Precipice* (NeurIPS 2021), directly rather than depending on
rliable — the implementation is ~50 lines and understanding it is part of the
curriculum (see ADR 0005). Semantics match rliable: IQM is the 25%-trimmed
mean, and the stratified bootstrap resamples runs with replacement
independently within each task.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def iqm(scores: FloatArray) -> float:
    """Interquartile mean: the mean of the middle 50% of scores.

    Discards the lowest 25% and highest 25% of values (``floor(0.25 * n)``
    from each end, matching ``scipy.stats.trim_mean``), then averages the
    rest. More robust than the mean, more statistically efficient than the
    median — the recommended point estimate for few-seed deep RL results.

    Args:
        scores: Scores of any shape; flattened before trimming.

    Returns:
        The interquartile mean.

    Raises:
        ValueError: If ``scores`` is empty.
    """
    flat = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    if flat.size == 0:
        raise ValueError("iqm() requires at least one score.")
    cut = int(0.25 * flat.size)
    return float(np.mean(flat[cut : flat.size - cut]))


def stratified_bootstrap_ci(
    score_matrix: FloatArray,
    n_resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the IQM, stratified by task.

    Args:
        score_matrix: Scores shaped ``(n_runs, n_tasks)``; one column per
            task (environment), one row per run (seed). A 1-D array is
            treated as a single task.
        n_resamples: Bootstrap resamples to draw.
        confidence: CI mass, e.g. ``0.95`` for a 95% interval.
        seed: RNG seed — CIs are deterministic given the same inputs.

    Returns:
        ``(lower, upper)`` bounds of the confidence interval.

    Raises:
        ValueError: If the matrix is empty.
    """
    matrix = np.atleast_2d(np.asarray(score_matrix, dtype=np.float64).T).T
    n_runs, n_tasks = matrix.shape
    if matrix.size == 0:
        raise ValueError("stratified_bootstrap_ci() requires at least one score.")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        # Resample runs with replacement independently within each task.
        idx = rng.integers(0, n_runs, size=(n_runs, n_tasks))
        stats[i] = iqm(np.take_along_axis(matrix, idx, axis=0))
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(stats, alpha)),
        float(np.quantile(stats, 1.0 - alpha)),
    )


def align_curves(
    curves: list[tuple[FloatArray, FloatArray]],
    grid: FloatArray,
) -> FloatArray:
    """Interpolate learning curves onto a common ``global_step`` grid.

    Runs log at slightly different steps (episodes end when they end), so
    curves must be aligned before any cross-run statistic is meaningful.

    Args:
        curves: One ``(steps, values)`` pair per run; steps must be
            increasing.
        grid: The common step grid to interpolate onto.

    Returns:
        Array shaped ``(n_runs, len(grid))`` of interpolated values. Values
        outside a run's logged range clamp to its first/last value.
    """
    return np.stack(
        [np.interp(grid, np.asarray(s, np.float64), np.asarray(v, np.float64)) for s, v in curves]
    )


def iqm_curve_with_ci(
    curves: list[tuple[FloatArray, FloatArray]],
    grid: FloatArray,
    n_resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Sample-efficiency curve: pointwise IQM over runs with bootstrap CI band.

    Args:
        curves: One ``(steps, values)`` pair per run.
        grid: Common step grid.
        n_resamples: Bootstrap resamples per grid point.
        confidence: CI mass.
        seed: RNG seed.

    Returns:
        ``(iqm_values, lower_band, upper_band)``, each of ``len(grid)``.
    """
    aligned = align_curves(curves, grid)
    point = np.array([iqm(aligned[:, j]) for j in range(aligned.shape[1])])
    lo = np.empty_like(point)
    hi = np.empty_like(point)
    for j in range(aligned.shape[1]):
        lo[j], hi[j] = stratified_bootstrap_ci(
            aligned[:, j], n_resamples=n_resamples, confidence=confidence, seed=seed
        )
    return point, lo, hi


def final_scores(
    curves: list[tuple[FloatArray, FloatArray]],
    last_fraction: float = 0.1,
) -> FloatArray:
    """Per-run final performance: mean return over the last fraction of training.

    Args:
        curves: One ``(steps, values)`` pair per run.
        last_fraction: Fraction of each run's step range that counts as
            "final", e.g. ``0.1`` for the last 10%.

    Returns:
        One score per run.
    """
    scores = []
    for steps, values in curves:
        steps_arr = np.asarray(steps, np.float64)
        values_arr = np.asarray(values, np.float64)
        threshold = steps_arr[-1] - last_fraction * (steps_arr[-1] - steps_arr[0])
        tail = values_arr[steps_arr >= threshold]
        scores.append(float(np.mean(tail if tail.size else values_arr[-1:])))
    return np.asarray(scores)
