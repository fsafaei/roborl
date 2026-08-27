"""Verification figures: sample-efficiency curves with CI bands, final bars.

Requires the ``benchmark`` extra. Uses the Agg backend so plotting works
headless (CI, servers).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from roborl.benchmark.stats import FloatArray, iqm_curve_with_ci


def plot_comparison(
    ours: list[tuple[FloatArray, FloatArray]],
    reference: list[tuple[FloatArray, FloatArray]],
    grid: FloatArray,
    out_path: Path,
    title: str,
    ours_label: str = "roborl",
    reference_label: str = "reference",
    final_stats: dict[str, tuple[float, float, float]] | None = None,
) -> Path:
    """Render the two-panel verification figure.

    Left panel: pointwise-IQM sample-efficiency curves with stratified
    bootstrap CI bands for both run sets. Right panel: final-performance IQM
    bars with CI error bars (when ``final_stats`` is given).

    Args:
        ours: Our runs as ``(steps, values)`` pairs.
        reference: Reference runs as ``(steps, values)`` pairs.
        grid: Common step grid for curve alignment.
        out_path: Where to write the PNG.
        title: Figure title, e.g. ``"dqn on CartPole-v1"``.
        ours_label: Legend label for our runs.
        reference_label: Legend label for the reference runs.
        final_stats: Optional ``{label: (iqm, ci_lo, ci_hi)}`` for the bars.

    Returns:
        The path the PNG was written to.
    """
    n_panels = 2 if final_stats else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5), squeeze=False)

    ax = axes[0][0]
    for curves, label, color in ((reference, reference_label, "C1"), (ours, ours_label, "C0")):
        point, lo, hi = iqm_curve_with_ci(curves, grid)
        ax.plot(grid, point, label=f"{label} (n={len(curves)})", color=color)
        ax.fill_between(grid, lo, hi, alpha=0.2, color=color)
    ax.set_xlabel("global_step")
    ax.set_ylabel("episodic return (IQM, 95% CI)")
    ax.set_title(title)
    ax.legend()

    if final_stats:
        ax = axes[0][1]
        labels = list(final_stats)
        values = [final_stats[label][0] for label in labels]
        errors = np.array(
            [
                [final_stats[label][0] - final_stats[label][1] for label in labels],
                [final_stats[label][2] - final_stats[label][0] for label in labels],
            ]
        )
        ax.bar(labels, values, yerr=errors, capsize=6, color=["C1", "C0"][: len(labels)])
        ax.set_ylabel("final episodic return (IQM, 95% CI)")
        ax.set_title("final performance")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
