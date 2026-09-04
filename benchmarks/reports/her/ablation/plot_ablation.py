r"""Ablation figure: per-rung success-rate IQM curves with CI bands + final-window table.

    uv run python benchmarks/reports/her/ablation/plot_ablation.py \\
        --rung "R0 no HER (sparse)" runs/her-sac-r0-*-success.csv \\
        --rung "R5 future k=4" runs/her-sac-FetchPush-v4-s{1,2,3}-*-success.csv \\
        --out benchmarks/reports/her/ablation

Inputs are the ``--save-episodes`` success CSVs (``run_id, global_step,
episodic_success``; 0/1 per training episode) or the SB3 converter's
``*-success.csv``. Each run's 0/1 stream is smoothed with a trailing window
of ``--window`` episodes (default 20 = 1000 Fetch steps) before the
cross-run statistics — pointwise IQM with 95% stratified bootstrap CIs from
``roborl.benchmark.stats`` — so the curves read as success *rates*. The
final window is the last 10% of each run's steps. Descriptive only: verdict
language belongs to ``roborl benchmark compare``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from roborl.benchmark.stats import (
    FloatArray,
    final_scores,
    iqm,
    iqm_curve_with_ci,
    stratified_bootstrap_ci,
)


def load_success_curves(paths: list[Path], window: int) -> list[tuple[FloatArray, FloatArray]]:
    """Load success CSVs into per-run ``(steps, smoothed success)`` curves.

    Args:
        paths: Success CSV files (one or more runs each).
        window: Trailing window, in episodes, for the rolling success rate.

    Returns:
        One curve per ``run_id`` found, sorted by run id.

    Raises:
        ValueError: If a file lacks the required columns.
    """
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = {"run_id", "global_step", "episodic_success"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    curves = []
    for _, group in table.groupby("run_id", sort=True):
        ordered = group.sort_values("global_step")
        smoothed = ordered["episodic_success"].rolling(window, min_periods=1).mean()
        curves.append(
            (ordered["global_step"].to_numpy(dtype=float), smoothed.to_numpy(dtype=float))
        )
    return curves


def main() -> None:
    """Render ``ablation.png`` and ``final_table.md`` for the given rungs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rung",
        action="append",
        nargs="+",
        metavar=("LABEL", "CSV"),
        required=True,
        help="A rung label followed by its success CSV files; repeat per rung.",
    )
    parser.add_argument("--out", type=Path, default=Path("benchmarks/reports/her/ablation"))
    parser.add_argument("--window", type=int, default=20, help="rolling window in episodes")
    parser.add_argument("--grid-points", type=int, default=100)
    parser.add_argument("--title", default="HER ablation on FetchPush-v4 — training success rate")
    args = parser.parse_args()

    rungs = [(spec[0], [Path(p) for p in spec[1:]]) for spec in args.rung]
    curves_by_rung = {label: load_success_curves(paths, args.window) for label, paths in rungs}
    all_curves = [c for curves in curves_by_rung.values() for c in curves]
    grid_start = max(float(steps[0]) for steps, _ in all_curves)
    grid_end = min(float(steps[-1]) for steps, _ in all_curves)
    grid = np.linspace(grid_start, grid_end, args.grid_points)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    rows = ["| Rung | n | final success IQM [95% CI] |", "|---|---|---|"]
    for i, (label, curves) in enumerate(curves_by_rung.items()):
        point, lo, hi = iqm_curve_with_ci(curves, grid)
        color = f"C{i}"
        ax.plot(grid, point, label=f"{label} (n={len(curves)})", color=color)
        ax.fill_between(grid, lo, hi, alpha=0.15, color=color)
        final = final_scores(curves, last_fraction=0.1)
        ci = stratified_bootstrap_ci(final)
        rows.append(f"| {label} | {len(curves)} | {iqm(final):.3f} [{ci[0]:.3f}, {ci[1]:.3f}] |")
    ax.set_xlabel("global_step")
    ax.set_ylabel(f"success rate (rolling {args.window} episodes; IQM, 95% CI)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(args.title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "ablation.png", dpi=150)
    plt.close(fig)
    table = "\n".join(rows) + "\n"
    (args.out / "final_table.md").write_text(table)
    print(table)
    print(f"figure: {args.out / 'ablation.png'}")


if __name__ == "__main__":
    main()
