"""Render a markdown verification report from a comparison.

The report is the evidence behind every "verified" claim in the README
status table: it records what was compared, under which commit, the
statistics, the figure, and the verdict. Thresholds are policy, documented
in ``docs/benchmarking.md``; this module only applies them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from roborl.benchmark import fetch, plots
from roborl.benchmark.stats import final_scores, iqm, stratified_bootstrap_ci
from roborl.telemetry.logger import git_provenance


@dataclass(frozen=True)
class ComparisonResult:
    """Everything the comparison computed, for the report and for tests.

    Attributes:
        verdict: ``"PASS"`` or ``"INVESTIGATE"`` (see docs/benchmarking.md).
        ours_iqm: Final-performance IQM of our runs.
        ours_ci: 95% stratified bootstrap CI for our IQM.
        reference_iqm: Final-performance IQM of the reference runs.
        reference_ci: 95% CI for the reference IQM.
        n_ours: Number of our runs.
        n_reference: Number of reference runs.
        report_path: The markdown file written.
        figure_path: The PNG written.
    """

    verdict: str
    ours_iqm: float
    ours_ci: tuple[float, float]
    reference_iqm: float
    reference_ci: tuple[float, float]
    n_ours: int
    n_reference: int
    report_path: Path
    figure_path: Path


def decide_verdict(
    ours_iqm: float,
    ours_ci: tuple[float, float],
    reference_iqm: float,
    reference_ci: tuple[float, float],
    n_reference: int,
) -> str:
    """Apply the verification policy of ``docs/benchmarking.md``.

    PASS when our final IQM's CI overlaps the reference's CI; with too few
    reference runs to bootstrap meaningfully (< 3), PASS when our IQM is at
    least 90% of the reference IQM. Anything else is INVESTIGATE, which
    triggers the debugging protocol and a lab-notebook entry.

    Args:
        ours_iqm: Our final-performance IQM.
        ours_ci: Our CI bounds.
        reference_iqm: Reference final-performance IQM.
        reference_ci: Reference CI bounds.
        n_reference: Number of reference runs.

    Returns:
        ``"PASS"`` or ``"INVESTIGATE"``.
    """
    if n_reference < 3:
        return "PASS" if ours_iqm >= 0.9 * reference_iqm else "INVESTIGATE"
    overlap = ours_ci[0] <= reference_ci[1] and reference_ci[0] <= ours_ci[1]
    return "PASS" if overlap else "INVESTIGATE"


def run_compare(
    ours_paths: list[Path],
    reference_paths: list[Path],
    algo: str,
    env_id: str,
    out_dir: Path,
    reference_label: str = "reference",
    ours_label: str = "roborl",
    grid_points: int = 50,
    last_fraction: float = 0.1,
) -> ComparisonResult:
    """Compare our runs against reference runs and write report + figure.

    Args:
        ours_paths: CSV/parquet curve files for our runs.
        reference_paths: CSV/parquet curve files for the reference runs
            (e.g. the parquet cache written by ``roborl benchmark fetch``).
        algo: Algorithm name for the report header.
        env_id: Environment id for the report header.
        out_dir: Output directory; ``report.md`` and ``curves.png`` land here.
        reference_label: Name of the reference source for labels.
        ours_label: Name of our run set for labels — override when the
            reference is also a roborl run set (e.g. "roborl FlashSAC" vs
            "roborl SAC (verified)"), where a bare "roborl" is ambiguous.
        grid_points: Resolution of the common step grid.
        last_fraction: Fraction of training counted as final performance.

    Returns:
        The comparison result, with paths to the written files.

    Raises:
        ValueError: If either run set is empty.
    """
    ours = fetch.to_curves(fetch.load_runs(ours_paths))
    reference = fetch.to_curves(fetch.load_runs(reference_paths))
    if not ours or not reference:
        raise ValueError("Both run sets must contain at least one run.")

    all_curves = ours + reference
    grid_start = max(float(steps[0]) for steps, _ in all_curves)
    grid_end = min(float(steps[-1]) for steps, _ in all_curves)
    grid = np.linspace(grid_start, grid_end, grid_points)

    ours_final = final_scores(ours, last_fraction)
    reference_final = final_scores(reference, last_fraction)
    ours_iqm, ours_ci = iqm(ours_final), stratified_bootstrap_ci(ours_final)
    reference_iqm = iqm(reference_final)
    reference_ci = stratified_bootstrap_ci(reference_final)
    verdict = decide_verdict(ours_iqm, ours_ci, reference_iqm, reference_ci, len(reference))

    out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = plots.plot_comparison(
        ours,
        reference,
        grid,
        out_dir / "curves.png",
        title=f"{algo} on {env_id}",
        ours_label=ours_label,
        reference_label=reference_label,
        final_stats={
            reference_label: (reference_iqm, *reference_ci),
            ours_label: (ours_iqm, *ours_ci),
        },
    )

    provenance = git_provenance()
    report_path = out_dir / "report.md"
    report_path.write_text(
        _render_markdown(
            algo=algo,
            env_id=env_id,
            verdict=verdict,
            ours_iqm=ours_iqm,
            ours_ci=ours_ci,
            reference_iqm=reference_iqm,
            reference_ci=reference_ci,
            n_ours=len(ours),
            n_reference=len(reference),
            reference_label=reference_label,
            ours_label=ours_label,
            last_fraction=last_fraction,
            git_sha=str(provenance["git_sha"]),
            git_dirty=bool(provenance["git_dirty"]),
            ours_sources=[str(p) for p in ours_paths],
            reference_sources=[str(p) for p in reference_paths],
        )
    )
    return ComparisonResult(
        verdict=verdict,
        ours_iqm=ours_iqm,
        ours_ci=ours_ci,
        reference_iqm=reference_iqm,
        reference_ci=reference_ci,
        n_ours=len(ours),
        n_reference=len(reference),
        report_path=report_path,
        figure_path=figure_path,
    )


def _render_markdown(
    *,
    algo: str,
    env_id: str,
    verdict: str,
    ours_iqm: float,
    ours_ci: tuple[float, float],
    reference_iqm: float,
    reference_ci: tuple[float, float],
    n_ours: int,
    n_reference: int,
    reference_label: str,
    ours_label: str,
    last_fraction: float,
    git_sha: str,
    git_dirty: bool,
    ours_sources: list[str],
    reference_sources: list[str],
) -> str:
    """Fill the report template with computed values."""
    dirty_note = " (dirty)" if git_dirty else ""
    return f"""# Verification report: {algo} on {env_id}

| | |
|---|---|
| Algorithm | `{algo}` |
| Environment | `{env_id}` |
| Commit | `{git_sha[:12]}`{dirty_note} |
| Our runs | {n_ours} |
| Reference runs | {n_reference} ({reference_label}) |
| Final window | last {last_fraction:.0%} of training |
| **Verdict** | **{verdict}** |

## Final performance (IQM over final window, 95% stratified bootstrap CI)

| Run set | IQM | 95% CI |
|---|---|---|
| {ours_label} | {ours_iqm:.2f} | [{ours_ci[0]:.2f}, {ours_ci[1]:.2f}] |
| {reference_label} | {reference_iqm:.2f} | [{reference_ci[0]:.2f}, {reference_ci[1]:.2f}] |

## Sample efficiency

![learning curves](curves.png)

## Sources

- Ours: {", ".join(f"`{s}`" for s in ours_sources)}
- Reference: {", ".join(f"`{s}`" for s in reference_sources)}

Verdict policy: see `docs/benchmarking.md`. A PASS means our final IQM's CI
overlaps the reference's (or IQM >= 90% of reference when the reference has
fewer than 3 runs). INVESTIGATE triggers the debugging protocol
(`docs/debugging-rl.md`) and a lab-notebook entry.
"""
