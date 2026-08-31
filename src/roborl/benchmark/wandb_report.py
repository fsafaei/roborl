"""Programmatic W&B Reports: one shareable report per experiment.

Builds a report in the style of the public RL baselines reports — an intro
block with provenance, then one section per environment whose panels
overlay aggregated curves (mean line, min/max band) from up to three run
sets: our runs for the algorithm, optionally a baseline algorithm from the
same project (e.g. FlashSAC versus our verified SAC), and optionally a
CleanRL reference from ``openrlbenchmark/cleanrl``.

Run sets are selected by ``config.exp_name`` and ``config.env_id`` — the
keys every roborl run records (and CleanRL's runs share) — so a report
regenerates correctly as new seeds land. The W&B report is a *view* onto
tracked runs; verdicts still come exclusively from ``roborl benchmark
compare`` and the committed reports (ADR 0005).
"""

from __future__ import annotations

import re
from pathlib import Path

import wandb
import wandb_workspaces.expr as expr
import wandb_workspaces.reports.v2 as wr

REFERENCE_ENTITY = "openrlbenchmark"
REFERENCE_PROJECT = "cleanrl"
REPO_URL = "https://github.com/fsafaei/roborl"


def _env_notes(reports_dir: Path, algo: str, env_id: str) -> str:
    """Per-environment explanation sourced from the committed harness report.

    Extracts the verdict and the final-performance table from
    ``benchmarks/reports/<algo>/<env_id>/report.md`` so every number shown
    in the W&B report traces to committed evidence. Environments without a
    committed report get an explicit "verdict pending" note instead.
    """
    path = reports_dir / algo / env_id / "report.md"
    if not path.exists():
        return (
            "*No committed verification report for this environment yet — the "
            "verdict lands via `roborl benchmark compare` once its runs "
            "complete, and this section will be updated with the numbers.*"
        )
    text = path.read_text()
    verdict_match = re.search(r"\|\s*\*\*Verdict\*\*\s*\|\s*\*\*(.+?)\*\*\s*\|", text)
    verdict = verdict_match.group(1) if verdict_match else "see committed report"
    table_lines: list[str] = []
    in_section = False
    for line in text.splitlines():
        if line.startswith("## Final performance"):
            in_section = True
            continue
        if in_section:
            if line.startswith("## "):
                break
            if line.lstrip().startswith("|"):
                table_lines.append(line.strip())
    link = f"{REPO_URL}/blob/main/benchmarks/reports/{algo}/{env_id}/report.md"
    note = (
        f"**Verdict: {verdict}** — IQM over the last 10% of training with 95% "
        f"stratified bootstrap CIs, from the [committed harness report]({link}):"
    )
    if table_lines:
        note += "\n\n" + "\n".join(table_lines)
    return note


def _runset(entity: str, project: str, exp_name: str, env_id: str, label: str) -> wr.Runset:
    """One run set: all runs of ``exp_name`` on ``env_id`` in a project."""
    return wr.Runset(
        entity=entity,
        project=project,
        name=label,
        filters=[
            expr.Config("exp_name") == exp_name,
            expr.Config("env_id") == env_id,
        ],
    )


def _panel(metric: str, title: str | None = None) -> wr.LinePlot:
    """An aggregated line panel: mean curve per run set with a min/max band."""
    return wr.LinePlot(
        title=title or metric,
        x="global_step",
        y=[metric],
        aggregate=True,
        groupby_aggfunc="mean",
        groupby_rangefunc="minmax",
        smoothing_factor=0.9,
        smoothing_type="exponentialTimeWeighted",
        layout=wr.Layout(w=12, h=8),
    )


def build_report(
    algo: str,
    env_ids: list[str],
    entity: str | None,
    project: str,
    metrics: list[str],
    title: str | None = None,
    description: str = "",
    intro: str = "",
    baseline_algo: str | None = None,
    reference_exp_name: str | None = None,
    reports_dir: Path | None = None,
    update_url: str | None = None,
    ours_runset_label: str | None = None,
    baseline_runset_label: str | None = None,
    reference_runset_label: str | None = None,
) -> str:
    """Create (or overwrite-as-new) the experiment's W&B report.

    Args:
        algo: Our ``exp_name`` for the experiment (e.g. ``"sac"``).
        env_ids: Environments, one report section each.
        entity: W&B entity holding our runs; None uses the API default.
        project: W&B project holding our runs.
        metrics: Metric keys to plot per environment section.
        title: Report title; default ``"<algo> results"``.
        description: One-line report description (shown under the title).
        intro: Markdown for the intro block (provenance, methodology).
        baseline_algo: Optional second ``exp_name`` from *our* project to
            overlay (e.g. ``"sac"`` under a ``"flashsac"`` report).
        reference_exp_name: Optional CleanRL ``exp_name`` to overlay from
            ``openrlbenchmark/cleanrl`` (e.g. ``"sac_continuous_action"``).
        reports_dir: Root of the committed harness reports; when given, each
            environment section opens with its verdict and final-performance
            table extracted from ``<reports_dir>/<algo>/<env_id>/report.md``.
        update_url: Existing report URL to update in place (keeps the link
            stable); None creates a new report.
        ours_runset_label: Display name for our run set in the chart
            legends (default ``"<algo> (roborl)"``).
        baseline_runset_label: Display name for the baseline run set.
        reference_runset_label: Display name for the CleanRL run set.

    Returns:
        The saved report's URL.

    Raises:
        ValueError: If no entity is given and the API has no default.
    """
    resolved_entity = entity or wandb.Api().default_entity
    if not resolved_entity:
        raise ValueError("No W&B entity: pass --entity or log in so the API has a default.")

    blocks: list[wr.interface.BlockTypes] = [wr.TableOfContents()]
    if intro:
        blocks.append(wr.MarkdownBlock(text=intro))

    for env_id in env_ids:
        ours_label = ours_runset_label or f"{algo} (roborl)"
        runsets = [_runset(resolved_entity, project, algo, env_id, ours_label)]
        if baseline_algo:
            baseline_label = baseline_runset_label or f"{baseline_algo} (roborl)"
            runsets.append(_runset(resolved_entity, project, baseline_algo, env_id, baseline_label))
        if reference_exp_name:
            reference_label = reference_runset_label or f"{reference_exp_name} (CleanRL reference)"
            runsets.append(
                _runset(
                    REFERENCE_ENTITY,
                    REFERENCE_PROJECT,
                    reference_exp_name,
                    env_id,
                    reference_label,
                )
            )
        blocks.append(wr.H2(text=env_id))
        if reports_dir is not None:
            blocks.append(wr.MarkdownBlock(text=_env_notes(reports_dir, algo, env_id)))
        blocks.append(wr.PanelGrid(runsets=runsets, panels=[_panel(m) for m in metrics]))

    if update_url is not None:
        report = wr.Report.from_url(update_url)
        report.title = title or f"{algo} results"
        report.description = description
        report.blocks = blocks
    else:
        report = wr.Report(
            entity=resolved_entity,
            project=project,
            title=title or f"{algo} results",
            description=description,
            blocks=blocks,
        )
    report.save()
    url: str = report.url
    return url


def build_workspace(
    algo: str,
    entity: str | None,
    project: str,
    metrics: list[str],
    name: str | None = None,
) -> str:
    """Create a saved W&B workspace view for one experiment's runs.

    The view filters the project's runs to ``config.exp_name == algo`` and
    groups them by ``config.env_id``, so every panel shows one aggregated
    curve per environment. Panels are bucketed into sections by metric
    namespace (``charts/``, ``losses/``, ``diagnostics/``, ...).

    Args:
        algo: Our ``exp_name`` for the experiment.
        entity: W&B entity; None uses the API default.
        project: W&B project holding the runs.
        metrics: Metric keys to plot, sectioned by their namespace prefix.
        name: View name (default ``"<algo>"``). W&B forbids emoji here.

    Returns:
        The saved view's URL.

    Raises:
        ValueError: If no entity is given and the API has no default.
    """
    import wandb_workspaces.workspaces as ws

    resolved_entity = entity or wandb.Api().default_entity
    if not resolved_entity:
        raise ValueError("No W&B entity: pass --entity or log in so the API has a default.")

    sections: dict[str, list[str]] = {}
    for metric in metrics:
        namespace = metric.split("/", 1)[0] if "/" in metric else "misc"
        sections.setdefault(namespace, []).append(metric)

    workspace = ws.Workspace(
        entity=resolved_entity,
        project=project,
        name=name or algo,
        sections=[
            ws.Section(
                name=namespace,
                panels=[
                    wr.LinePlot(x="global_step", y=[metric], title=metric)
                    for metric in section_metrics
                ],
                is_open=(namespace == "charts"),
            )
            for namespace, section_metrics in sections.items()
        ],
        runset_settings=ws.RunsetSettings(
            filters=[expr.Config("exp_name") == algo],
            groupby=[expr.Config("env_id")],
        ),
    )
    workspace.save()
    url: str = workspace.url
    return url
