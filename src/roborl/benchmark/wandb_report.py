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

import wandb
import wandb_workspaces.expr as expr
import wandb_workspaces.reports.v2 as wr

REFERENCE_ENTITY = "openrlbenchmark"
REFERENCE_PROJECT = "cleanrl"


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
        runsets = [_runset(resolved_entity, project, algo, env_id, f"{algo} (roborl)")]
        if baseline_algo:
            baseline_label = f"{baseline_algo} (roborl)"
            runsets.append(_runset(resolved_entity, project, baseline_algo, env_id, baseline_label))
        if reference_exp_name:
            runsets.append(
                _runset(
                    REFERENCE_ENTITY,
                    REFERENCE_PROJECT,
                    reference_exp_name,
                    env_id,
                    f"{reference_exp_name} (CleanRL reference)",
                )
            )
        blocks.append(wr.H2(text=env_id))
        blocks.append(wr.PanelGrid(runsets=runsets, panels=[_panel(m) for m in metrics]))

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
