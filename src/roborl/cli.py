"""Command-line entry point: ``roborl <subcommand>`` via tyro.

Subcommands are frozen config dataclasses; tyro derives flags, defaults, and
help text from their fields and docstrings. The benchmark subcommands import
their heavy dependencies lazily so the base install can run the demo without
the ``benchmark`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import tyro

from roborl.algos.flashsac.flashsac import FlashSacConfig, run_flashsac
from roborl.algos.ppo.ppo import PpoConfig, run_ppo
from roborl.algos.ppo.ppo_continuous import PpoContinuousConfig, run_ppo_continuous
from roborl.algos.sac.sac import SacConfig, run_sac
from roborl.demo import DemoConfig, run_demo


@dataclass(frozen=True)
class BenchmarkFetchArgs:
    """Download reference curves into the local parquet cache."""

    algo: str
    """Algorithm as the reference spells it (a CleanRL exp_name, e.g. "ppo")."""
    env_id: str
    """Environment id the reference ran (env-version parity rule)."""
    entity: str = "openrlbenchmark"
    """W&B entity holding the reference runs."""
    project: str = "cleanrl"
    """W&B project holding the reference runs."""
    force: bool = False
    """Re-download even when a cache file exists."""


@dataclass(frozen=True)
class BenchmarkCompareArgs:
    """Compare our runs against reference runs; write report + figure."""

    ours: list[Path]
    """Curve files (CSV/parquet) for our runs, e.g. from --save-episodes."""
    reference: list[Path]
    """Curve files for the reference runs, e.g. the fetch cache parquet."""
    algo: str
    """Algorithm name for the report header."""
    env_id: str
    """Environment id for the report header."""
    out: Path | None = None
    """Output directory (default: benchmarks/reports/<algo>/<env_id>)."""
    reference_label: str = "reference"
    """Label for the reference source in figure and report."""


@dataclass(frozen=True)
class BenchmarkWandbReportArgs:
    """Create/update a shareable W&B report for one experiment's tracked runs."""

    algo: str
    """Our exp_name for the experiment (e.g. "sac", "flashsac")."""
    env_ids: list[str]
    """Environments, one report section each."""
    title: str | None = None
    """Report title (default "<algo> results")."""
    description: str = ""
    """One-line description shown under the title."""
    intro: str = ""
    """Markdown intro block (provenance, methodology, verdict links)."""
    entity: str | None = None
    """W&B entity holding our runs (default: the API's default entity)."""
    project: str = "roborl"
    """W&B project holding our runs."""
    baseline_algo: str | None = None
    """Second exp_name from our project to overlay (e.g. "sac" under flashsac)."""
    reference_exp_name: str | None = None
    """CleanRL exp_name to overlay from openrlbenchmark/cleanrl."""
    metrics: list[str] = field(default_factory=lambda: ["charts/episodic_return", "charts/SPS"])
    """Metric keys plotted per environment section."""


BenchmarkCommand = (
    Annotated[BenchmarkFetchArgs, tyro.conf.subcommand("fetch")]
    | Annotated[BenchmarkCompareArgs, tyro.conf.subcommand("compare")]
    | Annotated[BenchmarkWandbReportArgs, tyro.conf.subcommand("report-wandb")]
)


@dataclass(frozen=True)
class BenchmarkArgs:
    """Benchmarking harness: fetch reference curves, compare runs."""

    command: BenchmarkCommand


def _run_benchmark(args: BenchmarkArgs) -> None:
    """Dispatch a benchmark subcommand, importing heavy deps lazily."""
    try:
        from roborl.benchmark.fetch import OpenRLBenchmarkAdapter
        from roborl.benchmark.report import run_compare
        from roborl.benchmark.wandb_report import build_report
    except ImportError as error:
        raise SystemExit(
            "The benchmark harness needs the 'benchmark' extra:\n"
            "    uv sync --group dev --extra benchmark\n"
            f"(import failed: {error})"
        ) from error

    command = args.command
    if isinstance(command, BenchmarkWandbReportArgs):
        url = build_report(
            algo=command.algo,
            env_ids=command.env_ids,
            entity=command.entity,
            project=command.project,
            metrics=command.metrics,
            title=command.title,
            description=command.description,
            intro=command.intro,
            baseline_algo=command.baseline_algo,
            reference_exp_name=command.reference_exp_name,
        )
        print(f"report: {url}")
    elif isinstance(command, BenchmarkFetchArgs):
        adapter = OpenRLBenchmarkAdapter(entity=command.entity, project=command.project)
        frame = adapter.fetch(command.algo, command.env_id, force=command.force)
        n_runs = frame["run_id"].nunique()
        print(
            f"fetched {n_runs} reference runs ({len(frame)} points) -> "
            f"{adapter.cache_path(command.algo, command.env_id)}"
        )
    else:
        out_dir = command.out or Path("benchmarks/reports") / command.algo / command.env_id
        result = run_compare(
            ours_paths=command.ours,
            reference_paths=command.reference,
            algo=command.algo,
            env_id=command.env_id,
            out_dir=out_dir,
            reference_label=command.reference_label,
        )
        print(
            f"verdict: {result.verdict}\n"
            f"roborl final IQM {result.ours_iqm:.2f} "
            f"[{result.ours_ci[0]:.2f}, {result.ours_ci[1]:.2f}] (n={result.n_ours})  vs  "
            f"{command.reference_label} {result.reference_iqm:.2f} "
            f"[{result.reference_ci[0]:.2f}, {result.reference_ci[1]:.2f}] "
            f"(n={result.n_reference})\n"
            f"report: {result.report_path}"
        )


def main() -> None:
    """Parse the subcommand, run it, and print its summary."""
    config = tyro.extras.subcommand_cli_from_dict(
        {
            "demo": DemoConfig,
            "sac": SacConfig,
            "flashsac": FlashSacConfig,
            "ppo": PpoConfig,
            "ppo-continuous": PpoContinuousConfig,
            "benchmark": BenchmarkArgs,
        },
        description="roborl — learning RL for robotics by building it.",
        config=(tyro.conf.OmitSubcommandPrefixes,),
    )
    if isinstance(config, SacConfig):
        print(run_sac(config).render())
    elif isinstance(config, FlashSacConfig):
        print(run_flashsac(config).render())
    elif isinstance(config, PpoContinuousConfig):
        # Before PpoConfig: the continuous config subclasses the discrete one.
        print(run_ppo_continuous(config).render())
    elif isinstance(config, PpoConfig):
        print(run_ppo(config).render())
    elif isinstance(config, DemoConfig):
        print(run_demo(config).render())
    elif isinstance(config, BenchmarkArgs):
        _run_benchmark(config)
