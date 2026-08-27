"""Command-line entry point: ``roborl <subcommand>`` via tyro.

Subcommands are frozen config dataclasses; tyro derives flags, defaults, and
help text from their fields and docstrings. ``benchmark`` subcommands arrive
with the benchmarking harness.
"""

from __future__ import annotations

import tyro

from roborl.demo import DemoConfig, run_demo


def main() -> None:
    """Parse the subcommand, run it, and print its summary."""
    config = tyro.extras.subcommand_cli_from_dict(
        {"demo": DemoConfig},
        description="roborl — learning RL for robotics by building it.",
    )
    if isinstance(config, DemoConfig):
        print(run_demo(config).render())
