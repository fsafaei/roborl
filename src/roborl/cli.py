"""Command-line entry point for roborl.

Phase 1 placeholder: subcommands (``demo``, ``benchmark``) arrive with the
core library in the next phase.
"""

from __future__ import annotations

import roborl


def main() -> None:
    """Print the package version.

    Replaced by the tyro-powered subcommand dispatcher in Phase 2.
    """
    print(f"roborl {roborl.__version__} — run `roborl demo` once Phase 2 lands.")
