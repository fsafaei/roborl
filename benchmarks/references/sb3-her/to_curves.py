# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["pandas==3.0.5"]
# ///
r"""Convert SB3 ``Monitor`` CSVs into the roborl benchmark curves format.

    uv run --script benchmarks/references/sb3-her/to_curves.py \\
        benchmarks/references/sb3-her/monitor/FetchPush-v4-s*.monitor.csv \\
        --out benchmarks/references/sb3-her/curves

For each ``{env_id}-s{seed}.monitor.csv`` this writes two files under
``--out``:

- ``{env_id}-s{seed}.csv`` — ``run_id, global_step, episodic_return`` with
  ``global_step`` the cumulative sum of episode lengths (the step at which
  each episode ended, exactly as ``--save-episodes`` records ours), the
  shape ``roborl benchmark compare --reference`` consumes;
- ``{env_id}-s{seed}-success.csv`` — ``run_id, global_step,
  episodic_success`` (final-step ``is_success`` per episode) for the
  success-rate figures.

A Monitor file's first line is a ``#{json header}`` comment; pandas skips it
with ``comment="#"``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def convert(monitor_csv: Path, out_dir: Path) -> tuple[Path, Path]:
    """Convert one Monitor CSV; returns the (curves, success) paths written."""
    frame = pd.read_csv(monitor_csv, comment="#")
    missing = {"r", "l", "is_success"} - set(frame.columns)
    if missing:
        raise ValueError(f"{monitor_csv}: Monitor CSV lacks columns {sorted(missing)}")
    run_id = monitor_csv.name.removesuffix(".monitor.csv")
    global_step = frame["l"].astype(int).cumsum()
    curves = pd.DataFrame(
        {"run_id": run_id, "global_step": global_step, "episodic_return": frame["r"].astype(float)}
    )
    success = pd.DataFrame(
        {
            "run_id": run_id,
            "global_step": global_step,
            "episodic_success": frame["is_success"].astype(float).round().astype(int),
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    curves_path = out_dir / f"{run_id}.csv"
    success_path = out_dir / f"{run_id}-success.csv"
    curves.to_csv(curves_path, index=False)
    success.to_csv(success_path, index=False)
    return curves_path, success_path


def main() -> None:
    """Convert every Monitor CSV given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("monitor_csvs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/references/sb3-her/curves"))
    args = parser.parse_args()
    for path in args.monitor_csvs:
        curves_path, success_path = convert(path, args.out)
        n = sum(1 for _ in curves_path.open()) - 1
        print(f"{path.name}: {n} episodes -> {curves_path}, {success_path}")


if __name__ == "__main__":
    main()
