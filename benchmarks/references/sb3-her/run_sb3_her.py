# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "stable-baselines3==2.9.0",
#   "gymnasium==1.3.0",
#   "gymnasium-robotics==1.4.2",
#   "mujoco==3.11.0",
#   "torch==2.13.0",
#   "numpy==2.5.2",
#   "pandas==3.0.5",
# ]
# ///
r"""Reference runner: Stable-Baselines3 SAC + HerReplayBuffer on Fetch (ADR 0008).

Runs in its own isolated environment via ``uv run --script`` — SB3 never
enters roborl's dependency tree. Every hyperparameter below is the
``her-sac`` column of ``docs/algos/her.md`` verbatim, so the comparison is a
verification of our HER buffer + SAC against SB3's, not a benchmark fight.

    uv run --script benchmarks/references/sb3-her/run_sb3_her.py \\
        --env-id FetchPush-v4 --seed 1 --total-timesteps 1000000 \\
        --log-dir benchmarks/references/sb3-her/monitor

Writes ``{log_dir}/{env_id}-s{seed}.monitor.csv`` (SB3 ``Monitor`` format:
one row per episode with ``r, l, t, is_success``) plus a ``.json`` sidecar
with package versions, the runner's git SHA, host, and wall-clock. Convert
to the harness format with ``to_curves.py``.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def _git_sha() -> str:
    """Return the roborl checkout's HEAD SHA (``-dirty`` suffixed), or ``unknown``."""
    repo = Path(__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=repo
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=repo
        )
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    """Train one SB3 SAC+HER run under the her-sac recipe and write Monitor CSV + sidecar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="FetchPush-v4")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument(
        "--log-dir", type=Path, default=Path("benchmarks/references/sb3-her/monitor")
    )
    parser.add_argument("--device", default="auto", help="SB3 device string (auto, cpu, mps, cuda)")
    # The her-sac recipe. Overriding any of these breaks parity; they exist as
    # flags only so short pilots can be run, and the sidecar records them.
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[512, 512, 512])
    args = parser.parse_args()

    import gymnasium as gym
    import gymnasium_robotics
    import mujoco
    import numpy as np
    import stable_baselines3 as sb3
    import torch
    from stable_baselines3 import SAC, HerReplayBuffer
    from stable_baselines3.common.monitor import Monitor

    gym.register_envs(gymnasium_robotics)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    stem = args.log_dir / f"{args.env_id}-s{args.seed}"

    env = Monitor(gym.make(args.env_id), filename=str(stem), info_keywords=("is_success",))
    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={"n_sampled_goal": 4, "goal_selection_strategy": "future"},
        gamma=0.95,
        tau=0.05,
        learning_rate=1e-3,
        batch_size=args.batch_size,
        buffer_size=1_000_000,
        learning_starts=args.learning_starts,
        policy_kwargs={"net_arch": list(args.net_arch)},
        seed=args.seed,
        device=args.device,
        verbose=0,
    )
    start = time.perf_counter()
    model.learn(total_timesteps=args.total_timesteps, log_interval=None)
    elapsed = time.perf_counter() - start
    env.close()

    provenance = {
        "env_id": args.env_id,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "net_arch": list(args.net_arch),
        "gamma": 0.95,
        "tau": 0.05,
        "learning_rate": 1e-3,
        "buffer_size": 1_000_000,
        "her": {"n_sampled_goal": 4, "goal_selection_strategy": "future"},
        "device": str(model.device),
        "elapsed_seconds": round(elapsed, 1),
        "sps": round(args.total_timesteps / elapsed, 2),
        "versions": {
            "python": sys.version.split()[0],
            "stable_baselines3": sb3.__version__,
            "gymnasium": gym.__version__,
            "gymnasium_robotics": gymnasium_robotics.__version__,
            "mujoco": mujoco.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "runner_git_sha": _git_sha(),
        "host": platform.node(),
        "platform": platform.platform(),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    stem.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
