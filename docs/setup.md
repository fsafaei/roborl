# Setup

The promise: a fresh clone is productive in under five minutes on any machine,
with or without a GPU. Everything *runs* on CPU; a GPU is an accelerator,
never a requirement.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) — installs Python itself if needed:
  `curl -LsSf https://astral.sh/uv/install.sh | sh`
- git

## Install matrix

| Machine | Command | What you get |
|---|---|---|
| macOS (Apple Silicon) | `uv sync` | PyPI torch wheels with MPS acceleration |
| Linux with NVIDIA GPU | `uv sync` | PyPI torch wheels with bundled CUDA (large download) |
| Linux, CPU only (incl. CI) | `uv sync --extra cpu` | Lean CPU-only wheels from the PyTorch index |
| Linux/Windows, pinned CUDA | `uv sync --extra cu130` | Wheels from the `cu130` PyTorch index |

Notes:

- `--extra cpu` and `--extra cu130` are mutually exclusive (declared as a
  conflict in `pyproject.toml`).
- `--extra cu130` cannot install on macOS — PyTorch publishes no CUDA builds
  for it. That is expected, not a bug.
- Everything resolves from the committed `uv.lock`; never edit it by hand.

## Simulation extras

The base install includes classic-control environments (CartPole, Acrobot,
Pendulum). Heavier simulators are opt-in:

| Extra | Adds | Notes |
|---|---|---|
| `--extra mujoco` | MuJoCo environments (Hopper, HalfCheetah, ...) | Prebuilt wheels, CPU-friendly on all platforms |
| `--extra box2d` | LunarLander, BipedalWalker | Needs `swig` at build time; uv resolves a `swig` wheel, but if the build still fails install swig via your package manager (`brew install swig` / `apt install swig`) |
| `--extra benchmark` | pandas, pyarrow, matplotlib | Required for `roborl benchmark ...`; included in `make setup` |

robosuite and Gymnasium-Robotics are deliberately **not** dependencies yet;
they arrive with their roadmap phases.

## Developer setup

```bash
make setup       # uv sync --group dev --extra benchmark + pre-commit install
make check       # format check + lint + typecheck + fast tests — run before every commit
```

## Verify the install

```bash
uv run roborl demo
```

This runs a random agent through the entire pipeline (config → seeding →
device resolution → env factory → telemetry → summary). If it prints episode
statistics, your install works.

## Weights & Biases

Telemetry is off by default (`track=False`) — no account needed. Three modes:

| Mode | How | When |
|---|---|---|
| disabled | default; or `--no-track` | tests, CI, quick checks |
| online | `wandb login` once, then `--track` | normal experiment tracking |
| offline | `WANDB_MODE=offline` + `--track` | air-gapped machines; sync later with `wandb sync wandb/offline-run-*` |

The W&B entity comes from `--wandb-entity`, else the `WANDB_ENTITY`
environment variable, else your account default.

## Device selection

Every entry point takes `--device auto|cpu|cuda|mps` (default `auto`, which
resolves cuda > mps > cpu). Requesting an unavailable accelerator raises
instead of silently falling back to CPU.

MPS (Apple Silicon) caveats:

- MPS has no float64 support — observations must be cast to float32 before
  they reach the network (a standard step in our training templates).
- MPS determinism is weaker than CPU determinism. Debug reproducibility
  issues on CPU first (`--device cpu`); see [debugging-rl.md](debugging-rl.md).

## Headless MuJoCo rendering (Linux servers)

Video capture needs an OpenGL context. On a headless server set:

```bash
MUJOCO_GL=egl    # GPU-accelerated headless rendering (preferred)
MUJOCO_GL=osmesa # pure-software fallback (apt install libosmesa6-dev)
```

## Troubleshooting

- **`uv sync` rebuilds the world after switching extras** — expected when
  switching between `cpu`/`cu130` torch builds; the uv cache makes it fast
  the second time.
- **box2d build errors mentioning swig** — install swig system-wide (see
  table above), then `uv sync --extra box2d` again.
- **`ValueError: Device 'cuda' was requested but CUDA is not available`** —
  you asked for CUDA explicitly on a machine without it; use `--device auto`.
- **Videos not written** — pass `--capture-video`; the env needs
  `render_mode="rgb_array"` support (all classic-control and MuJoCo envs do).
