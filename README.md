# roborl

**Learning reinforcement learning for robotics by building it.**

[![CI](https://github.com/fsafaei/roborl/actions/workflows/ci.yml/badge.svg)](https://github.com/fsafaei/roborl/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Every algorithm here is implemented from scratch, **verified against
reference implementations** (primarily [CleanRL](https://docs.cleanrl.dev))
before it counts as done, and instrumented so its behavior can actually be
understood. The destination is contact-rich robotic manipulation; the road
starts deliberately simple, with classic control.

## Why this repo exists

**Implementing and verifying beats reading.** You don't understand an RL
algorithm until your from-scratch implementation matches a trusted reference
on standard benchmarks — with honest statistics, not a cherry-picked seed.
Verification against CleanRL's public benchmark runs is built into the
workflow here, not bolted on.

**Debugging and telemetry literacy are the actual skills.** Most of applied
RL is diagnosing silently-wrong training runs. This repo treats that as
first-class curriculum: a [debugging protocol](docs/debugging-rl.md), a
[guide to reading training metrics](docs/telemetry.md), and a lab notebook
of real investigations — the failure modes are documented next to the code
that produces them.

**The road goes toward contact-rich manipulation, starting simple on
purpose.** Classic control → continuous control → goal-conditioned RL →
manipulation (peg-in-hole, nut assembly, wiping). Deep algorithmic
understanding first; robotic complications once the foundations are proven.

Positioning among neighbors: [CleanRL](https://github.com/vwxyzjn/cleanrl)
is the reference oracle, [Spinning Up](https://spinningup.openai.com) the
theory companion; roborl is the learn-by-building-with-engineering-rigor
layer on top — typed, tested, CI-gated, with recorded design decisions.

## Quickstart

```bash
git clone https://github.com/fsafaei/roborl && cd roborl
uv sync                        # add --extra mujoco for MuJoCo envs
uv run roborl demo             # random agent on CartPole — verifies the whole pipeline
uv run pytest -m "unit or smoke"
```

Install matrix ([details](docs/setup.md)):

| Machine | Command |
|---|---|
| macOS (Apple Silicon → MPS) | `uv sync` |
| Linux with NVIDIA GPU | `uv sync` |
| Linux CPU-only / CI | `uv sync --extra cpu` |

**Weights & Biases** (optional): telemetry is off by default. `wandb login`
once, then re-run the demo with `--track` to see it live in W&B. On machines
without internet, `WANDB_MODE=offline uv run roborl demo --track` records
locally for later `wandb sync`.

## Repository map

```
src/roborl/
├── cli.py           # `roborl` CLI: demo, benchmark fetch|compare
├── config.py        # one frozen dataclass = one experiment
├── demo.py          # random-agent pipeline check — template for training scripts
├── utils/           # seeding, device resolution (cuda > mps > cpu)
├── envs/factory.py  # seeded env thunks with episode-stats & video wrappers
├── telemetry/       # W&B wrapper (online/offline/disabled) + canonical metric names
├── benchmark/       # reference fetching, IQM/CI statistics, plots, reports
└── algos/           # empty — algorithms arrive via the lifecycle below
docs/                # setup, telemetry, debugging, lifecycle, benchmarking, ADRs, lab notebook
benchmarks/reports/  # committed verification reports (the evidence)
tests/               # unit + smoke (CPU, offline, fast)
```

## Algorithms & status

Statuses: `planned → in progress → implemented → verified ✅` — where
`verified ✅` requires a committed verification report, linked in the last
column. Nothing here is implemented yet; the infrastructure came first.

| Algorithm | Envs | Status | Verified against | Report |
|---|---|---|---|---|
| SAC | Pendulum, MuJoCo | implemented | CleanRL | — |
| PPO (discrete) | CartPole, Acrobot, LunarLander | planned | CleanRL | — |
| PPO (continuous) | Pendulum, MuJoCo | planned | CleanRL | — |
| FlashSAC | MuJoCo, high-dimensional control | planned | published results ([Kim et al., 2026](https://arxiv.org/abs/2604.04539)) | — |
| HER + SAC | Fetch (Reach, Push, PickAndPlace) | planned | SB3/zoo + published results | — |

Beyond these, the roadmap ends in **vision-language-action (VLA) policies**
for manipulation — a research phase building on existing open VLA models
rather than a from-scratch reimplementation, so it lives in the roadmap
below rather than this table.

## How verification works

Each algorithm is verified by re-running it under the reference's exact
hyperparameters, seeds ≥ 5, on the same env version the reference used, and
comparing learning curves with the statistics few-seed RL actually needs:
**IQM with 95% stratified bootstrap confidence intervals** (Agarwal et al.,
NeurIPS 2021). `roborl benchmark fetch` pulls CleanRL's public runs from the
`openrlbenchmark` W&B entity; `roborl benchmark compare` aligns curves,
computes final-performance IQM with CIs, renders a markdown report + figure,
and issues a mechanical verdict: **PASS** when our CI overlaps the
reference's, **INVESTIGATE** otherwise — which triggers the
[debugging protocol](docs/debugging-rl.md) and a lab-notebook entry.
Algorithms without a CleanRL reference (FlashSAC, HER) verify against their
papers' published results or the SB3 zoo through the same harness, via the
corresponding reference adapters.

Reports are committed under `benchmarks/reports/` and are the only thing
that flips a status row to `verified ✅`. Details and thresholds:
[docs/benchmarking.md](docs/benchmarking.md).

## Telemetry

Every run logs CleanRL-compatible metric names (`charts/episodic_return`,
`losses/value_loss`, ...) against `global_step`, so roborl curves overlay
reference curves 1:1 in a single W&B workspace. roborl-specific additions
live in `diagnostics/` and `eval/` namespaces. Every run records its config,
git SHA, library versions, and resolved device — results that can't be
traced to a commit don't exist. What each metric means and what its failure
modes look like: [docs/telemetry.md](docs/telemetry.md).

## Roadmap

| Phase | Focus | Envs | Algorithms | Verification reference |
|---|---|---|---|---|
| 0 ✅ | Infrastructure: tooling, telemetry, benchmark harness, docs, CI | — | — (random-agent pipeline check) | — |
| 1 | Core model-free algorithms | CartPole-v1, Acrobot, Pendulum, (LunarLander w/ box2d), MuJoCo: Hopper, HalfCheetah, Walker2d | SAC, PPO (discrete + continuous) | CleanRL |
| 2 | Scaling off-policy RL | MuJoCo + high-dimensional control tasks from the paper | FlashSAC ([Kim et al., 2026](https://arxiv.org/abs/2604.04539)): SAC with few-update/large-batch scaling and weight/feature/gradient norm bounding | published results (+ reference code if released) |
| 3 | Goal-conditioned RL | Gymnasium-Robotics Fetch (Reach, Push, PickAndPlace) | HER + SAC | SB3/zoo + published results |
| 4 | Contact-rich manipulation & VLAs | robosuite: Lift, Door, Wipe, NutAssembly, TwoArmPegInHole | FlashSAC/SAC variants; demonstrations, residual RL; then vision-language-action policies (fine-tuning and evaluating open VLA models) as research threads | published results + ablation baselines |

Phase 1 uses locomotion *tasks* purely as standard verification benchmarks —
the destination is manipulation. Phases advance only when the previous
phase's algorithms are `verified ✅`.

## Learning resources

- Huang et al., [CleanRL: High-quality Single-file Implementations of Deep RL Algorithms](https://docs.cleanrl.dev) (JMLR 2022) — the verification oracle; public runs under the [openrlbenchmark](https://wandb.ai/openrlbenchmark/cleanrl) W&B entity
- Agarwal et al., [Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264) (NeurIPS 2021) + [rliable](https://github.com/google-research/rliable) — the evaluation methodology
- Huang et al., [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) (ICLR Blog Track 2022) — the model for our per-algorithm spec notes
- Kim et al., [FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control](https://arxiv.org/abs/2604.04539) (2026) — the roadmap's scaling phase
- Andy Jones, [Debugging Reinforcement Learning](https://andyljones.com/posts/rl-debugging.html) — foundation for our debugging protocol
- John Schulman, *The Nuts and Bolts of Deep RL Research*
- [OpenAI Spinning Up](https://spinningup.openai.com) — theory companion
- [Gymnasium](https://gymnasium.farama.org) / [Gymnasium-Robotics](https://robotics.farama.org) / [robosuite](https://robosuite.ai) / [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) / [W&B docs](https://docs.wandb.ai)
- Sutton & Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) (2nd ed.)

## Contributing & license

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). MIT
licensed ([LICENSE](LICENSE)).
