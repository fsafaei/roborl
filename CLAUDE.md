# CLAUDE.md

Guidance for Claude Code sessions in this repository.

## Project

`roborl` — learning reinforcement learning for robotics by building it:
algorithms implemented from scratch, verified against reference baselines
(primarily CleanRL), instrumented to be understood. Educational repo held to
professional engineering standards, on a trajectory toward contact-rich
manipulation. Read `README.md` for the front door, `docs/` for methodology.

## Commands

```bash
make setup        # dev env (uv sync --group dev --extra benchmark) + pre-commit hooks
make check        # fmt-check + lint + typecheck + fast tests — REQUIRED before every commit
make fmt          # auto-format + autofix
make test         # unit + smoke (the CI suite)
make test-all     # includes slow tests
uv run roborl demo                       # random-agent pipeline check
uv run roborl benchmark fetch|compare    # verification harness (needs benchmark extra)
uv run pytest tests/unit/test_stats.py -k iqm    # run a single test file / selection
```

Extras: `uv sync --extra mujoco|box2d|benchmark`; `--extra cpu` (lean CPU
torch, Linux CI) conflicts with `--extra cu130`. Never hand-edit `uv.lock`
(`uv add`/`uv lock` only); CI fails on a stale lockfile (`uv lock --check`).

## Architecture

```
src/roborl/
├── cli.py          # tyro subcommands (demo, benchmark); heavy imports stay lazy
├── config.py       # ExperimentConfig frozen dataclass; run_name/group derivation
├── demo.py         # random-agent pipeline check — the template for training scripts
├── utils/          # seeding.py (seed_everything), device.py (resolve_device)
├── envs/factory.py # make_env thunk: RecordEpisodeStatistics, RecordVideo, seeding
├── telemetry/      # logger.py (W&B wrapper: online/offline/disabled), metrics.py (canonical names)
├── benchmark/      # fetch.py (reference adapters), stats.py (IQM/CIs), plots.py, report.py
└── algos/          # EMPTY until algorithms land — each in its own package
```

**Architecture policy (ADR 0003): infrastructure is shared; algorithm math
is local.** Each algorithm is a single readable top-to-bottom training loop
in its own package, duplicating small math helpers rather than importing a
premature abstraction. Promote a helper to shared core only on its third use
("rule of three"), via a dedicated PR. Every algorithm must stay diffable
against its CleanRL counterpart.

## Conventions

- ruff (line 100, Google docstrings) + mypy; full type hints on public
  functions. Never weaken lint/type rules to get green — fix the code or add
  a narrowly scoped ignore with a one-line justification.
- Code must stay Python 3.10-compatible (ruff `target-version = "py310"`;
  CI tests 3.10 and 3.12 on Linux + macOS) — no 3.11+-only syntax.
- pytest markers: `unit`, `smoke`, `slow`. Bare `uv run pytest` defaults to
  `-m "unit or smoke"` via addopts; slow tests need `make test-all` or
  `-m ""`. **No test touches the network**;
  W&B disabled/offline in tests and CI; reference data comes from committed
  fixtures.
- Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `ci:`,
  `chore:`); small scoped commits; branch-and-PR workflow per
  CONTRIBUTING.md. `make check` before every commit — never commit red.

## Commit attribution

This is a public repo under the author's name. Commits and PR descriptions
must not mention Claude, Claude Code, or any AI assistance:

- Do **not** add `Co-Authored-By: Claude ...` trailers.
- Do **not** add "Generated with Claude Code" footers to PR bodies.
- Commits are authored and SSH-signed as Farhad Safaei via existing global
  git config (`commit.gpgsign = true`); do not override author, email, or
  signing settings.

## Telemetry rules (ADR 0004, docs/telemetry.md)

- Metric names mirror CleanRL exactly (`charts/...`, `losses/...`); roborl
  additions live in `diagnostics/` and `eval/`. Constants live in
  `src/roborl/telemetry/metrics.py` — **never hand-type metric strings**.
- x-axis is `global_step` (env steps) everywhere.
- Every run records config + git SHA + dirty flag + versions + resolved
  device. Run identity: project `roborl`, group `{exp_name}-{env_id}`, name
  `{exp_name}-{env_id}-s{seed}-{timestamp}`.

## Algorithm definition of done (docs/algorithm-lifecycle.md)

0. Spec note in `docs/algos/<algo>.md` (paper + CleanRL reading; the
   implementation details that matter) →
1. single-loop implementation in `src/roborl/algos/<algo>/` →
2. math unit-tested against hand-computed fixtures →
3. CPU smoke test →
4. solves a trivial env →
5. verification runs (≥5 seeds, reference hyperparameters, same env
   version and budget, tracked) →
6. `roborl benchmark compare` → committed report in `benchmarks/reports/` →
7. on INVESTIGATE: debugging protocol + lab-notebook entry →
8. finalize algo doc, extend telemetry table, flip README status →
9. PR with checklist, CI green.

## Integrity rules

- **Never fabricate results.** Every reported number traces to a W&B run id
  and commit SHA.
- Benchmark verdicts come from the harness (`roborl benchmark compare`),
  not eyeballing.
- README status flips to `verified ✅` only with a committed report linked.
- An honest "planned" beats an aspirational lie, everywhere in the repo.

## Dependency policy

Ask the user before adding any dependency; record significant ones in an ADR
(`docs/decisions/`).

## Gotchas

- MPS: no float64 — cast observations to float32; weaker determinism than
  CPU (debug reproducibility on CPU).
- Headless MuJoCo on Linux servers needs `MUJOCO_GL=egl` (or `osmesa`).
- box2d needs swig at build time (`brew install swig` / `apt install swig`).
- Gymnasium 1.x autoreset semantics: the step after termination returns the
  reset observation — naive replay-buffer writes at episode boundaries store
  corrupted transitions.
- `--extra cu130` cannot install on macOS (no CUDA builds); expected.
- W&B offline mode: `WANDB_MODE=offline` + later `wandb sync`; disabled
  mode is `track=False` (default in demo/tests).
