# ADR 0002 — uv with a single lockfile and torch index extras

Status: accepted · Date: 2026-08-27

## Context

The repo promises "fresh clone to working install in minutes on any
machine, GPU or not." PyTorch complicates this: PyPI wheels bundle CUDA on
Linux (huge downloads for CPU-only boxes), while lean CPU and pinned-CUDA
wheels live on separate PyTorch index URLs. Alternatives considered: conda
(heavy, second ecosystem), plain pip + requirements files (no cross-platform
lock), Docker (excluded for now — deliberate simplicity, see prompt-era
decision log).

## Decision

Use **uv** with a single committed `uv.lock`. A plain `uv sync` gives
working PyPI wheels on every platform (MPS on Apple Silicon, CUDA-enabled on
Linux). Optional extras select torch wheel sources per uv's official PyTorch
guide: `--extra cpu` (lean CPU wheels; CI and CPU boxes) and `--extra cu130`
(pinned CUDA index), declared mutually exclusive via `[tool.uv] conflicts`
with explicit `[[tool.uv.index]]` entries. Simulation extras (`mujoco`,
`box2d`, `benchmark`) keep the base install light. Dev tools live in the
`dev` dependency group.

## Consequences

- One command, reproducible everywhere; the lockfile is never hand-edited.
- `--extra cu130` cannot install on macOS (PyTorch ships no macOS CUDA
  builds) — documented, expected.
- The CUDA extra name tracks PyTorch's current index and will be renamed as
  PyTorch moves (cu130 → cu1xx); that rename is a routine chore, not a
  design change.
