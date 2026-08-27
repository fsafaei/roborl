# ADR 0006 — GPU-optional, CPU-first device policy

Status: accepted · Date: 2026-08-27

## Context

Contributors and learners run heterogeneous hardware: Apple Silicon (MPS),
Linux+NVIDIA (CUDA), and CPU-only laptops/CI. A repo that quietly assumes
CUDA excludes most of them; a repo that silently falls back to CPU produces
confusing 10×-slower "GPU" runs.

## Decision

**Correctness is CPU-first: everything must run on CPU; a GPU is an
accelerator, never a requirement.** Every entry point takes
`--device auto|cpu|cuda|mps`; `auto` resolves cuda > mps > cpu. An explicit
request for an unavailable accelerator **raises** instead of falling back.
The resolved device is logged and recorded in run provenance. CI runs
CPU-only on Linux and macOS, proving the promise on every PR.

## Consequences

- Determinism debugging happens on CPU, where exact same-seed
  reproducibility holds; GPU runs are only statistically comparable
  (documented in `seed_everything`).
- MPS caveats are ours to handle: no float64 (cast observations to
  float32), weaker determinism — documented in docs/setup.md and the
  common-bugs checklist.
- Training templates must keep device transfers explicit and minimal, which
  also keeps them readable.
