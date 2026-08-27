# ADR 0005 — Verification methodology and statistics

Status: accepted · Date: 2026-08-27

## Context

Few-seed deep RL results are notoriously unreliable; comparing lone means is
how folklore results happen. Agarwal et al. (NeurIPS 2021) established IQM
with stratified bootstrap CIs as the standard for honest few-run evaluation,
with `rliable` as the reference implementation. We must decide both the
methodology and whether to depend on rliable.

## Decision

Adopt the rliable methodology: curves aligned on a common `global_step`
grid; pointwise **IQM with 95% stratified bootstrap CI bands** for sample
efficiency; **IQM over the last 10% of training** for final performance.
Verdicts: PASS on CI overlap with the reference (or IQM ≥ 90% of reference
when the reference has < 3 runs); otherwise INVESTIGATE, which triggers the
debugging protocol. **Implement the statistics directly** in
`src/roborl/benchmark/stats.py` (~50 lines, unit-tested against
hand-computed values) instead of depending on rliable: the implementation is
small, understanding it is part of the curriculum, and it avoids rliable's
dependency friction on current Python versions. Thresholds live in
`docs/benchmarking.md` as revisable policy.

## Consequences

- Every "verified ✅" claim traces to a committed report produced by this
  machinery — verdicts come from the harness, not eyeballing.
- We own ~50 lines of statistics code and its correctness; the unit tests
  against hand-computed values and the rliable paper keep it honest.
- If our needs grow (performance profiles, probability of improvement),
  revisit adopting rliable directly.
