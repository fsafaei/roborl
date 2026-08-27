# ADR 0003 — Single readable training loops; rule of three for abstraction

Status: accepted · Date: 2026-08-27

## Context

RL libraries die pedagogically the day their training logic disappears into
a class hierarchy. CleanRL's core virtue is that every algorithm is one
readable top-to-bottom file — but pure single-file copies also duplicate
infrastructure (seeding, logging, env setup) that has nothing to teach and
everything to get subtly wrong.

## Decision

**Infrastructure is shared; algorithm math is local.** Cross-cutting
concerns — seeding, device resolution, config, telemetry, env factory,
benchmarking — live once in the `roborl` core. Each algorithm lives in its
own package under `src/roborl/algos/` as a single readable training loop,
duplicating small math helpers (GAE, target computation) rather than
importing a premature abstraction. A helper is promoted to shared core only
on its **third** use, via a dedicated PR that explains the abstraction.

## Consequences

- Every algorithm stays diffable line-by-line against its CleanRL
  counterpart — which is exactly how step 5 of the debugging protocol works.
- Some math is duplicated between two algorithms for a while; that is
  accepted cost, and watching the third use force the abstraction teaches
  *when* to abstract, not just how.
- Shared-core changes affect all algorithms and get reviewed accordingly.
