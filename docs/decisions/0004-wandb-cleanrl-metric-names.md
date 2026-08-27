# ADR 0004 — W&B telemetry with CleanRL-compatible metric names

Status: accepted · Date: 2026-08-27

## Context

Verification means comparing our learning curves against CleanRL's
reference runs, which are public on W&B under the `openrlbenchmark` entity.
Comparison is only frictionless if our metrics share their names and x-axis.
Alternatives: TensorBoard (no hosted cross-repo overlay), MLflow (weaker for
curve-centric RL work), CSV-only (no live dashboards, no videos).

## Decision

Log to **Weights & Biases** with **CleanRL's exact metric names** for
everything CleanRL logs (`charts/...`, `losses/...`), `global_step` as the
x-axis everywhere. Our additions live in separate namespaces
(`diagnostics/`, `eval/`) so provenance stays obvious. Metric names are
constants in `src/roborl/telemetry/metrics.py`; hand-typed metric strings in
algorithm code are treated as bugs. Every run records config + git SHA +
versions + device. Offline mode (`WANDB_MODE=offline`) and disabled mode
(`track=False`) are first-class; tests and CI never talk to W&B.

## Consequences

- Our curves overlay 1:1 with reference runs in one workspace — the
  verification workflow's backbone.
- We inherit CleanRL's occasional naming quirks (e.g. `charts/SPS`) as the
  price of compatibility.
- The W&B dependency stays in the base install, but nothing requires an
  account until a user opts into `--track`.
