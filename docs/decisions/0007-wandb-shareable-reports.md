# ADR 0007 — Shareable W&B reports per experiment

Status: accepted · Date: 2026-08-28

## Context

The committed harness reports (`benchmarks/reports/`, ADR 0005) are the
evidence behind every verdict, but they are static markdown + PNG. W&B
Reports give a shareable, interactive view — aggregated seed curves with
range bands, reference overlays, live-updating while a campaign runs — in
the style of the public RL baselines reports. Authoring them by hand in the
UI is unreproducible; wandb 0.29 moved programmatic report authoring out of
the core package into `wandb-workspaces`.

## Decision

Add **`wandb-workspaces`** to the `benchmark` extra and generate one report
per experiment with `roborl benchmark report-wandb`
(`src/roborl/benchmark/wandb_report.py`): a provenance intro block, then
one section per environment whose panels overlay aggregated run sets (mean
line, min/max band) selected by `config.exp_name` + `config.env_id` — our
runs, optionally a baseline algorithm from our project, optionally the
CleanRL reference from `openrlbenchmark/cleanrl`.

## Consequences

- Reports are regenerable from the CLI; the run-set filters pick up new
  seeds automatically, so a report created mid-campaign fills in live.
- The W&B report is a *view*, never evidence: verdicts still come
  exclusively from `roborl benchmark compare` and the committed reports.
- Each invocation creates a new report URL (the API has no idempotent
  upsert); regenerate sparingly and update the links in
  `docs/benchmarking.md`.
