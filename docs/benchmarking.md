# Benchmarking and verification

The repo's core claim is that implementations are **verified against
standard baselines**. This document makes that claim mechanical: what we
compare against, how we run, which statistics we report, and what
PASS/INVESTIGATE mean. Thresholds here are revisable policy — changing them
takes a PR to this file, not a quiet edit to code.

## Reference sources

References are pluggable adapters (`src/roborl/benchmark/fetch.py`), all
producing the same curves format (`run_id, global_step, episodic_return`):

| Adapter | Status | What it is |
|---|---|---|
| `openrlbenchmark` | implemented | CleanRL's public benchmark runs, fetched read-only from the `openrlbenchmark` W&B entity and cached as parquet under `.cache/benchref/` |
| `cleanrl-local` | planned | Re-run pinned CleanRL scripts in an isolated uv environment for same-machine, same-env-version parity |
| `sb3-zoo` | planned | For algorithms CleanRL lacks (e.g. HER) |
| `manual-csv` | implemented (trivially) | Published tables / arbitrary local curves via `roborl benchmark compare --reference <files>` |

Confirm the exact CleanRL project and `exp_name` per algorithm in CleanRL's
docs at fetch time.

## Env-version parity rule

Verification runs happen on the **same env id and version the reference
used** (e.g. `HalfCheetah-v4` if that's what CleanRL ran), because
verification is about algorithm correctness, not environment novelty.
Development and showcase runs may use the latest env versions afterwards.

## Protocol per algorithm/env

- **Seeds:** ≥5. For expensive MuJoCo budgets ≥3 is acceptable, stated
  explicitly in the report — fewer seeds means wider CIs and a weaker claim;
  that tradeoff must be visible, not hidden.
- **Hyperparameters:** the reference's, exactly. Every deviation is listed
  in the report.
- **Budget:** the reference's total-timestep budget.
- **Tracking:** all verification runs are tracked to W&B.

## Statistics

Methodology follows Agarwal et al., *Deep RL at the Edge of the Statistical
Precipice* (NeurIPS 2021); implemented directly in
`src/roborl/benchmark/stats.py` (~50 lines, unit-tested against
hand-computed values — see ADR 0005):

- Curves are aligned on a common `global_step` grid by interpolation.
- The sample-efficiency curve reports pointwise **IQM** (interquartile mean:
  the mean of the middle 50% of scores) with **95% stratified bootstrap CI**
  bands.
- Final performance is the IQM over the **last 10% of training**, again with
  95% stratified bootstrap CIs.
- Never compare lone means of 2 seeds. The point is teaching honest
  evaluation; an inconclusive honest result beats a conclusive dishonest one.

## Verdict policy

| Condition | Verdict |
|---|---|
| Our final IQM's 95% CI overlaps the reference's 95% CI | **PASS** |
| Reference has < 3 runs (too few to bootstrap): our IQM ≥ 90% of reference IQM | **PASS** |
| Otherwise | **INVESTIGATE** |

INVESTIGATE is not failure; it is a trigger. It starts the
[debugging protocol](debugging-rl.md) and ends in a lab-notebook entry —
either a fix and a re-run, or a documented, understood deviation.

## Reports

```bash
roborl benchmark fetch --algo dqn --env-id CartPole-v1
roborl benchmark compare --ours <our curve files> \
    --reference .cache/benchref/openrlbenchmark/dqn/CartPole-v1.parquet \
    --algo dqn --env-id CartPole-v1
```

`compare` renders `benchmarks/reports/<algo>/<env>/report.md` (header with
algorithm/env/commit/seed counts, final-performance table with CIs, verdict,
sources) plus `curves.png` (curves with CI bands, final-performance bars).
Reports are **committed** — they are the evidence behind every `verified ✅`
in the README status table. Keep them small and curated.
