# FlashSAC on HalfCheetah-v4: INVESTIGATE — resolved as significant improvement over the SAC baseline

**Date:** 2026-08-30 ·
**Trigger:** `roborl benchmark compare` verdict on
[benchmarks/reports/flashsac/HalfCheetah-v4](../../benchmarks/reports/flashsac/HalfCheetah-v4/report.md) ·
**Runs:** 5 seeds × 1M steps at commit `1f99c7c` (W&B group `flashsac-HalfCheetah-v4`)

## Symptom

The harness returned **INVESTIGATE**: final IQM 12773.08 [12128.06, 13084.16]
for FlashSAC vs 10367.25 [8127.52, 11703.52] for the reference run set — the
CIs do not overlap. The reference here is **our own CleanRL-verified SAC**,
not a reference implementation of the same algorithm: there is no CleanRL
FlashSAC, so per the verification plan
([docs/algos/flashsac.md](../algos/flashsac.md)) the comparison is
"FlashSAC versus roborl SAC" at the same budget and env versions.

## Hypothesis

The PASS/INVESTIGATE policy tests *parity*: CI overlap is evidence that our
implementation of an algorithm matches a trusted implementation of the
*same* algorithm. Against a *different-algorithm baseline* the policy is
one-sided-blind — non-overlap fires for "significantly better" exactly as
it would for "significantly worse". Hypothesis: this INVESTIGATE is the
former — FlashSAC's stability changes genuinely improving on SAC — not an
implementation artifact inflating returns.

## Experiment / evidence

Direction: FlashSAC's CI lower bound (12128) sits above SAC's upper bound
(11704) — the deviation is an improvement, seed-consistent (last-10 means
12847–13234 across all 5 seeds).

Return integrity: `charts/episodic_return` comes from
`RecordEpisodeStatistics` on raw env rewards; the adaptive reward scaling
is internal to the critic update and cannot inflate reported returns.
Actions reach the env in [-1, 1] via `RescaleAction` (asserted at
construction).

Health diagnostics (seed 1 final values, same picture on all seeds):
`diagnostics/target_clamp_fraction` = 0 (targets never leave the fixed
[-5, 5] support — the adaptive reward scaling works);
`diagnostics/reward_scale` settled at ≈ 389;
`diagnostics/grad_norm` ≈ 0.03 and `diagnostics/param_norm` ≈ 123, both
bounded with **no gradient clipping**, and `diagnostics/critic_feature_norm`
≈ 14.8 ≈ √256 — the paper's three bounded-norm claims hold in practice.
`charts/SPS` ≈ 7.6 under 8-way CPU contention vs SAC's much higher rate —
the expected honest cost at this operating point.

Implementation trust: the implementation passed a component-by-component
Pass B diff against the authors' reference code (adjudicated table in the
algo doc), hand-computed fixture tests, gradient-isolation tests, and the
Pendulum sanity gate before any verification run.

Plausibility against the paper: qualitatively consistent with the paper's
claim that the stability changes improve over SAC on MuJoCo at the CPU
recipe. We make no "matches the paper" claim — its curves are not digitised.

## Fix / resolution

No fix: this is a **documented, understood deviation** — FlashSAC
significantly outperforms the roborl SAC baseline on HalfCheetah-v4 at
1M steps. The report keeps its mechanical INVESTIGATE verdict (the harness
tests parity, and honesty beats relabeling); this entry is the diagnosis.
Follow-up that would strengthen attribution: the ablation ladder
(lifecycle Phase 7), which assigns the improvement to individual changes
rather than the six-change bundle.
