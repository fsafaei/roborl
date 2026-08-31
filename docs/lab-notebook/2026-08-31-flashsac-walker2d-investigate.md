# FlashSAC on Walker2d-v4: INVESTIGATE — resolved as significant improvement over the SAC baseline

**Date:** 2026-08-31 ·
**Trigger:** `roborl benchmark compare` verdict on
[benchmarks/reports/flashsac/Walker2d-v4](../../benchmarks/reports/flashsac/Walker2d-v4/report.md) ·
**Runs:** 5 seeds × 1M steps at commit `38223c3` (W&B group `flashsac-Walker2d-v4`; second campaign wave — identical algorithm source to wave 1's `1f99c7c`)

## Symptom

**INVESTIGATE**: final IQM 6123.55 [5954.14, 6531.65] for FlashSAC vs
4609.62 [4204.40, 5058.98] for the roborl-SAC reference set — CIs do not
overlap. Same shape as the
[HalfCheetah-v4 entry](2026-08-30-flashsac-halfcheetah-investigate.md): the
baseline is a *different algorithm* (our verified SAC), so the parity-based
verdict policy fires on "significantly better" exactly as it would on
"significantly worse".

## Diagnosis (same evidence pattern as the HalfCheetah entry)

Direction: FlashSAC's CI lower bound (5954) is far above SAC's upper bound
(5059); last-10 means 5091–6491 across seeds, with three seeds tightly at
6.3–6.5k. Returns are raw env returns via `RecordEpisodeStatistics` —
internal reward scaling cannot inflate them.

Health diagnostics across all 5 seeds:
`diagnostics/target_clamp_fraction` = 0 everywhere;
`diagnostics/grad_norm` 0.016–0.035 (bounded, no clipping);
`diagnostics/critic_feature_norm` 15.2–16.3 ≈ √256;
`diagnostics/reward_scale` settled at 197–227 with no drift.

## Resolution

No fix — a **documented, understood deviation**: FlashSAC significantly
outperforms the roborl SAC baseline on Walker2d-v4 at 1M steps. The
mechanical verdict stands in the report. Campaign-level picture across the
three envs: two significant improvements (HalfCheetah, Walker2d) and one
parity-with-tighter-CI (Hopper). Attribution to individual changes is the
ablation ladder's job (lifecycle Phase 7 follow-up).
