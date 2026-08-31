# Verification report: ppo_continuous_action on Walker2d-v4

| | |
|---|---|
| Algorithm | `ppo_continuous_action` |
| Environment | `Walker2d-v4` |
| Commit | `9203c4a7ef14` (dirty) |
| Our runs | 5 |
| Reference runs | 9 (cleanrl) |
| Final window | last 10% of training |
| **Verdict** | **PASS** |

## Final performance (IQM over final window, 95% stratified bootstrap CI)

| Run set | IQM | 95% CI |
|---|---|---|
| roborl PPO (continuous) | 2998.91 | [2436.08, 3355.76] |
| cleanrl | 2978.08 | [2364.76, 3514.43] |

## Sample efficiency

![learning curves](curves.png)

## Sources

- Ours: `runs/ppo_continuous_action-Walker2d-v4-s1-20260828T102053.csv`, `runs/ppo_continuous_action-Walker2d-v4-s2-20260828T102054.csv`, `runs/ppo_continuous_action-Walker2d-v4-s3-20260828T102458.csv`, `runs/ppo_continuous_action-Walker2d-v4-s4-20260828T102528.csv`, `runs/ppo_continuous_action-Walker2d-v4-s5-20260828T102541.csv`
- Reference: `.cache/benchref/openrlbenchmark/ppo_continuous_action/Walker2d-v4.parquet`

Verdict policy: see `docs/benchmarking.md`. A PASS means our final IQM's CI
overlaps the reference's (or IQM >= 90% of reference when the reference has
fewer than 3 runs). INVESTIGATE triggers the debugging protocol
(`docs/debugging-rl.md`) and a lab-notebook entry.
