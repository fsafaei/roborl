# Verification report: ppo_continuous_action on HalfCheetah-v4

| | |
|---|---|
| Algorithm | `ppo_continuous_action` |
| Environment | `HalfCheetah-v4` |
| Commit | `9203c4a7ef14` (dirty) |
| Our runs | 5 |
| Reference runs | 9 (cleanrl) |
| Final window | last 10% of training |
| **Verdict** | **PASS** |

## Final performance (IQM over final window, 95% stratified bootstrap CI)

| Run set | IQM | 95% CI |
|---|---|---|
| roborl | 1621.67 | [1420.81, 2187.99] |
| cleanrl | 1851.36 | [1390.33, 3231.94] |

## Sample efficiency

![learning curves](curves.png)

## Sources

- Ours: `runs/ppo_continuous_action-HalfCheetah-v4-s1-20260828T101203.csv`, `runs/ppo_continuous_action-HalfCheetah-v4-s2-20260828T101203.csv`, `runs/ppo_continuous_action-HalfCheetah-v4-s3-20260828T101203.csv`, `runs/ppo_continuous_action-HalfCheetah-v4-s4-20260828T101203.csv`, `runs/ppo_continuous_action-HalfCheetah-v4-s5-20260828T101604.csv`
- Reference: `.cache/benchref/openrlbenchmark/ppo_continuous_action/HalfCheetah-v4.parquet`

Verdict policy: see `docs/benchmarking.md`. A PASS means our final IQM's CI
overlaps the reference's (or IQM >= 90% of reference when the reference has
fewer than 3 runs). INVESTIGATE triggers the debugging protocol
(`docs/debugging-rl.md`) and a lab-notebook entry.
