# Verification report: ppo on Acrobot-v1

| | |
|---|---|
| Algorithm | `ppo` |
| Environment | `Acrobot-v1` |
| Commit | `4ed2bee432de` (dirty) |
| Our runs | 5 |
| Reference runs | 6 (cleanrl) |
| Final window | last 10% of training |
| **Verdict** | **PASS** |

## Final performance (IQM over final window, 95% stratified bootstrap CI)

| Run set | IQM | 95% CI |
|---|---|---|
| roborl | -83.99 | [-84.88, -82.81] |
| cleanrl | -84.62 | [-86.25, -83.69] |

## Sample efficiency

![learning curves](curves.png)

## Sources

- Ours: `runs/ppo-Acrobot-v1-s1-20260827T211826.csv`, `runs/ppo-Acrobot-v1-s2-20260827T211827.csv`, `runs/ppo-Acrobot-v1-s3-20260827T211827.csv`, `runs/ppo-Acrobot-v1-s4-20260827T211901.csv`, `runs/ppo-Acrobot-v1-s5-20260827T211909.csv`
- Reference: `.cache/benchref/openrlbenchmark/ppo/Acrobot-v1.parquet`

Verdict policy: see `docs/benchmarking.md`. A PASS means our final IQM's CI
overlaps the reference's (or IQM >= 90% of reference when the reference has
fewer than 3 runs). INVESTIGATE triggers the debugging protocol
(`docs/debugging-rl.md`) and a lab-notebook entry.
