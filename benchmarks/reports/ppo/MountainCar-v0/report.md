# Verification report: ppo on MountainCar-v0

| | |
|---|---|
| Algorithm | `ppo` |
| Environment | `MountainCar-v0` |
| Commit | `088f3cc7cb19` (dirty) |
| Our runs | 5 |
| Reference runs | 6 (cleanrl) |
| Final window | last 10% of training |
| **Verdict** | **PASS** |

## Final performance (IQM over final window, 95% stratified bootstrap CI)

| Run set | IQM | 95% CI |
|---|---|---|
| roborl PPO | -200.00 | [-200.00, -200.00] |
| cleanrl | -200.00 | [-200.00, -200.00] |

## Sample efficiency

![learning curves](curves.png)

## Sources

- Ours: `runs/ppo-MountainCar-v0-s1-20260828T073252.csv`, `runs/ppo-MountainCar-v0-s2-20260828T073252.csv`, `runs/ppo-MountainCar-v0-s3-20260828T073252.csv`, `runs/ppo-MountainCar-v0-s4-20260828T073252.csv`, `runs/ppo-MountainCar-v0-s5-20260828T073328.csv`
- Reference: `.cache/benchref/openrlbenchmark/ppo/MountainCar-v0.parquet`

Verdict policy: see `docs/benchmarking.md`. A PASS means our final IQM's CI
overlaps the reference's (or IQM >= 90% of reference when the reference has
fewer than 3 runs). INVESTIGATE triggers the debugging protocol
(`docs/debugging-rl.md`) and a lab-notebook entry.
