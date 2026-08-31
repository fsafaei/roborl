# Verification report: sac on Walker2d-v4

| | |
|---|---|
| Algorithm | `sac` |
| Environment | `Walker2d-v4` |
| Commit | `088f3cc7cb19` (dirty) |
| Our runs | 5 |
| Reference runs | 6 (CleanRL (openrlbenchmark)) |
| Final window | last 10% of training |
| **Verdict** | **PASS** |

## Final performance (IQM over final window, 95% stratified bootstrap CI)

| Run set | IQM | 95% CI |
|---|---|---|
| roborl SAC | 4609.62 | [4204.40, 5058.98] |
| CleanRL (openrlbenchmark) | 3846.62 | [3335.72, 4538.48] |

## Sample efficiency

![learning curves](curves.png)

## Sources

- Ours: `runs/sac-Walker2d-v4-s1-20260827T144739.csv`, `runs/sac-Walker2d-v4-s2-20260827T144740.csv`, `runs/sac-Walker2d-v4-s3-20260827T160151.csv`, `runs/sac-Walker2d-v4-s4-20260827T160225.csv`, `runs/sac-Walker2d-v4-s5-20260827T160231.csv`
- Reference: `.cache/benchref/openrlbenchmark/sac_continuous_action/Walker2d-v4.parquet`
- W&B runs (group `sac-Walker2d-v4`, project `fsafaei/roborl`): [s1](https://wandb.ai/fsafaei/roborl/runs/d1uuxa37), [s2](https://wandb.ai/fsafaei/roborl/runs/2eawdcee), [s3](https://wandb.ai/fsafaei/roborl/runs/b4a2rbmy), [s4](https://wandb.ai/fsafaei/roborl/runs/p2i2eg78), [s5](https://wandb.ai/fsafaei/roborl/runs/e8w0628s)

Verdict policy: see `docs/benchmarking.md`. A PASS means our final IQM's CI
overlaps the reference's (or IQM >= 90% of reference when the reference has
fewer than 3 runs). INVESTIGATE triggers the debugging protocol
(`docs/debugging-rl.md`) and a lab-notebook entry.
