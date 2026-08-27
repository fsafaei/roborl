# Verification report: sac on HalfCheetah-v4

| | |
|---|---|
| Algorithm | `sac` |
| Environment | `HalfCheetah-v4` |
| Commit | `cb421b3c883a` (dirty) |
| Our runs | 5 |
| Reference runs | 6 (CleanRL (openrlbenchmark)) |
| Final window | last 10% of training |
| **Verdict** | **PASS** |

## Final performance (IQM over final window, 95% stratified bootstrap CI)

| Run set | IQM | 95% CI |
|---|---|---|
| roborl | 10367.25 | [8127.52, 11703.52] |
| CleanRL (openrlbenchmark) | 9750.26 | [8608.42, 11082.99] |

## Sample efficiency

![learning curves](curves.png)

## Sources

- Ours: `runs/sac-HalfCheetah-v4-s1-20260827T133406.csv`, `runs/sac-HalfCheetah-v4-s2-20260827T133406.csv`, `runs/sac-HalfCheetah-v4-s3-20260827T133406.csv`, `runs/sac-HalfCheetah-v4-s4-20260827T144628.csv`, `runs/sac-HalfCheetah-v4-s5-20260827T144728.csv`
- Reference: `.cache/benchref/openrlbenchmark/sac_continuous_action/HalfCheetah-v4.parquet`
- W&B runs (group `sac-HalfCheetah-v4`, project `fsafaei/roborl`): [s1](https://wandb.ai/fsafaei/roborl/runs/pjly8kj1), [s2](https://wandb.ai/fsafaei/roborl/runs/eh5ns2hm), [s3](https://wandb.ai/fsafaei/roborl/runs/qyx96p5m), [s4](https://wandb.ai/fsafaei/roborl/runs/kw5qav8m), [s5](https://wandb.ai/fsafaei/roborl/runs/9rux9yvc)

Verdict policy: see `docs/benchmarking.md`. A PASS means our final IQM's CI
overlaps the reference's (or IQM >= 90% of reference when the reference has
fewer than 3 runs). INVESTIGATE triggers the debugging protocol
(`docs/debugging-rl.md`) and a lab-notebook entry.
