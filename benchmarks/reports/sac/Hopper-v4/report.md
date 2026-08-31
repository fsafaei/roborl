# Verification report: sac on Hopper-v4

| | |
|---|---|
| Algorithm | `sac` |
| Environment | `Hopper-v4` |
| Commit | `088f3cc7cb19` (dirty) |
| Our runs | 5 |
| Reference runs | 6 (CleanRL (openrlbenchmark)) |
| Final window | last 10% of training |
| **Verdict** | **PASS** |

## Final performance (IQM over final window, 95% stratified bootstrap CI)

| Run set | IQM | 95% CI |
|---|---|---|
| roborl SAC | 3082.19 | [2603.61, 3388.67] |
| CleanRL (openrlbenchmark) | 2366.30 | [2045.23, 2720.89] |

## Sample efficiency

![learning curves](curves.png)

## Sources

- Ours: `runs/sac-Hopper-v4-s1-20260827T111703.csv`, `runs/sac-Hopper-v4-s2-20260827T111703.csv`, `runs/sac-Hopper-v4-s3-20260827T111703.csv`, `runs/sac-Hopper-v4-s4-20260827T111703.csv`, `runs/sac-Hopper-v4-s5-20260827T133406.csv`
- Reference: `.cache/benchref/openrlbenchmark/sac_continuous_action/Hopper-v4.parquet`
- W&B runs (group `sac-Hopper-v4`, project `fsafaei/roborl`): [s1](https://wandb.ai/fsafaei/roborl/runs/h9j5vi9s), [s2](https://wandb.ai/fsafaei/roborl/runs/16qlvtcg), [s3](https://wandb.ai/fsafaei/roborl/runs/03y58kl8), [s4](https://wandb.ai/fsafaei/roborl/runs/d3lzct5v), [s5](https://wandb.ai/fsafaei/roborl/runs/qnb5n6ry)

Verdict policy: see `docs/benchmarking.md`. A PASS means our final IQM's CI
overlaps the reference's (or IQM >= 90% of reference when the reference has
fewer than 3 runs). INVESTIGATE triggers the debugging protocol
(`docs/debugging-rl.md`) and a lab-notebook entry.
