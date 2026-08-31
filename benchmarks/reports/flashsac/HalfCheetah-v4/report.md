# Verification report: flashsac on HalfCheetah-v4

| | |
|---|---|
| Algorithm | `flashsac` |
| Environment | `HalfCheetah-v4` |
| Commit | `38223c3a39f8` (dirty) |
| Our runs | 5 |
| Reference runs | 5 (roborl SAC (verified)) |
| Final window | last 10% of training |
| **Verdict** | **INVESTIGATE** |

## Final performance (IQM over final window, 95% stratified bootstrap CI)

| Run set | IQM | 95% CI |
|---|---|---|
| roborl FlashSAC | 12773.08 | [12128.06, 13084.16] |
| roborl SAC (verified) | 10367.25 | [8127.52, 11703.52] |

## Sample efficiency

![learning curves](curves.png)

## Sources

- Ours: `runs/flashsac-HalfCheetah-v4-s1-20260828T142321.csv`, `runs/flashsac-HalfCheetah-v4-s2-20260828T142321.csv`, `runs/flashsac-HalfCheetah-v4-s3-20260828T142321.csv`, `runs/flashsac-HalfCheetah-v4-s4-20260828T142321.csv`, `runs/flashsac-HalfCheetah-v4-s5-20260828T142321.csv`
- Reference: `runs/sac-HalfCheetah-v4-s1-20260827T133406.csv`, `runs/sac-HalfCheetah-v4-s2-20260827T133406.csv`, `runs/sac-HalfCheetah-v4-s3-20260827T133406.csv`, `runs/sac-HalfCheetah-v4-s4-20260827T144628.csv`, `runs/sac-HalfCheetah-v4-s5-20260827T144728.csv`
- W&B runs (group `flashsac-HalfCheetah-v4`, project `fsafaei/roborl`): [s1](https://wandb.ai/fsafaei/roborl/runs/3218wgif), [s2](https://wandb.ai/fsafaei/roborl/runs/mtyjk0zc), [s3](https://wandb.ai/fsafaei/roborl/runs/vnk16vr1), [s4](https://wandb.ai/fsafaei/roborl/runs/ld7d4sp5), [s5](https://wandb.ai/fsafaei/roborl/runs/8o35qht3)

Verdict policy: see `docs/benchmarking.md`. A PASS means our final IQM's CI
overlaps the reference's (or IQM >= 90% of reference when the reference has
fewer than 3 runs). INVESTIGATE triggers the debugging protocol
(`docs/debugging-rl.md`) and a lab-notebook entry.
