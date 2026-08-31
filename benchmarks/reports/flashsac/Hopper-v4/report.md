# Verification report: flashsac on Hopper-v4

| | |
|---|---|
| Algorithm | `flashsac` |
| Environment | `Hopper-v4` |
| Commit | `1f99c7c24adb` (s1-s3), `38223c3a39f8` (s4-s5) — identical algorithm source, both dirty |
| Our runs | 5 |
| Reference runs | 5 (roborl SAC (verified)) |
| Final window | last 10% of training |
| **Verdict** | **PASS** |

## Final performance (IQM over final window, 95% stratified bootstrap CI)

| Run set | IQM | 95% CI |
|---|---|---|
| roborl FlashSAC | 2968.38 | [2916.37, 3202.74] |
| roborl SAC (verified) | 3082.19 | [2603.61, 3388.67] |

## Sample efficiency

![learning curves](curves.png)

## Sources

- Ours: `runs/flashsac-Hopper-v4-s1-20260828T142321.csv`, `runs/flashsac-Hopper-v4-s2-20260828T142321.csv`, `runs/flashsac-Hopper-v4-s3-20260828T142321.csv`, `runs/flashsac-Hopper-v4-s4-20260830T023700.csv`, `runs/flashsac-Hopper-v4-s5-20260830T023804.csv`
- Reference: `runs/sac-Hopper-v4-s1-20260827T111703.csv`, `runs/sac-Hopper-v4-s2-20260827T111703.csv`, `runs/sac-Hopper-v4-s3-20260827T111703.csv`, `runs/sac-Hopper-v4-s4-20260827T111703.csv`, `runs/sac-Hopper-v4-s5-20260827T133406.csv`
- W&B runs (group `flashsac-Hopper-v4`, project `fsafaei/roborl`): [s1](https://wandb.ai/fsafaei/roborl/runs/8h4ayssd), [s2](https://wandb.ai/fsafaei/roborl/runs/nv5sngrb), [s3](https://wandb.ai/fsafaei/roborl/runs/rwcw1bia), [s4](https://wandb.ai/fsafaei/roborl/runs/8b4rfhp8), [s5](https://wandb.ai/fsafaei/roborl/runs/my1cewkz)

Verdict policy: see `docs/benchmarking.md`. A PASS means our final IQM's CI
overlaps the reference's (or IQM >= 90% of reference when the reference has
fewer than 3 runs). INVESTIGATE triggers the debugging protocol
(`docs/debugging-rl.md`) and a lab-notebook entry.
