# Verification report: flashsac on HalfCheetah-v4

| | |
|---|---|
| Algorithm | `flashsac` |
| Environment | `HalfCheetah-v4` |
| Commit | `2461dc803d9a` (dirty) |
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

Verdict policy: see `docs/benchmarking.md`. A PASS means our final IQM's CI
overlaps the reference's (or IQM >= 90% of reference when the reference has
fewer than 3 runs). INVESTIGATE triggers the debugging protocol
(`docs/debugging-rl.md`) and a lab-notebook entry.
