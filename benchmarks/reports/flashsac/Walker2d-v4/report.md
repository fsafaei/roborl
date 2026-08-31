# Verification report: flashsac on Walker2d-v4

| | |
|---|---|
| Algorithm | `flashsac` |
| Environment | `Walker2d-v4` |
| Commit | `2461dc803d9a` (dirty) |
| Our runs | 5 |
| Reference runs | 5 (roborl SAC (verified)) |
| Final window | last 10% of training |
| **Verdict** | **INVESTIGATE** |

## Final performance (IQM over final window, 95% stratified bootstrap CI)

| Run set | IQM | 95% CI |
|---|---|---|
| roborl FlashSAC | 6123.55 | [5954.14, 6531.65] |
| roborl SAC (verified) | 4609.62 | [4204.40, 5058.98] |

## Sample efficiency

![learning curves](curves.png)

## Sources

- Ours: `runs/flashsac-Walker2d-v4-s1-20260830T023947.csv`, `runs/flashsac-Walker2d-v4-s2-20260830T024300.csv`, `runs/flashsac-Walker2d-v4-s3-20260830T024337.csv`, `runs/flashsac-Walker2d-v4-s4-20260830T024341.csv`, `runs/flashsac-Walker2d-v4-s5-20260830T024344.csv`
- Reference: `runs/sac-Walker2d-v4-s1-20260827T144739.csv`, `runs/sac-Walker2d-v4-s2-20260827T144740.csv`, `runs/sac-Walker2d-v4-s3-20260827T160151.csv`, `runs/sac-Walker2d-v4-s4-20260827T160225.csv`, `runs/sac-Walker2d-v4-s5-20260827T160231.csv`

Verdict policy: see `docs/benchmarking.md`. A PASS means our final IQM's CI
overlaps the reference's (or IQM >= 90% of reference when the reference has
fewer than 3 runs). INVESTIGATE triggers the debugging protocol
(`docs/debugging-rl.md`) and a lab-notebook entry.
