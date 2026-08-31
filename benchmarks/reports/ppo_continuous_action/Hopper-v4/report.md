# Verification report: ppo_continuous_action on Hopper-v4

| | |
|---|---|
| Algorithm | `ppo_continuous_action` |
| Environment | `Hopper-v4` |
| Commit | `9203c4a7ef14` (one seed dirty) |
| Our runs | 5 |
| Reference runs | 9 (cleanrl) |
| Final window | last 10% of training |
| **Verdict** | **PASS** |

## Final performance (IQM over final window, 95% stratified bootstrap CI)

| Run set | IQM | 95% CI |
|---|---|---|
| roborl PPO (continuous) | 2336.36 | [1784.38, 2604.05] |
| cleanrl | 2176.59 | [1930.39, 2536.55] |

## Sample efficiency

![learning curves](curves.png)

## Sources

- Ours: `runs/ppo_continuous_action-Hopper-v4-s1-20260828T101604.csv`, `runs/ppo_continuous_action-Hopper-v4-s2-20260828T101606.csv`, `runs/ppo_continuous_action-Hopper-v4-s3-20260828T101607.csv`, `runs/ppo_continuous_action-Hopper-v4-s4-20260828T102021.csv`, `runs/ppo_continuous_action-Hopper-v4-s5-20260828T102052.csv`
- Reference: `.cache/benchref/openrlbenchmark/ppo_continuous_action/Hopper-v4.parquet`

Verdict policy: see `docs/benchmarking.md`. A PASS means our final IQM's CI
overlaps the reference's (or IQM >= 90% of reference when the reference has
fewer than 3 runs). INVESTIGATE triggers the debugging protocol
(`docs/debugging-rl.md`) and a lab-notebook entry.
