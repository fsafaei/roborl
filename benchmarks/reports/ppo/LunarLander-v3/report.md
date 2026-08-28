# Verification report: ppo on LunarLander-v3 — NO REFERENCE (verdict N/A)

| | |
|---|---|
| Algorithm | `ppo` |
| Environment | `LunarLander-v3` |
| Commit | `4ed2bee432de` |
| Our runs | 5 |
| Reference runs | **0 — none exist** |
| Final window | last 10% of training |
| **Verdict** | **N/A (no reference)** |

Unlike the CartPole-v1 and Acrobot-v1 reports in this directory, this file
is written by hand because `roborl benchmark compare` cannot run: the
`openrlbenchmark/cleanrl` W&B project holds **no CleanRL `ppo` runs on any
LunarLander version**. Queried 2026-08-27: `exp_name="ppo"` covers exactly
`CartPole-v1` (6 runs), `Acrobot-v1` (6 runs), and `MountainCar-v0`
(6 runs); fetches for `LunarLander-v2` and `LunarLander-v3` both return
nothing. Per the integrity rules, no verdict is reported — an honest N/A
beats a fabricated PASS.

## Final performance (same statistics the harness computes)

IQM over the last 10% of training with 95% stratified bootstrap CI,
computed with `roborl.benchmark.stats.final_scores` / `iqm` /
`stratified_bootstrap_ci` on the same episode CSVs the harness would
consume:

| Run set | IQM | 95% CI |
|---|---|---|
| roborl (5 seeds) | 26.71 | [8.59, 40.67] |

Per-seed final scores (mean return over the last 10% of steps):
26.9, 47.5, 2.8, 20.2, 33.0 (seeds 1–5).

Context, not verdict: a random agent measured with `roborl demo` on
LunarLander-v3 (20k steps, 211 episodes, same commit) scores −185.3 ± 111.2;
the env's "solved" convention is 200. Five hundred thousand steps at
CleanRL's classic-control defaults lands PPO clearly airborne but short of
solved — consistent with expectations for this budget and these
hyperparameters, but there is no reference to test that statement against,
which is exactly why the verdict is N/A.

## Runs

| Seed | W&B run | Episode CSV |
|---|---|---|
| 1 | [mt25cixx](https://wandb.ai/fsafaei/roborl/runs/mt25cixx) | `runs/ppo-LunarLander-v3-s1-20260827T211910.csv` |
| 2 | [hlzjfl61](https://wandb.ai/fsafaei/roborl/runs/hlzjfl61) | `runs/ppo-LunarLander-v3-s2-20260827T211911.csv` |
| 3 | [f1u9crnl](https://wandb.ai/fsafaei/roborl/runs/f1u9crnl) | `runs/ppo-LunarLander-v3-s3-20260827T211945.csv` |
| 4 | [8rhrwnvp](https://wandb.ai/fsafaei/roborl/runs/8rhrwnvp) | `runs/ppo-LunarLander-v3-s4-20260827T211951.csv` |
| 5 | [1c950zuo](https://wandb.ai/fsafaei/roborl/runs/1c950zuo) | `runs/ppo-LunarLander-v3-s5-20260827T211952.csv` |

Config: CleanRL `ppo.py` defaults exactly (see `docs/algos/ppo.md`),
500k steps, seeds 1–5, CPU.

## If a reference materializes

Candidate future references: CleanRL adding LunarLander to its benchmark,
or the SB3 zoo's tuned PPO (different hyperparameters — would need the
deviation listed and a `reference_label` making the source obvious). Until
then the README counts PPO (discrete) as verified on CartPole-v1,
Acrobot-v1, and MountainCar-v0 only.
