# sb3-her — the local SB3 SAC + HER reference (ADR 0008)

The verdict source for `her-sac` (`docs/algos/her.md`,
[ADR 0008](../../../docs/decisions/0008-her-verification-via-local-sb3-reference.md)):
Stable-Baselines3's `SAC` + `HerReplayBuffer`, **run locally** on the same
machine, the same `Fetch*-v4` env versions, and the same hyperparameters as
our runs. Stable-Baselines3 is not a roborl dependency — both scripts here
are PEP 723 inline-metadata scripts with exact pins, executed with
`uv run --script` in their own environment.

## Pins (authoring time, 2026-09-03)

| Package | Version | Why this one |
|---|---|---|
| `stable-baselines3` | 2.9.0 | current PyPI release; supports Gymnasium 1.x (`gymnasium<2.0,>=0.29.1`) |
| `gymnasium` | 1.3.0 | the version roborl's lockfile resolves |
| `gymnasium-robotics` | 1.4.2 | roborl's `fetch` extra |
| `mujoco` | 3.11.0 | roborl's `fetch` extra pin (`<3.12`: 3.12.0 breaks Fetch resets) |
| `torch` | 2.13.0 | roborl's lockfile |
| `numpy` / `pandas` | 2.5.2 / 3.0.5 | roborl's lockfile |

Upstream references consulted: SB3 master `3246f50` (2026-08-17, the Pass B
diff oracle); rl-baselines3-zoo master `bef2e8f` (2026-08-18) for the Fetch
recipe (`hyperparams/tqc.yml`, adapted TQC→SAC).

## Hyperparameters (identical on both sides)

`gamma 0.95 · tau 0.05 · learning_rate 1e-3 (actor, critics, entropy) ·
batch_size 2048 · buffer_size 1_000_000 · learning_starts 1_000 ·
net_arch [512, 512, 512] (2 critics) · ent_coef auto (target −dim(A) = −4) ·
HerReplayBuffer(n_sampled_goal=4, goal_selection_strategy="future") · no
VecNormalize`. Budgets: Reach 100k, Push 1M, PickAndPlace 1M; seeds 1–5.

Known, deliberate algorithm-side differences between SB3's SAC and our
CleanRL-shaped SAC (listed in every report; see the Pass B table in
`docs/algos/her.md`): SB3 updates actor and temperature every gradient
step where CleanRL delays by 2 and compensates; SB3's critic loss carries a
`0.5` factor; SB3's temperature loss differentiates `log α` rather than `α`.

## Commands

```bash
# one reference run (writes monitor/{env}-s{seed}.monitor.csv + .json sidecar)
uv run --script benchmarks/references/sb3-her/run_sb3_her.py \
    --env-id FetchPush-v4 --seed 1 --total-timesteps 1000000

# convert every Monitor CSV of an env into harness curves (+ success CSVs)
uv run --script benchmarks/references/sb3-her/to_curves.py \
    benchmarks/references/sb3-her/monitor/FetchPush-v4-s*.monitor.csv

# compare ours against them (manual-csv path; no new adapter code)
uv run roborl benchmark compare \
    --ours runs/her-sac-FetchPush-v4-s*.csv \
    --reference benchmarks/references/sb3-her/curves/FetchPush-v4-s*.csv \
    --algo her --env-id FetchPush-v4 --reference-label "SB3 SAC+HER (local)" \
    --ours-label "roborl her-sac"
```

## What was run, when

*Filled in as the reference campaign runs — every row names the sidecar
JSON (versions, git SHA of the runner, host, wall-clock, SPS).*

| Env | Seeds | Steps | Machine | Date | Sidecars |
|---|---|---|---|---|---|
| — | — | — | — | — | planned |
