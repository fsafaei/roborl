# HER — Hindsight Experience Replay

Spec note per [algorithm-lifecycle.md](../algorithm-lifecycle.md) step 0,
written before the implementation. Roadmap **Phase 3 — goal-conditioned
RL**, delivered in two stages: `her-sac` (HER composed with the repo's
CleanRL-verified [SAC](sac.md)), then `her-flashsac` (the same HER
machinery composed with the verified [FlashSAC](flashsac.md) modules).

**Paper:** Andrychowicz et al., *Hindsight Experience Replay*, NeurIPS 2017
([arXiv:1707.01495](https://arxiv.org/abs/1707.01495)). Benchmark
definitions: Plappert et al., *Multi-Goal Reinforcement Learning:
Challenging Robotics Environments and Request for Research*, 2018
([arXiv:1802.09464](https://arxiv.org/abs/1802.09464)).
**Reference implementation (oracle):** Stable-Baselines3
[`HerReplayBuffer`](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/her/her_replay_buffer.py)
with the community-tuned Fetch recipe from
[rl-baselines3-zoo `hyperparams/tqc.yml`](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/tqc.yml).
**Verdict source:** SB3's SAC + `HerReplayBuffer` run locally under the
same recipe ([ADR 0008](../decisions/0008-her-verification-via-local-sb3-reference.md)).
The original OpenAI Baselines DDPG+HER is *history deliberately not copied*
(observation clipping/normalisation, Q-target clipping, MPI epoch
structure — DDPG-era tuning that SB3's SAC+HER works without).

**Source-of-truth protocol.** Pass A implements blind from this spec
(the SB3 buffer source stays unopened); every judgement call lands in
"Open questions" below. Pass B reads the SB3 source and fills the
component-by-component diff table at the end of this document — that table
is the deliverable that proves correctness. Running SB3 as a black-box
library (the reference runs) is allowed in any phase; the blindness applies
to reading the buffer's code.

## What HER is

A sparse-reward goal-conditioned task pays `-1` every step unless the goal
is achieved. A random policy on FetchPush essentially never achieves the
goal, so every episode is 50 steps of `-1` and TD learning has no gradient
signal to propagate. HER's move: after an episode, *pretend the goal was
something the agent actually achieved* — replay a transition with the
desired goal replaced by an achieved goal from later in the same episode,
and the reward recomputed under the substitute goal. Because the dynamics
do not depend on the goal, the relabeled transition is an ordinary valid
transition of the goal-conditioned MDP and any off-policy algorithm can
consume it. The critic learns "what reaches *nearby* goals" long before the
real goal is ever reached, and the sparse task becomes learnable.

**HER changes the data, not the algorithm.** The SAC update equations, twin
critics, temperature autotuning, polyak targets, `terminated`-only dones
with bootstrapping through truncation, the single non-autoresetting env
with explicit `reset()`, and the float32-at-the-buffer-boundary rule are
all untouched. What changes, exhaustively:

| # | Change | Replaces |
|---|---|---|
| 1 | Goal-conditioned envs: Fetch dict observations (`observation`, `achieved_goal`, `desired_goal`), success flag, `compute_reward` | flat Box observations |
| 2 | Episode-aware replay buffer with **sample-time goal relabeling** (`future`, k = 4) and reward recomputation | uniform transition ring buffer |
| 3 | Network input = `concat(observation, desired_goal)`; MLP sizes configurable (`512-512-512`) | raw observation into fixed `256-256` |
| 4 | Fetch recipe: `gamma 0.95`, `tau 0.05`, `lr 1e-3` for all three optimisers, `batch 2048` | CleanRL SAC defaults |
| 5 | Success-rate telemetry + periodic deterministic evaluation | training-episode returns only |

Two structural facts about Fetch the design leans on:

- **Episodes never terminate** — the task is infinite-horizon; episodes end
  only by truncation at 50 steps. Every stored done flag is 0, targets
  always bootstrap, and the value function approximates the infinite-horizon
  value, bounded in `[-1/(1-γ), 0] = [-20, 0]` at `γ = 0.95`. Relabeling
  must not manufacture terminations.
- **`achieved_goal` is part of the state**, stored explicitly. It is never
  recomputed from `observation` and never fed to the networks.

## The goal-env contract (preflight, 2026-09-03)

Confirmed against the installed packages by direct inspection
(`gymnasium 1.3.0`, `gymnasium-robotics 1.4.2`, `mujoco 3.11.0`):

| Env | `observation` | `achieved_goal` / `desired_goal` | action | `max_episode_steps` |
|---|---|---|---|---|
| `FetchReach-v4` | `Box(-inf, inf, (10,), float64)` | `Box(-inf, inf, (3,), float64)` | `Box(-1, 1, (4,), float32)` | 50 |
| `FetchPush-v4`, `FetchPickAndPlace-v4`, `FetchPushDense-v4` | `Box(-inf, inf, (25,), float64)` | same | same | 50 |

- **v4 is the newest registered version** (`Fetch*-v1` is also registered
  but targets the dead mujoco-py stack). All three dict pieces arrive as
  **float64** — cast once at the act/buffer boundary (MPS has no float64).
- Sparse reward: `0.0` (arrives as `-0.0`) iff `‖ag − dg‖₂ ≤ 0.05`
  (`distance_threshold = 0.05`, inclusive), else `-1.0`; computed on the
  **post-step** achieved goal. Verified for every step of a random episode:
  `reward == compute_reward(next_ag, dg, info)`. Dense variants return
  `−distance` (float64).
- `terminated` is always `False`; `truncated` at step 50.
  `desired_goal` is constant within an episode.
- `info["is_success"]` is present at **every step** as `numpy.float32`
  0.0/1.0 (not at reset — the reset info dict is empty).
- `compute_reward(achieved_goal, desired_goal, info)` exists on
  **`env.unwrapped` only** — the wrapper stack does not forward it. It is
  vectorised over leading batch dimensions: a `(5, 3)` pair returns shape
  `(5,)` (float32 for sparse); an unbatched `(3,)` pair returns a 0-d
  scalar. It ignores `info`; we pass `None`.
- Registration: importing `gymnasium_robotics` registers the ids (Gymnasium
  1.x has no entry-point autoloading). `envs/factory.py` therefore catches
  `gymnasium.error.NameNotFound`, lazily imports and registers, and retries
  once — with a clear "install the `fetch` extra" error otherwise.
- **mujoco pin.** `mujoco 3.12.0` breaks every Fetch reset
  (`set_joint_qpos` asserts `joint_type in (mjJNT_HINGE, mjJNT_SLIDE)`,
  and 3.12's enum no longer compares equal to the `numpy.int32` the model
  hands back; 3.3.7 through 3.11.0 all pass an isolated probe; upstream
  `main` still carries the same code). The `fetch` extra therefore pins
  `mujoco<3.12`; the plain `mujoco` extra is unaffected.
- Zoo recipe confirmed at rl-baselines3-zoo `master` = `bef2e8f`
  (2026-08-18; `tqc.yml` last touched in `f94cef4`, 2026-07-18):
  `FetchPush-v1: n_timesteps 1e6, buffer_size 1e6, batch_size 2048,
  gamma 0.95, learning_rate 1e-3, tau 0.05, HerReplayBuffer(future, 4),
  net_arch [512, 512, 512], n_critics 2`; `FetchPickAndPlace-v1` inherits
  it; `FetchSlide-v1` is 3e6 (out of scope). `sac.yml` carries only
  `FetchReach-v1` (20k steps, `[64, 64]`, `normalize: True`,
  `learning_starts 1000`, batch 256) — not copied except for
  `learning_starts`. Neither Push entry sets `learning_starts`, so the SB3
  default (100) would apply there.

## Storage — episode-major, staging then commit

Notation: batch `B`, observation dim `S`, goal dim `G`, action dim `A`,
episode capacity `N`, horizon `T = 50`.

```
capacity_episodes N = buffer_size // max_episode_steps        # 1_000_000 // 50 = 20_000

observations      (N, T, S)   float32      achieved_goals    (N, T, G)   float32
next_observations (N, T, S)   float32      next_achieved     (N, T, G)   float32
desired_goals     (N, T, G)   float32      actions           (N, T, A)   float32
rewards           (N, T)      float32      terminateds       (N, T)      float32
ep_len            (N,)        int64        # 0 = slot empty
```

`add(...)` appends one transition to a **staging** area for the episode in
flight; `commit_episode()` — called by the loop at `terminated or
truncated` — copies the staged episode into slot `pos`, sets `ep_len[pos]`,
advances `pos = (pos + 1) % N` (overwriting the oldest episode once full).
**The in-progress episode is never sampleable**: sampling sees committed
slots only (SB3's `ep_length > 0` validity mask, the paper's
episode-batched storage), at the cost of at most one episode of staleness.

`desired_goal` is stored per step even though Fetch holds it constant
within an episode — the buffer must not exploit a Fetch-specific invariant.
`__len__` = `ep_len.sum()` (transitions). The *true* next observation is
stored at episode end (non-autoresetting env, as in `sac.py`);
`terminateds` holds `terminated` only (all zeros on Fetch — asserted in the
smoke test — but the buffer stays general). The buffer takes
`compute_reward: Callable[[ndarray, ndarray, None], ndarray]` at
construction (the unwrapped env method, or a pure fake in tests) and imports
nothing from Gymnasium.

## Sampling with relabeling — the algorithm, exactly

`sample(batch_size, device)`; all index arrays are `(B,)`; NumPy's global
RNG throughout (as everywhere in the repo), so same-seed runs replay the
same minibatch *and* relabeling sequence:

```
# 1. which transitions — uniform over committed TRANSITIONS (SB3 semantics)
flat = randint(0, sum(ep_len), size=B)
ep   = searchsorted(cumsum(ep_len), flat, side="right")   # empty slots have zero width
t    = flat - (cumsum(ep_len)[ep] - ep_len[ep])
# (Pass A drew episode-then-step; identical on fixed-length Fetch, aligned
#  to SB3 in Pass B so variable-length episodes weight by length.)

# 2. real/virtual split — a fixed count, not a Bernoulli coin
p_her      = her_k / (her_k + 1)                     # k = 4  ->  0.8
nb_virtual = int(p_her * B)                          # B = 2048 -> 1638
virtual    = rows [0, nb_virtual); real = the rest

# 3. substitute goals for the virtual rows       # strategy = future | final | episode
future:  f = randint(t, ep_len[ep])                  # high-exclusive: f in [t, L-1]
final:   f = ep_len[ep] - 1
episode: f = randint(0, ep_len[ep])
goal[virtual] = next_achieved[ep, f]                 # achieved AFTER step f, i.e. ag_{f+1}
goal[real]    = desired_goals[ep, t]                 # stored goal, untouched

# 4. rewards
reward[virtual] = compute_reward(next_achieved[ep, t], goal[virtual], None)
reward[real]    = rewards[ep, t]                     # stored env reward

# 5. network inputs — the SAME substituted goal on both sides
obs      = concat([observations[ep, t],      goal], axis=-1)    # (B, S + G)
next_obs = concat([next_observations[ep, t], goal], axis=-1)    # (B, S + G)

# 6. dones — stored terminated only; NEVER derived from relabeled success
done = terminateds[ep, t]
```

Returned as SAC's frozen `ReplayBatch` shape (`observations, actions,
next_observations, rewards, dones`; rewards and dones 1-D), so the update
block is line-for-line `sac.py`. The buffer also exposes two scalars for
telemetry: the realized virtual fraction and the fraction of virtual
rewards equal to `0.0`.

Three semantic points, each a pitfall and a fixture:

- **`future` includes the transition's own successor.** `f = t` selects
  `ag_{t+1}` — "what I achieved one step later" — giving that transition
  reward `0`. This floor of achievable reward bootstraps learning; a stray
  `randint(t+1, L)` shrinks it and crashes on an empty range at `t = L−1`.
  `randint(t, L)` is never empty.
- **Goals come from `next_achieved`, and the recomputed reward uses
  `next_achieved[ep, t]`** — the Fetch reward is a function of the
  *post-step* state. Using `achieved_goals` in either place is a silent
  one-step-off bug that still learns something.
- **The substituted goal replaces `desired_goal` in both network inputs.**
  A mismatch trains `Q(s, a, g')` against a target at `(s', g'')` — the TD
  equation stops being about any single MDP.

## `her_sac.py` — deltas from `sac.py`, exhaustively

Anything not listed here survives a side-by-side diff untouched.

1. `HerSacConfig(ExperimentConfig)`: SAC's fields, plus/changed —

   | Field | Default | Note |
   |---|---|---|
   | `env_id` | `"FetchPush-v4"` | |
   | `total_timesteps` | `1_000_000` | Reach runs pass `100_000` |
   | `gamma` | `0.95` | Fetch recipe (zoo), not 0.99 |
   | `tau` | `0.05` | zoo, 10× SAC's default |
   | `policy_lr` / `q_lr` | `1e-3` / `1e-3` | one rate everywhere, incl. temperature |
   | `batch_size` | `2048` | zoo |
   | `learning_starts` | `1_000` | shared deviation from SB3's default 100 (ambiguity row 6) |
   | `net_arch` | `(512, 512, 512)` | actor and critics |
   | `her_enabled` | `True` | `False` = ablation R0/R1: stores identically, relabels nothing (`nb_virtual = 0`) |
   | `her_strategy` | `"future"` | `future / final / episode` |
   | `her_k` | `4` | `n_sampled_goal`; virtual fraction `k/(k+1)` |
   | `eval_interval` | `10_000` | env steps between eval passes |
   | `eval_episodes` | `20` | deterministic episodes per pass |

   Unchanged from SAC: `buffer_size 1_000_000`, `policy_frequency 2`
   (delayed-compensated, kept for `sac.py` diffability — ambiguity row 9),
   `target_network_frequency 1`, `autotune True` with target
   `−dim(A) = −4`, `track True`, `save_episodes` / `episode_dir`.
2. `Actor` / `SoftQNetwork` take `hidden_sizes` instead of the hard-coded
   256-256 (local copies, ADR 0003; same log-std tanh-squash bounds
   `[-5, 2]`, same action rescaling machinery — Fetch is already `[-1, 1]`).
3. Action selection and stepping run on
   `flatten(obs_dict) = concat(observation, desired_goal).astype(float32)`;
   the buffer receives the raw dict pieces.
4. On `terminated or truncated`: `buffer.commit_episode()`, log
   `diagnostics/success_rate` from the **final step's** `info["is_success"]`
   alongside return/length, then explicit `reset()`.
5. Every `eval_interval` steps: `eval_episodes` episodes on a separate,
   identically-wrapped env (factory, `seed + 1000`, no video), acting with
   the deterministic mean action; log `eval/success_rate` (final-step
   `is_success` mean), `eval/episodic_return`, `eval/episodic_length`. The
   eval env's RNG stream is its own; the training env and NumPy's global
   RNG are untouched (determinism test runs with eval on).
6. The update block is `sac.py`'s, consuming the HER buffer's
   `ReplayBatch`. Additionally, every 100 steps:
   `diagnostics/her_virtual_fraction`,
   `diagnostics/her_virtual_reward_zero_fraction`, and
   `diagnostics/q_lower_bound_violation` — the fraction of the minibatch's
   `min(Q1, Q2)` below `−1/(1−γ)·1.05` (`−21.0` at `γ = 0.95`). Logged,
   never clipped (Baselines clipped; SB3 does not; we follow SB3).
7. `_save_episode_log` writes the same 3-column CSV, plus
   `runs/{run_name}-success.csv` with `run_id, global_step,
   episodic_success` (0/1 per training episode).

Registered as `roborl her-sac`.

## Hyperparameters

The `her-sac` column is what we implement **and** what the local SB3
reference runs use — both sides share every knob.

| Parameter | `her-sac` (ours **and** SB3 reference) | zoo published (context) | `her-flashsac` (Stage 2) |
|---|---|---|---|
| env ids | `FetchReach/Push/PickAndPlace-v4` | `Fetch*-v1` (dead mujoco-py stack) | `FetchPush-v4`, `FetchPickAndPlace-v4` |
| `total_timesteps` | Reach `100_000`; Push, PickAndPlace `1_000_000` | Reach 20k; Push/P&P 1e6 (Slide 3e6) | `1_000_000` |
| `buffer_size` | `1_000_000` (= 20_000 episodes) | 1_000_000 | `1_000_000` |
| `gamma` | `0.95` | 0.95 | `0.95` (task-side; ambiguity row 8) |
| `tau` | `0.05` polyak | 0.05 | `0.01` EMA (FlashSAC recipe) |
| learning rate | `1e-3` — policy, critics, temperature | `learning_rate: 1e-3` | FlashSAC cosine `3e-4 → 1.5e-4` |
| `batch_size` | `2048` | 2048 | `512` |
| `learning_starts` | `1_000` | zoo omits → SB3 default 100 | `10_000` |
| net | MLP `512-512-512`, 2 critics | `net_arch=[512,512,512]`, `n_critics: 2` | FlashSAC residual nets (128/2, 256/2) |
| entropy | autotune, target `−dim(A) = −4` | `ent_coef: auto` | `sigma_tgt = 0.15` target |
| HER | `future`, `k = 4` → virtual fraction 0.8 | `future`, `n_sampled_goal: 4` | same |
| obs normalisation | **none** | none for Push/P&P (Reach's `sac.yml` used VecNormalize — dropped) | embedder BatchNorm (built in) |
| exploration | SAC entropy only | same | Zeta-repeated noise |

The recipe is the zoo's TQC Fetch entry adapted to SAC: two standard
critics instead of TQC's quantile ensemble, no top-quantile truncation.
Context, never verdicts: the zoo's published TQC+HER agent on
`FetchPush-v1` reports mean episode reward ≈ `−11.6 ± 6.2` at 1M steps
(huggingface `sb3/tqc-FetchPush-v1`, older config); SB3-grade SAC+HER on
Push/PickAndPlace typically ends near success rates of 0.9–1.0; Reach
saturates at 1.0 within a few thousand steps.

**Compute honesty (measured 2026-09-03, Apple Silicon Mac, 10 cores,
32 GB).** `batch 2048` through `512³` MLPs, six network passes per env
step:

| Loop / device | SPS (update phase) | Notes |
|---|---|---|
| `her-sac`, `--device mps` | 35–38 | Reach gate: 100k steps in 44 min; unaffected by concurrent CPU jobs |
| `her-sac`, `--device cpu`, alone | ≈ 26 | one process at ~110 % CPU — the matmuls hit Accelerate/AMX, torch thread count does not scale (1/2/4/8 threads all 16.8 ms per 512³ update) |
| `her-sac` CPU + SB3 CPU concurrently | 13.6 + 14.2 | **CPU aggregate is fixed at ≈ 27 SPS regardless of process count** |
| SB3 SAC+HER, CPU | ≈ 14 under contention (≈ 27 alone, by the same bound) | must run on CPU: SB3 hands float64 dict observations to the device and MPS has no float64 |

The machine therefore delivers ≈ 62 SPS in total (one MPS stream + one
CPU-equivalent), i.e. about 5.4M env steps per day. At that rate: Stage 1
verification alone (ours 10.5M + SB3 10.5M, and the SB3 half is CPU-only)
≈ 4 days; the 18M-step ablation ≈ 3.5 more days; Stage 2 ≈ 2 more —
≈ 9–10 days for the full ≈ 50M-step plan. This is the "unreasonable"
branch of the plan: seed counts and scope are the user's decision, recorded
below once made, not silently shrunk.

**Campaign status (2026-09-03): postponed until another machine is
available.** Everything up to and including the pilots is done; Phases
5–9 (SB3 reference seeds, our verification seeds, reports, ablation,
Stage 2) have not started. Runs stopped by hand when the laptop was
reclaimed, all at commit `5aa0612` (algorithm source):

| Run | W&B / file | Stopped at | Last `eval/success_rate` passes |
|---|---|---|---|
| Push pilot, seed 1, CPU | `9iuu5pug` | 116k of 300k | 0.60, 0.60, 0.80, 0.65, 0.65 (70k → 110k) |
| Push seed 1, MPS, 1M budget | `qdootgok` | 152k of 1M | 0.75, 0.80, 0.75, 0.80, 0.70, 0.80 (100k → 150k) |
| SB3 SAC+HER Push pilot, CPU | `runs/pilots-2026-09-03/sb3-*.monitor.csv` (local, not committed) | 55k of 300k | training success 0.10 over its last 20 episodes |

Both of our Push runs show the expected shape (near 0 for the first
~40k, then a steep climb to 0.6–0.8 by 100–150k). W&B marks the two runs
as crashed/killed; they are pilots, not verification evidence. Resume plan:
launch Phases 5–6 on the new machine from this branch with the commands in
`benchmarks/references/sb3-her/README.md` and
`roborl her-sac --env-id <env> --seed <s> --save-episodes`, then Phase 7
onward as written above.

## The implementation details that matter (pitfall catalogue)

Each item is a way this implementation silently trains something that is
not HER; each gets an assertion, a unit test, or both.

**Relabeling semantics**

1. **Goal from another episode.** Any index arithmetic letting `f` escape
   `[t, ep_len)` of *its own episode* poisons relabeling with unreachable
   goals. Fuzz: episodes with disjoint achieved-goal ranges; every
   substituted goal must fall in its source episode's range.
2. **`future` off-by-one.** `f = randint(t, L)`, goal = `next_achieved[f]`.
3. **Recompute against the wrong achieved goal.** Post-step
   `next_achieved[t]`, never `achieved_goals[t]`.
4. **Goal substituted on one side only.** `g'` in both `obs` and `next_obs`.
5. **Manufactured terminations.** Dones come from stored `terminated`,
   never from `reward == 0`; setting done there inflates Q toward 0.
6. **Coin-flip instead of fixed split.** Deterministic `int(0.8·B)`.
7. **k semantics.** `her_k = 4` means virtual fraction `4/5` (SB3's
   online-sampling ratio), not 4 extra stored copies (the paper's
   storage-time formulation; equivalent in expectation, not implemented).
8. **Relabeled reward scale.** Raw env-scale `{−1, 0}`; nothing rescales
   in Stage 1; Stage 2's normaliser divides at sample time but never learns
   statistics from relabeled rewards.

**Buffer plumbing**

9. **Sampling the episode in flight.** Commit-then-sample; a fresh buffer
   with only staged steps raises on `sample`.
10. **Deriving `achieved_goal` from `observation`.** Store all three pieces.
11. **The autoreset trap, doubled.** Under Gymnasium 1.x autoreset the step
    after truncation returns the *reset* observation; storing it corrupts
    the committed episode's last transition *and* its `next_achieved` —
    which relabeling then serves as goals. Explicit `reset()` is mandatory;
    the smoke test asserts stored `next_achieved[L−1]` differs from the
    next episode's `ag[0]`.
12. **float64 leaks.** Cast once at the buffer/act boundary.
13. **Capacity arithmetic.** `buffer_size // max_episode_steps` episodes;
    `__len__` counts transitions; overwrite per slot.
14. **RNG discipline.** All sampling through NumPy's global generator.

**Loop and environment**

15. **`γ = 0.95` actually plumbed** — fixture asserts the config value
    reaches `soft_td_target`.
16. **Success metric definition.** Final-step `is_success` (HER-literature
    convention), not "any step", not a per-step average.
17. **Eval isolation.** Separate env (`seed + 1000`); eval advances neither
    the training env, the buffer, nor NumPy's global RNG.
18. **`compute_reward` from the unwrapped env**, taken once at setup.
19. **Q lower bound is a diagnostic, not a clamp.**
20. **Dict spaces through the factory.** Space seeding recurses into Dict
    spaces; smoke runs with `capture_video=False`; one slow test covers
    video on.
21. **Telemetry constants** in `telemetry/metrics.py` only; success metrics
    live in `diagnostics/` and `eval/`.

**Stage 2 composition**

22. **Normaliser contamination** — statistics only from the collected raw
    env stream, accumulator physically next to env stepping.
23. **BatchNorm discipline unchanged** (FlashSAC pitfalls 12–16).
24. **Support fit under sparse rewards** — watch
    `diagnostics/target_clamp_fraction`.
25. **Goal into the embedder** — concatenate before the embedder so its
    BatchNorm normalises `[obs; goal]` jointly; no second normaliser.
26. **Zeta noise at eval** — temperature 0, deterministic `tanh(mean)`.

## Telemetry

CleanRL-shaped names reused as-is (`charts/episodic_return`,
`charts/episodic_length`, `charts/SPS`, `losses/qf1_values`, `losses/qf1_loss`,
`losses/qf2_*`, `losses/qf_loss`, `losses/actor_loss`, `losses/alpha`,
`losses/alpha_loss`; Stage 2 additionally reuses the FlashSAC
`diagnostics/` set). HER's additions (ADR 0004):

| Metric | Why it exists / healthy | Failure mode |
|---|---|---|
| `diagnostics/success_rate` | Final-step `is_success` per training episode; the task's real progress signal. Push/P&P: near 0 for the first ~50–150k steps, then climbing; Reach → 1 fast | Flat at 0 at 300k on Push with HER on → relabeling or reward recompute broken |
| `eval/success_rate` | Deterministic-policy success, the headline number | Far below the training curve → eval plumbing or exploration-vs-mean gap |
| `eval/episodic_return`, `eval/episodic_length` | Existing constants, now emitted | — |
| `diagnostics/her_virtual_fraction` | Realized relabel fraction; ≈ `k/(k+1)` = 0.8, constant | Anything else → split arithmetic |
| `diagnostics/her_virtual_reward_zero_fraction` | Fraction of relabeled rewards = 0. Analytic floor from own-successor hits alone: mean of 1/(T−t) = H_T/T ≈ 0.09 at T = 50; healthy ≈ 0.1–0.3 | ≈ 0 → wrong achieved-goal source or broken index math; ≈ 1 → degenerate goals |
| `diagnostics/q_lower_bound_violation` | Fraction of min-Q below `−1/(1−γ)·1.05` (= −21); ≈ 0 after warmup | Growing → divergence: done wiring, γ plumbing, relabel-both-sides |

## Verification plan (ADR 0008)

There is no CleanRL HER, and the zoo's published Fetch agents are TQC on
`-v1` envs — wrong algorithm, unrunnable env stack. The verdict source is
SB3's SAC + `HerReplayBuffer` run **locally** by
`benchmarks/references/sb3-her/run_sb3_her.py` (a PEP 723 script, pinned
versions, SB3 never enters roborl's dependency tree): same machine, same
`-v4` envs, the hyperparameter column above verbatim, 5 seeds ×
{Reach 100k, Push 1M, PickAndPlace 1M}. SB3's `Monitor` CSV
(`r, l, t, is_success`) is converted to the harness format
(`run_id, global_step, episodic_return`, `global_step` = cumulative `l`)
plus a parallel success CSV; `roborl benchmark compare` consumes both sides
through the existing `--reference <files>` path — no new adapter code.

Verdicts per [benchmarking.md](../benchmarking.md): IQM of
`charts/episodic_return` over the last 10% of training, 95% stratified
bootstrap CIs, PASS = CI overlap. On sparse Fetch, return and success are
near-affine (return ≈ −steps-to-goal; −50 = total failure); each report
says so once and shows both.

**Sanity gate (step 4):** `FetchReach-v4`, 1 seed, 100k steps,
`eval/success_rate ≥ 0.9` well before the end (slow test: ≥ 0.9 at budget).
**Pilot rule:** a `FetchPush-v4` run flat at 300k steps is broken, not
slow — debug before launching seeds.

*Sanity gate result (2026-09-03, commit `5aa0612`, W&B run `ki4k3fdm`,
`--device mps`, full recipe): PASSED.* `eval/success_rate` was 1.0 at the
first pass (10k steps) and 1.0 at 100k; training success 1.0 with returns
around −1 to −2 from ~10k on. Two transient eval dips (0.8 at 40k, 0.0 at
70k — one pass of 20 deterministic episodes, return −48.9) recovered by the
next pass; `diagnostics/q_lower_bound_violation` stayed at 0 throughout and
`her_virtual_reward_zero_fraction` sat at 0.7–0.8 (Reach's slow gripper
makes most future goals reachable within 5 cm — the 0.1–0.3 healthy band
is a Push figure). Wall-clock 44 min at 38 SPS end to end. The recorded
`git_dirty` flag comes from untracked files in the working tree; the
algorithm source is exactly `5aa0612`.

*Pilot (Phase 4b, `FetchPush-v4`, seed 1, commit `5aa0612`, `--device cpu`,
W&B run `9iuu5pug`, 300k budget — reported at 70k while still running):*
learning is clearly off the floor, so the pilot rule ("flat at 300k =
broken") is already settled. `eval/success_rate` by 10k-step pass:
0.15, 0.10, 0.05, 0.30, 0.55, 0.55, 0.60 (10k → 70k); training success per
10k bin: 0.04, 0.06, 0.07, 0.15, 0.38, 0.56, 0.59, 0.67. Diagnostics:
`her_virtual_reward_zero_fraction` 0.98 → 0.87 as the policy starts
moving the box; `q_lower_bound_violation` peaked at 0.03–0.06 around 6–8k
steps (the early transient while Q is still positive from the entropy
bonus) and sat at ≈ 0.002 from 20k on; `alpha` decayed 1.0 → 0.003;
`losses/qf1_values` ≈ −2.7 at 70k. A concurrent SB3 SAC+HER pilot with the
identical recipe (CPU, 300k) sat at success 0.10 / return −45 over its
last 20 episodes at 8k steps — too early to compare; both continue.

### Ablation — `FetchPush-v4`, 3 seeds per rung, 1M steps

| Rung | Config | The claim it isolates | Expectation (paper-shaped, to be confirmed or refuted) |
|---|---|---|---|
| R0 | SAC, sparse, `her_enabled=False` | Sparse Push is ~unlearnable without HER | flat near 0 |
| R1 | SAC, `FetchPushDense-v4`, no HER | Distance-shaped reward is not the fix | poor or unstable |
| R2 | HER `final`, k=4 | Only the episode's end state as goal | below `future` |
| R3 | HER `episode`, k=4 | Any state of the episode | below `final` |
| R4 | HER `future`, k=1 | Relabel ratio ½ | somewhat below k=4 |
| R5 | **HER `future`, k=4** | The method (= verification config; reuses Phase 6 seeds) | climbs to 0.9+ |
| R6 | HER `future`, k=8 | Relabel ratio 8/9 | ≈ k=4 |

Deliverable: one figure of `eval/success_rate` (or episodic success) as
IQM curves with CI bands per rung plus a final-window table, under
`benchmarks/reports/her/ablation/`, built with `stats.py`. The ablation is
descriptive; verdict language applies only to the SB3 comparison.

### Stage 2 — `her-flashsac` vs our `her-sac`

No external reference exists for FlashSAC + HER. Same envs (Push,
PickAndPlace), same 1M budget, 5 seeds (or 3 with the justification
stated), same harness, reported as "her-flashsac versus roborl her-sac" —
sample-efficiency claims only, `charts/SPS` reported honestly. Either
direction is a reportable research result at this operating point (1 env,
UTD 1, CPU recipe).

## Stage 2 composition rules (`her_flashsac.py`)

Gate: Stage 1's verification verdicts are in and reported first.
`her_flashsac.py` is `flashsac.py`'s wiring with the HER buffer and goal
plumbing swapped in; it **imports** FlashSAC's fixture-tested modules
(`layers`, `networks`, `distrib`, `rewards`, `noise`) under ADR 0009
(`buffer.py` is not imported — the HER buffer replaces it).

- Network input `concat(observation, desired_goal)` of dimension `S + G`
  into the embedder; its leading BatchNorm normalises obs and goal jointly
  and nothing else is added.
- Keep the `RescaleAction` wrap + assert (a no-op on Fetch).
- **Reward normaliser statistics update on the collected raw env stream
  only**, accumulator reset on `terminated or truncated` as in FlashSAC;
  relabeled rewards never touch the statistics; at update time real and
  virtual rewards are divided by the same denominator. Consistent by
  construction: relabeled rewards come from the same `{−1, 0}` reward
  function, so their discounted returns obey the same
  `|G| ≤ 1/(1−γ)` bound and scaled targets stay inside `[−5, 5]`.
- `γ = 0.95` task-side; every algorithm-side FlashSAC knob keeps its CPU
  recipe value (`batch 512`, `tau 0.01`, cosine `3e-4 → 1.5e-4`,
  `alpha_init 0.01`, `sigma_tgt 0.15`, Zeta `mu 2.0 / max 16`,
  `learning_starts 10_000`, n-step 1).
- HER knobs identical to Stage 1; same success/eval telemetry and
  diagnostics (the q-lower-bound diagnostic reads the categorical critic's
  expected values).

## Ambiguity register

Decisions already made where sources disagree or are silent. Rows 1–5 are
confirmed or overturned in Pass B.

| # | Question | Resolution encoded | Confidence |
|---|---|---|---|
| 1 | Relabel at store time (paper: k extra stored copies) vs sample time | Sample time, SB3 online-sampling semantics; equivalent in expectation, half the memory | **Pass B: confirmed** (`her_ratio = 1 − 1/(n_sampled_goal+1)`, applied per sampled batch) |
| 2 | `future` index range | `randint(t, L)` high-exclusive; goals from `next_achieved`; includes own successor | **Pass B: confirmed** (`np.random.randint(current_indices_in_episode, batch_ep_length)`, goals from `next_observations["achieved_goal"]`; SB3 documents the inclusivity) |
| 3 | Which achieved goal feeds the recomputed reward | `next_achieved[t]` (post-step) | **Pass B: confirmed** (`compute_reward(next_obs["achieved_goal"], obs["desired_goal"], infos)`, with a comment deriving exactly this) |
| 4 | Real/virtual split | Deterministic `int(0.8·B)` head of the batch | **Pass B: confirmed** (`nb_virtual = int(her_ratio * batch_size)`; `np.split` — no coin flip; SB3 returns real rows first, ours virtual first — order is irrelevant to the update) |
| 5 | Are real transitions' rewards recomputed too? | No — stored rewards; the recompute-consistency *test* proves they would match | **Pass B: confirmed** (`_get_real_samples` reads `self.rewards`) |
| 6 | `learning_starts` | 1000 (zoo's Reach entry) instead of SB3's default 100; applied to **both** sides, listed in the report | n/a — shared choice |
| 7 | Episode-uniform vs transition-uniform sampling | Pass A: episode-then-step. **Pass B: SB3 is transition-uniform (`choice(valid_indices)`); aligned** — identical on fixed-length Fetch, differs only on variable-length episodes | high — Pass B |
| 8 | γ for `her-flashsac` | 0.95 — γ is task-side (horizon); if Stage 2 stalls, a γ=0.99 probe is a *documented* deviation | judgement call |
| 9 | Actor update cadence | `sac.py`'s delayed-compensated `policy_frequency=2` (CleanRL) though SB3 updates per step; same average update count | medium — micro-deviation, listed in reports |
| 10 | Obs normalisation / clipping (Baselines had both) | None in Stage 1 (zoo Push/P&P has none); Stage 2 gets embedder BatchNorm inherently | high |
| 11 | Q-target clipping to `[−1/(1−γ), 0]` (Baselines) | Not implemented; logged as a diagnostic; SB3 does not clip | high |
| 12 | Success definition | Final-step `is_success` (HER-literature and SB3 Monitor convention) | high |
| 13 | Reach budget & recipe | 100k steps with the shared Push recipe (no VecNormalize, big nets) instead of zoo's 20k+VecNormalize+64²; both sides share it | n/a — shared choice |
| 14 | TQC→SAC adaptation | 2 standard critics, no quantile truncation | high |
| 15 | `timeouts` handling | SB3 stores `done·(1−timeout)`; we store `terminated` only — same semantics, already the repo rule | high |

## Open questions (Pass A judgement calls)

Recorded during implementation; each is confirmed or corrected in Pass B.

| # | Question the spec leaves open | Pass A resolution | Status |
|---|---|---|---|
| A1 | What `sample` does when only some slots are filled and `B` exceeds the transition count | Sample with replacement regardless of count; raise only when *no* episode is committed | **Pass B: confirmed** — `np.random.choice(valid_indices, replace=True)`; `RuntimeError` when nothing is valid |
| A2 | Whether `compute_reward`'s output dtype/shape is trusted | Cast to float32 and assert shape `(B,)` after every call — Fetch returns float32 for sparse and float64 for dense | **Pass B: confirmed** — SB3 casts `.astype(np.float32)`; no shape check (ours is stricter) |
| A3 | How `nb_virtual` behaves at `her_enabled=False` | `nb_virtual = 0`: the sampled batch is exactly a uniform batch of stored transitions with stored goals — the R0/R1 rungs differ from R5 in relabeling only | **Pass B: equivalent** — SB3 has no switch, but `n_sampled_goal = 0` gives `her_ratio = 0` and the same code path |
| A4 | Whether the staging area is bounded | Staging holds at most `max_episode_steps` transitions and `add` raises beyond it — a loop that forgets to commit fails loudly instead of growing without bound | **Pass B: n/a** — SB3 writes into the flat ring immediately and marks the episode valid at `done`; ours is a stricter layout with the same sampleability rule |
| A5 | Episodes shorter than `T` (a terminating goal env) | Stored with their true `ep_len`; the sampler draws `t` and `f` inside `[0, ep_len)` | **Pass B: confirmed** — SB3 tracks `ep_length` per transition and, after the row-7 alignment, weights episodes by length exactly as SB3 does |

## Pass B diff table

Component-by-component diff against `DLR-RM/stable-baselines3` master
(`3246f50`, 2026-08-17; `version.txt` 2.9.1a1; PyPI release 2.9.0), files
`stable_baselines3/her/her_replay_buffer.py`,
`her/goal_selection_strategy.py`, `common/buffers.py` (`DictReplayBuffer`),
`common/off_policy_algorithm.py`, `sac/sac.py`. Performed after Pass A
passed its unit tests and the FetchReach smoke test.

| Component | SB3 | Ours (Pass A) | Verdict |
|---|---|---|---|
| Storage layout | Flat transition ring `(buffer_size, n_envs)` with per-transition `ep_start` / `ep_length`; an episode becomes valid (`ep_length > 0`) at `done`; overwriting *any* transition of an old episode zeroes that whole episode's `ep_length` | Episode-major `(N, T, …)` with a staging area; `commit_episode()` at `terminated or truncated`; whole-slot overwrite | **matched** (different layout, same sampleability rule; SB3 transiently loses one partially overwritten episode at wrap — ours never does) |
| In-progress episode | `is_valid = ep_length > 0` masks it out; `RuntimeError` when no episode has finished | Never sampleable (staging); `ValueError` when nothing is committed | **matched** |
| Transition draw | `np.random.choice(valid_indices, size=B, replace=True)` — uniform over valid **transitions** | Pass A: uniform over episodes, then uniform step | **deviation — aligned in Pass B**: now `randint(0, Σ ep_len)` unravelled through `cumsum(ep_len)`; identical on fixed-length Fetch, differs only for variable-length episodes (ambiguity row 7) |
| Real / virtual split | `nb_virtual = int(her_ratio · B)`, `her_ratio = 1 − 1/(n_sampled_goal + 1)`; `np.split` → virtual = head of the drawn indices; output concatenates real first | Same count; virtual = head of the returned batch | **matched** (row order differs; the update is permutation-invariant) |
| `future` index | `randint(current_idx_in_episode, ep_length)` — inclusive of the current transition (documented) | `randint(t, L)` | **matched** |
| `final` index | `ep_length − 1` | same | **matched** |
| `episode` index | `randint(0, ep_length)` | same | **matched** |
| Goal source | `next_observations["achieved_goal"][transition_indices]` | `next_achieved[ep, f]` | **matched** |
| Reward recomputation | `env_method("compute_reward", next_obs["achieved_goal"], new desired goal, infos)`, `infos = [{}]*n` unless `copy_info_dict`; `.astype(np.float32)` | `compute_reward(next_achieved[ep, t], goal, None)`; float32 cast + shape assert | **matched** (`None` vs `{}` placeholder: Fetch ignores `info`; ours additionally asserts the `(B,)` shape) |
| Both sides | `obs["desired_goal"] = new_goals; next_obs["desired_goal"] = new_goals` | same goal concatenated onto `obs` and `next_obs` | **matched** |
| Real rows | Stored `rewards`, stored goals, no recomputation | same | **matched** |
| Done handling / timeouts | `dones · (1 − timeouts)`, `timeouts` from `info["TimeLimit.truncated"]` (`handle_timeout_termination=True`); `next_obs` is `terminal_observation` at `done` | store `terminated` only; true final observation from the non-autoresetting env | **matched** (ambiguity row 15) |
| Relabeled dones | Same masked `dones` for virtual rows — never derived from the new reward | stored `terminated` bitwise | **matched** |
| Normalisation hooks | `_normalize_obs` / `_normalize_reward` are no-ops without `VecNormalize` | none | **matched** (row 10) |
| RNG source | `np.random.choice` / `np.random.randint` (global NumPy RNG) | same generator (different draw order — irrelevant) | **matched** |
| `k` semantics | `n_sampled_goal` sets a *ratio* under online sampling, not extra stored copies | `her_k` → `k/(k+1)` | **matched** (row 1) |
| Warmup / update cadence | random actions while `num_timesteps < learning_starts`; `train()` once `num_timesteps > learning_starts`, `train_freq = 1 step`, `gradient_steps = 1` (UTD 1); `target_update_interval = 1` | same boundaries (CleanRL's) | **matched** |
| SAC update order (algorithm side) | per gradient step: temperature → critic → actor, **actor and temperature every step** | CleanRL: critic every step, actor + temperature every 2nd step ×2 (compensated) | **deviation, deliberate** (row 9; same average update count; kept for `sac.py` diffability; listed in reports) |
| Critic loss scale | `0.5 · Σ_i MSE_i` | `Σ_i MSE_i` (CleanRL) | **deviation, deliberate** — a constant factor on the critic gradient, which Adam normalises away (beyond ε) |
| Temperature loss | `−(log α · (log π + H̄).detach())` — gradient `−(log π + H̄)` | CleanRL `−(α · (log π + H̄))` — gradient `−α(log π + H̄)` | **deviation, deliberate** — same fixed point, different gradient scale; inherited from the verified SAC and already documented in `sac.md` |

**Outcome.** Every relabeling rule matched SB3 on the first blind pass; the
single alignment was the transition draw (no effect on Fetch). The
remaining deviations are all on the SAC side and are the same ones the
verified `sac.py` carries against SB3 — deliberate, listed here and in
every report.
