# FlashSAC — Fast and Stable Off-Policy RL for High-Dimensional Robot Control

Spec note per [algorithm-lifecycle.md](../algorithm-lifecycle.md) step 0.
Written before the implementation (Pass A: blind, from the paper and a
reconciled spec only, without opening the reference code). The Pass B diff
table at the bottom is filled in after a structured comparison against the
reference implementation.

**Paper:** Kim et al., *FlashSAC: Fast and Stable Off-Policy Reinforcement
Learning for High-Dimensional Robot Control*, arXiv:2604.04539 (RSS 2026).
**Reference implementation:** <https://github.com/Holiday-Robot/FlashSAC>
(MIT, package `flash_rl`, agent under `flash_rl/agents/flashSAC/`).
**Base algorithm:** our verified SAC ([sac.md](sac.md), PASS against CleanRL
on Hopper-v4 / HalfCheetah-v4 / Walker2d-v4).

## Scope of this implementation

The paper's headline is a **scaling** result: 1024 parallel GPU-simulated
environments, update-to-data ratio 2/1024, wall-clock reduced from hours to
minutes. We implement the authors' own **CPU/MuJoCo recipe** instead: one
environment, 1M steps, batch 512, UTD 1, on HalfCheetah-v4 / Hopper-v4 /
Walker2d-v4 — the exact envs our SAC is already verified on. At this scale
we are testing **half the paper**: the *stability* contributions
(architecture, distributional critic, reward scaling, entropy target, noise
repetition), not the *scaling* contributions (massive parallelism, tiny
UTD), which need a GPU simulator we do not have. Any improvement measured
here is attributable to the stability half only, and the reports say so.

## What FlashSAC is

Plain SAC with six changes. Everything else — twin critics, clipped
double-Q, a fresh reparameterised next action, a learned temperature,
polyak targets, a uniform replay buffer — is unchanged from [sac.md](sac.md).

| # | Change | Replaces |
|---|---|---|
| 1 | Residual MLP with BatchNorm + terminal RMSNorm, actor and critic | 256-256 ReLU MLP |
| 2 | Unit weight normalisation after every optimiser step | nothing |
| 3 | Cross-batch value prediction: one forward pass over `cat([current, next])` | two separate forward passes |
| 4 | Categorical distributional critic on fixed support [-5, 5] + adaptive reward scaling | scalar critic + MSE |
| 5 | Entropy target from a fixed action std (σ_tgt = 0.15) | `-dim(A)` |
| 6 | Temporally-correlated exploration: noise repeated for a Zeta-distributed run length | fresh noise every step |

The thesis: with a large replay distribution you can drive the
update-to-data ratio *down* and compensate with model size and batch size —
but only if the critic is prevented from accumulating bootstrapping error.
Changes 1–4 are all norm-bounding devices serving that one goal.

## Architecture

### Primitive layers (all with a `normalize_parameters()` hook)

- **UnitLinear** — bias-free linear, orthogonal init;
  `normalize_parameters()` renormalises every *output row* of the weight to
  unit L2 norm over its input features.
- **UnitBatchNorm** — standard BatchNorm1d semantics (momentum 0.01 in
  PyTorch's convention `new = 0.99·old + 0.01·batch`, eps 1e-5);
  `normalize_parameters()` rescales the *concatenation* `[γ; β]` jointly to
  L2 norm `sqrt(d)`.
- **UnitRMSNorm** — RMSNorm (`F.rms_norm`, torch ≥ 2.4, eps 1e-6);
  `normalize_parameters()` rescales the weight to L2 norm `sqrt(d)`.

### Blocks

- **Embedder**: `UnitBatchNorm(in_dim)` → `UnitLinear(in_dim, hidden)` —
  normalisation comes *first*; this leading BatchNorm over the raw input
  **is** the observation normaliser. FlashSAC has no separate running
  obs-normalisation wrapper.
- **Block** (inverted bottleneck, expansion 4):
  `x + relu(BN(W2 · relu(BN(W1 · x))))` — i.e.
  `w1 → BN → ReLU → w2 → BN → ReLU → + residual`, **no** activation after
  the residual add.

### Actor (hidden 128, 2 blocks)

`Embedder → Block → Block → UnitRMSNorm → NormalTanhPolicy`. The policy
head has separate `UnitLinear` weights + free bias parameters for mean and
log-std; log-std is tanh-squashed into **[-10, 2]** (not SAC's [-5, 2]).
Action is `tanh(u)` with **no rescaling** — the log-prob carries no
`action_scale` term, so the env must be wrapped with `RescaleAction` to
[-1, 1] and the bounds asserted. The tanh Jacobian correction uses the
numerically stable form `2·(log 2 − u − softplus(−2u))` with no epsilon and
no clamp. Evaluation action is `tanh(mean)`.

### Critic (hidden 256, 2 blocks, ensemble E = 2, categorical head)

Both critics computed as **one ensemble** with a leading dimension `(E, B,
·)` via `einsum`-based `EnsembleUnitLinear` (per-member orthogonal init,
per-member row normalisation) and per-member BatchNorm statistics — never
two separate modules. Head: `EnsembleUnitLinear(hidden, 101)` + free bias,
`log_softmax` over 101 atoms on fixed support **[-5, +5]** (bin width 0.1);
scalar Q is the expectation of the categorical distribution.

Parameter count sanity: on HalfCheetah-v4 (S=17, A=6) about 2.17M critic +
0.27M actor ≈ 2.44M — the paper's "2.5M-parameter, 6-layer network"
(6 weight layers along one path: embedder 1 + 2 blocks × 2 + head 1).
Asserted in a unit test.

## Update equations

### Distributional TD target (under `no_grad`, α read inside)

1. Fresh next action from the actor in BN-**eval** mode:
   `a', log π(a'|s')`.
2. **Cross-batch**: one forward pass of the *target* critic in BN-**train**
   mode over `cat([s, s'], 0), cat([a, a'], 0)`; chunk and keep the second
   half → `q'ᵢ (E, B)`, `log pᵢ (E, B, n)`.
3. Clipped double-Q, distributional form: `j = argmin_i q'ᵢ` on the
   **expected values**, then gather member *j*'s **whole distribution**
   (a per-atom min of two probability vectors is not a probability vector).
4. Shift every atom:
   `z_target = r + γⁿ · (z − α·log π(a'|s')) · (1 − d)`, clamped to
   [v_min, v_max]. `d` is **terminated only**; `r` is the reward-normalised
   reward (below). The entropy bonus enters by shifting every atom by
   `−α·log π`, the distributional analogue of the scalar soft value. At a
   true termination the target collapses to a point mass at `r`.
5. Two-point projection onto the support with the mass-conserving split
   `m_l += p·(1−f)`, `m_u += p·f` where `f = b − floor(b)`,
   `u = min(l+1, n−1)` — this form keeps total mass 1 even when the target
   lands exactly on an atom or is clamped to the boundary; the classic C51
   `(u−b)/(b−l)` split silently loses that mass.

### Critic loss (every update)

Cross-batch again on the **online** critic (BN-train), first half:
cross-entropy `−Σ m · log p(s,a)`, averaged over ensemble and batch.
After `optimizer.step()` + LR-scheduler step: `normalize_parameters()`
under `no_grad`.

### Target EMA (every update, τ = 0.01)

`p_t ← (1−τ)·p_t + τ·p` over **parameters only**. The target critic's
BatchNorm running statistics are its own, advanced by its own train-mode
forward passes — never copied, never EMA'd.

### Actor + temperature (every 2nd update — a plain skip, not compensated)

**Order (Pass B correction): the actor and temperature update runs *before*
the critic update**, so the critic's target is built with the just-updated
actor and temperature. Pass A had this reversed.

Cross-batch through the **actor** (BN-train) over `cat([s, s'], 0)`, keep
the first half `a_π, log π`. Critic *parameters* frozen with
`requires_grad_(False)` around a BN-**eval** critic call (gradient flows
through the action only; that BN asymmetry — train in the critic loss, eval
here — is deliberate):

`L_actor = E[α·log π − min_i Qᵢ(s, a_π)]`, α detached.

Temperature: `α = exp(log α)`, `log α` init `log(0.01)`;
`L_α = α · (H − H_target)` with `H = −E[log π]` **detached**. Same fixed
point as CleanRL's form. Entropy target from a fixed per-dimension action
std σ_tgt = 0.15:

`H_target = 0.5 · A · log(2πe·σ_tgt²) ≈ −0.4782·A`

— roughly half as negative as SAC's `−A`, so the target policy is *more*
stochastic, and the one number transfers across action dimensionalities.

After each actor / temperature step: scheduler step + (actor)
`normalize_parameters()`.

### Learning-rate schedule

All three optimisers (Adam, PyTorch defaults, no weight decay) share **one
schedule horizon**: `decay_steps = total env steps × UTD`, with
`warmup_steps = int(1e-6 · decay_steps)` (≤ 1 step — effectively none),
cosine 3e-4 → 1.5e-4. Each optimiser steps its *own* scheduler per its own
gradient update, so the critic ends at ~1.5e-4 while the actor and
temperature — stepping every 2nd update — end **mid-cosine at ~2.25e-4**.
That is the reference's behaviour (Pass B, ambiguity row 16), not a bug.
**No gradient clipping anywhere** — the paper's "gradient norm bounding" is
a consequence of the weight and feature norm constraints, not an explicit
clip; `diagnostics/grad_norm` shows it stays bounded.

## Adaptive reward scaling

Rewards are stored **raw** in the buffer and normalised on the sampled
batch at update time. Statistics are maintained on the *collected stream*,
per transition:

```
done = terminated OR truncated          # BOTH here, unlike the TD target
G    = gamma * (1 - done) * G + reward  # discounted-return accumulator
G_max_seen = max(G_max_seen, |G|)
G_rms.update(G)                         # running mean/var (Chan parallel update)
```

At update time: `r_norm = r / max(sqrt(G_rms.var + 1e-8), G_max_seen / 5.0)`.
The first branch keeps rewards at unit scale; the second guarantees the
discounted returns fit inside the critic's fixed support [-5, +5]. Turning
reward scaling off while keeping the fixed support is not a valid
configuration — targets leave the support, get clamped, and the critic
saturates at a boundary atom while looking superficially healthy
(`diagnostics/target_clamp_fraction` is the health signal).

## Exploration: temporally-correlated noise

Exploration noise is held constant for a run of `k` steps drawn from a Zeta
(power-law) distribution `pmf(k) ∝ k^−2` truncated to {1..16}, sampled by
inverse-CDF on a precomputed table. Per action selection (no_grad, actor in
BN-eval mode): `action = tanh(mean + std · noise · T)` with T = 1 in
training and 0 in evaluation; the noise tensor is per-environment, the run
length is shared across environments (reference semantics — moot with one
env but kept so the vectorised phase matches). This noise is used **only
for acting**; the log-probabilities in the losses come from fresh
`rsample()` draws inside the update.

## Hyperparameters

The CPU/MuJoCo column is what we implement and verify; the GPU column is
recorded for completeness and for the later vectorised phase.

| Parameter | CPU / MuJoCo (**ours**) | GPU sims (paper's headline) |
|---|---|---|
| `num_envs` | 1 | 1024 |
| `total_timesteps` | 1_000_000 | 50_000_896 (from `run_isaaclab.sh`; paper says "50M") |
| `buffer_size` | 1_000_000 | 10_000_000 |
| `learning_starts` | 10_000 | 100_000 |
| `batch_size` | 512 | 2048 |
| `updates_per_env_step` (UTD) | 1 | 2 / 1024 |
| `n_step` | 1 | 3 per `run_isaaclab.sh`; paper's table says 1 — **unresolved conflict** |
| `gamma` | 0.99 | 0.99 (0.97 on some tasks) |
| `tau` (target EMA, weight on source) | 0.01 | 0.01 |
| `actor_update_period` | 2 | 2 |
| `actor_hidden` / `actor_blocks` | 128 / 2 | 128 / 2 |
| `critic_hidden` / `critic_blocks` | 256 / 2 | 256 / 2 |
| `expansion` | 4 | 4 |
| `n_atoms` | 101 | 101 |
| `v_min` / `v_max` | -5.0 / +5.0 | -5.0 / +5.0 |
| `normalize_reward` / `G_max` | true / 5.0 | true / 5.0 |
| `lr` init / peak / end | 3e-4 / 3e-4 / 1.5e-4 cosine | same |
| `alpha_init` | 0.01 | 0.01 |
| `sigma_tgt` | 0.15 | 0.15 |
| `noise_zeta_mu` / `noise_zeta_max` | 2.0 / 16 | 2.0 / 16 |
| AMP | off | fp16 autocast (temperature update outside it) |

Sources: CPU column from `scripts/run_mujoco.sh` +
`configs/agent/flashSAC.yaml` in the reference repo, cross-checked against
the paper; GPU column from `scripts/run_isaaclab.sh` + the same YAML. The
paper itself does not print most of these values.

## The implementation details that matter

1. **Shapes are asserted, not trusted.** `reward`, `terminated`, and the
   entropy term are reshaped to `(B, 1)` explicitly before the atom shift;
   a `(B,)` tensor broadcasting against `(B, n)` only *raises* when
   `B ≠ n_atoms` — at `B == 101` it silently produces a transposed target.
   Mirrors `sac.py::soft_td_target`'s 1-D guard.
2. **`chunk` order.** `cat([current, next])` → `chunk(2)[0]` is current,
   `[1]` is next. Backwards, the loss still goes down. Unit-tested with
   distinguishable halves.
3. **BatchNorm mode discipline.** Rollout/eval paths: `training=False`
   always (a train-mode BN on batch 1 destroys the forward pass and poisons
   the running stats). Critic loss and target construction: `training=True`
   as a *single* concatenated forward (two separate train-mode calls give
   Q(s,a) and Q(s',a') different normalisation statistics — different
   units). Critic inside the actor loss: `training=False` — deliberate
   asymmetry.
4. **Gradient isolation.** α read inside `no_grad` in the target (else the
   critic loss backprops into `log α`); critic params frozen via
   `requires_grad_(False)` in the actor loss (not `no_grad` — the actor
   gradient must flow *through* the action into the critic); `α.detach()`
   in the actor loss and `log π` detached in the temperature loss.
5. **`normalize_parameters()` at construction and after every
   `optimizer.step()`, under `no_grad`, in-place** — inside the graph it
   corrupts Adam state. The init call matters: orthogonal init only gives
   unit rows where `out ≤ in`, so wide layers start off-manifold
   (Pass B finding). `β` starts at
   zeros, so after the first update the joint `[γ; β]` rescale fixes its
   norm regardless of magnitude; that is genuinely what the reference does
   (flagged in the Pass B diff, not "fixed").
6. **Two done flags, two answers.** `terminated` only in the TD target;
   `terminated OR truncated` resets the reward normaliser's accumulator.
7. **Env plumbing.** Single non-autoresetting env with explicit `reset()`
   (`sac.py`'s solution to Gymnasium 1.x autoreset), `RescaleAction` to
   [-1, 1] with asserted bounds, float32 at the buffer boundary.
8. **`alpha_init = 0.01`, not 1.0** — with the higher entropy target,
   starting at 1.0 dominates the target's atom shift early in training.
9. **`n_step = 1` in this phase.** The reference's n-step return uses
   `gamma ** n` unconditionally even when a mid-window truncation cut the
   accumulation (an approximation); deferred, and documented when it lands.

## Telemetry

CleanRL-named where the quantity is the same: `charts/episodic_return`,
`charts/episodic_length`, `charts/SPS`, `charts/learning_rate`,
`losses/qf1_values`, `losses/qf2_values` (expected values of each member's
distribution), `losses/qf_loss` (**cross-entropy**, not MSE — its magnitude
is not comparable to SAC's), `losses/actor_loss`, `losses/alpha`,
`losses/alpha_loss`.

New `diagnostics/` metrics (constants in `telemetry/metrics.py`, rows in
[telemetry.md](../telemetry.md)):

| Metric | Why it exists |
|---|---|
| `diagnostics/target_clamp_fraction` | fraction of `z_target` hitting the support bounds — the single best FlashSAC health signal |
| `diagnostics/reward_scale` | the normaliser's denominator; should settle, not drift |
| `diagnostics/return_rms_var` | running variance of the discounted return |
| `diagnostics/target_dist_entropy` | entropy of the projected target distribution; collapse to a spike means the critic is over-confident |
| `diagnostics/critic_feature_norm` | RMSNorm output norm — the paper's "feature norm bounded" claim, made visible |
| `diagnostics/param_norm`, `diagnostics/grad_norm` | the paper's other two bounded-norm claims |
| `diagnostics/noise_repeat_len` | mean sampled run length; should sit near the Zeta mean |
| `diagnostics/target_entropy` | constant, but logged so a run's entropy target is in its record |

## Verification plan

No CleanRL reference exists for FlashSAC, so two comparisons through the
existing harness (IQM over the last 10% of training, 95% stratified
bootstrap CIs):

1. **Against our own verified SAC** — same envs, same 1M budget, 5 seeds
   each, via `--reference` on the SAC episode CSVs. The defensible claim,
   reported as "FlashSAC versus roborl SAC", never "matches the paper".
2. **Against the paper's published curves** where they can be digitised
   faithfully (`manual-csv` adapter); skipped honestly otherwise.

Plus the **ablation ladder** (paper §6.3), one config flag per rung,
3 seeds each on one env: SAC baseline → +residual/BN blocks → +RMSNorm →
+distributional critic & reward scaling → +unit weight norm → +entropy
target & noise repetition (= full FlashSAC), plotting episodic return and
the three norm diagnostics per rung.

Wall-clock per env step is expected to be **worse** than SAC's at this
scale (≈11× parameters, doubled critic batches, a 101-way softmax);
`charts/SPS` is reported honestly. The paper's speed claim lives at a
different operating point.

## Open questions from the paper (Pass A judgement calls)

The paper has no pseudocode box; the spec used for Pass A already encodes
the reference implementation's answers to the ambiguities below. Each row
gets confirmed or corrected during the Pass B diff. Rows added during
implementation are marked *(added in Pass A)*.

| # | Paper is silent on | Resolution encoded in Pass A | Confidence |
|---|---|---|---|
| 1 | How `min` over two *distributional* critics is taken | argmin on expected value, then gather that member's whole distribution | **Pass B: confirmed** (`_select_min_q_log_probs`) |
| 2 | Whether the entropy bonus enters the distributional target | yes, as a shift of every atom by `−α·log π` | **Pass B: confirmed** (`_compute_categorical_td_target`, variable `actor_entropy` holds `α·log π`) |
| 3 | Inverted-bottleneck expansion factor | 4 | **Pass B: confirmed** |
| 4 | Exact order inside the residual block | `w1 → BN → ReLU → w2 → BN → ReLU → + residual`, no post-add activation | **Pass B: confirmed** (`FlashSACBlock`) |
| 5 | `log_std` bounds and parameterisation | tanh-squashed into [-10, 2] | **Pass B: confirmed** (`NormalTanhPolicy`) |
| 6 | Optimiser, betas, weight decay | Adam, PyTorch defaults, no weight decay | **Pass B: confirmed** (`fused=True` on CUDA only — numerics-equivalent) |
| 7 | Gradient clipping | none; norm bounding is implicit via weight and feature constraints | **Pass B: confirmed** (no clip anywhere in `update.py`) |
| 8 | Whether the actor also uses BatchNorm | yes, same block stack, narrower | **Pass B: confirmed** |
| 9 | Whether target EMA covers BatchNorm buffers | no, parameters only | **Pass B: confirmed** (`torch._foreach_lerp_` over `parameters()`) |
| 10 | Truncation vs termination in the target | `terminated` only in the target; `terminated or truncated` in the reward accumulator | **Pass B: confirmed** (`batch["terminated"]` in the target; `logical_or` in `_update_reward_stats`) |
| 11 | Running statistics for adaptive reward scaling | discounted-return accumulator + Chan parallel variance; `G_max_seen` running max | **Pass B: confirmed**, incl. the accumulator zeroing the *prior* return on the done step; reference adds a var-init-1 / 1e-4 count-epsilon regulariser we deliberately omit (≤1e-4 transient) |
| 12 | Evaluation protocol | deterministic `tanh(mean)`, exploration noise off | **Pass B: confirmed** (`temperature == 0.0` path leaves noise state untouched) |
| 13 | Observation normalisation | the embedder's leading BatchNorm; nothing else | **Pass B: confirmed** (no obs wrapper in `create_vec_env`) |
| 14 | Seed count and aggregation in the paper's results | bootstrap CIs shown; seed count not stated — we do not cite one | low; `run_mujoco.sh` uses 5 seeds (0,1000,2000,3000,4000) — the repo's recipe, still not a paper claim |
| 15 | n-step semantics under mid-window truncation | reference uses `gamma ** n` unconditionally (approximation); not exercised at n_step = 1 | **Pass B: confirmed** (`update_critic` passes `gamma**n_step`; the buffer stops reward accumulation at done but the discount power is fixed) |
| 16 | Which update count parameterises each optimiser's LR schedule *(added in Pass A)* | Pass A guessed per-optimiser budgets (all ending at 1.5e-4) | **Pass B: WRONG — fixed.** One shared horizon `decay_steps = total env steps × UTD` for all three; each steps per its own update, so actor and temperature end mid-cosine at ~2.25e-4 |

## Verification results (lifecycle steps 5-6)

5 seeds × 1M steps per environment at commit `1f99c7c`, the authors' CPU
recipe, no hyperparameter overrides; compared against **our
CleanRL-verified SAC** episode curves through `roborl benchmark compare`
(there is no CleanRL FlashSAC reference). IQM over the last 10% of
training, 95% stratified bootstrap CIs:

| Environment | FlashSAC IQM [95% CI] | roborl SAC IQM [95% CI] | Harness verdict | Reading |
|---|---|---|---|---|
| HalfCheetah-v4 | 12773.08 [12128.06, 13084.16] | 10367.25 [8127.52, 11703.52] | INVESTIGATE | significant **improvement** — [diagnosis](../lab-notebook/2026-08-30-flashsac-halfcheetah-investigate.md) |
| Hopper-v4 | 2968.38 [2916.37, 3202.74] | 3082.19 [2603.61, 3388.67] | PASS | parity, with a visibly tighter CI |
| Walker2d-v4 | 6123.55 [5954.14, 6531.65] | 4609.62 [4204.40, 5058.98] | INVESTIGATE | significant **improvement** — [diagnosis](../lab-notebook/2026-08-31-flashsac-walker2d-investigate.md) |

The INVESTIGATE verdicts are the parity-based policy firing on
*non-overlap in FlashSAC's favour*; the lab-notebook entries hold the
evidence (raw env returns, clamp fraction exactly 0 on every seed, grad
norms 0.016–0.035 with no clipping, feature norms pinned at ≈ √256,
settled reward scales). Reports:
[benchmarks/reports/flashsac/](../../benchmarks/reports/flashsac/).

Because this is the 1-env / UTD-1 operating point, these gains come from
the paper's **stability** half only, and a three-env aggregate cannot
attribute them to any single one of the six changes — that is the ablation
ladder's job. Wall-clock honesty: ~8 SPS under 8-way CPU parallelism,
far below SAC's — the paper's speed claim lives at the 1024-env GPU
operating point we do not exercise.

## Ablation ladder configuration

The paper's §6.3 architectural ablation, one flag per rung, on Walker2d-v4
(the env with the largest FlashSAC-vs-SAC effect, hence the most
attribution signal), 3 seeds × 1M steps per rung:

| Rung | Adds | Config |
|---|---|---|
| 1 | SAC baseline | the existing verified `sac` runs (CleanRL hyperparameters) |
| 2 | residual blocks with BatchNorm (+ the cross-batch treatment BN requires) | `--exp-name flashsac_abl2 --no-use-rmsnorm --no-use-distributional --no-use-weight-norm --no-use-flash-exploration --alpha-init 1.0` |
| 3 | + terminal RMSNorm | rung 2 minus `--no-use-rmsnorm` (`flashsac_abl3`) |
| 4 | + categorical distributional critic & adaptive reward scaling | rung 3 minus `--no-use-distributional` (`flashsac_abl4`) |
| 5 | + unit weight normalisation | rung 4 minus `--no-use-weight-norm` (`flashsac_abl5`) |
| 6 | + unified entropy target & noise repetition = full FlashSAC | the existing verified `flashsac` runs |

Honest caveats. (a) Rungs 2–6 all use FlashSAC's training recipe (batch
512, cosine LR, τ = 0.01, plain-skip actor period, 10k warmup), so
adjacent-rung deltas within 2–6 are clean; the rung 1→2 delta additionally
includes the recipe change — SAC's own hyperparameters are part of the
baseline. (b) Cross-batch prediction (change #3) is not a rung: train-mode
BatchNorm without it compares quantities under different normalisation
statistics, so it ships with the architecture at rung 2 (as in CrossQ).
(c) Rungs 2–5 use SAC's `-dim(A)` entropy target with `alpha_init = 1.0`
(the FlashSAC values are part of change #5/#6). (d) Rungs 1 and 6 reuse
the verified 5-seed campaigns; new rungs run 3 seeds.

## Pass B diff table

Component-by-component diff against `github.com/Holiday-Robot/FlashSAC`
(cloned 2026-08-28, default branch), performed after Pass A passed its unit
tests, CPU smoke test, and the Pendulum sanity gate. Reference files:
`flash_rl/agents/flashSAC/{agent,network,layer,update}.py`,
`flash_rl/agents/utils/{reward_normalization,network,scheduler,distribution}.py`,
`flash_rl/buffers/torch_buffer.py`, `flash_rl/envs/__init__.py`,
`configs/{flashSAC_base,agent/flashSAC,env/mujoco}.yaml`, `train.py`,
`scripts/run_mujoco.sh`.

| Component | Reference | Ours | Verdict |
|---|---|---|---|
| `UnitLinear` / `UnitBatchNorm` / `UnitRMSNorm` | bias-free orthogonal linear; BN momentum 0.01 eps 1e-5 via `F.batch_norm`; RMSNorm eps 1e-6; joint `[γ;β]` → `√d`, rows → 1, RMS weight → `√d`, all eps 1e-8 | identical | **matched** |
| Ensemble layers | einsum linear, per-member orthogonal; hand-rolled per-member BN (`lerp_` running stats, unbiased running var, biased normalisation); manual RMSNorm | einsum identical; BN as one flattened `F.batch_norm` over `E·d` channels — same statistics, verified against a single-member BN in tests; RMSNorm via `F.rms_norm` + per-member weight | **matched** (different code, same math) |
| Blocks / embedder / trunk order | BN-first embedder; `w1→BN→ReLU→w2→BN→ReLU→+res`; trunk `embed → blocks → RMSNorm → head` | identical | **matched** |
| Actor head | separate `UnitLinear` + free bias for mean/std; log_std tanh-squashed [-10, 2]; `2(log2 − u − softplus(−2u))` correction; log-prob `(B,)`; no action rescaling | identical | **matched** |
| Critic head | ensemble linear + free bias `(E, n)`; log_softmax; Q = expectation over `linspace(v_min, v_max, 101)` | identical | **matched** |
| Parameter init | orthogonal, then **`normalize_parameters()` on actor, critic, and target at construction** | Pass A skipped the init normalisation — wide layers started off-manifold | **deviation, bug — fixed** (loop now normalizes all three at init; regression test added) |
| Distributional target | fresh `a'` BN-eval under `no_grad`; α read inside; cross-batch target critic BN-train; chunk[1]; argmin-then-gather; `r + γⁿ(z − α·logπ)(1−d)` with `d = terminated`; clamp; mass-conserving two-point split | identical (we additionally clamp `l` — float-safety only — and return the pre-clamp support-violation fraction as a diagnostic) | **matched** |
| Critic loss | cross-batch online critic BN-train, chunk[0], CE vs projected target, mean over `(E, B)` | identical | **matched** |
| Actor loss | cross-batch actor BN-train; first half; critic params frozen via `requires_grad_(False)`, BN-eval; `(α_detached·logπ − min Q)`; optional BC term `bc_alpha` (0.0 in the MuJoCo recipe) | identical; BC term not implemented | **matched** (BC omission: deliberate, coefficient is 0 in the target recipe) |
| Temperature loss | `α · (H_detached − H_target)`, `H_target = 0.5·A·log(2πe·σ²)`, α init 0.01 | identical | **matched** |
| **Update order** | **actor + temperature first** (every 2nd update), then critic — whose target uses the *updated* actor and α — then target EMA | Pass A ran critic first | **deviation, bug — fixed** (loop reordered) |
| Target EMA | `torch._foreach_lerp_` over `parameters()` only, τ = 0.01; full `load_state_dict` copy at construction | per-parameter `lerp_`, same semantics | **matched** |
| **LR schedule** | one `warmup_cosine_decay` (optax-style) with shared `warmup = int(1e-6·N)`, `decay_steps = N = total env steps × UTD`; each optimiser's `LambdaLR` steps per its own update → actor/temp end mid-cosine ~2.25e-4 | Pass A gave each optimiser its own budget ending at 1.5e-4 | **deviation, bug — fixed** (shared horizon; ambiguity row 16 adjudicated) |
| Reward normaliser | `G = γ(1−(term∨trunc))G + r` (prior return zeroed on the done step); `G_max_seen` from the *new* G; stats on every collected transition incl. warmup; raw storage, batch-time division by `max(√(var+1e-8), G_max_seen/5)`; `RunningMeanStd` inits var=1 with a 1e-4 count-epsilon | identical semantics; clean Chan update (var init 0, no count-epsilon) | **matched**, one **deviation, deliberate**: no 1e-4 regulariser (≤1e-4 transient on the variance before ~100 samples; our fixtures verify exact statistics) |
| Exploration noise | candidate `randn` + Zeta draw *every* step, applied via `torch.where(reinit, …)` (CUDA-graph style); run length shared, noise per-env; count/reinit semantics; eval path (`T=0`) leaves state untouched | draw only on reinit — same distribution over used noise, different RNG-stream consumption | **matched** (semantics; implementation detail differs) |
| Replay buffer | torch ring buffer, float64→float32 enforced, stores terminated and truncated, n-step machinery (n=1 in this recipe), uniform sampling with replacement, `min_length` 10k | NumPy ring buffer (SAC's), float32, terminated only stored (truncated unused in the n=1 update path), same sampling | **matched** at n_step = 1 (n-step machinery deliberately deferred) |
| Warmup / update cadence | random actions until buffer ≥ 10k, then 1 update per env step; first update on the step the buffer fills | random actions for `learning_starts` steps, updates from `global_step > learning_starts` — one update later | **matched** (boundary differs by one update in ~990k) |
| Env stack | `RescaleAction(env, float32(±1))`, `TimeLimit` (default 1000), seeded obs/action spaces, true `final_obs` stored at episode end (autoreset vector env + patch) | same wrapper and bounds; single non-autoresetting env with explicit `reset()` gives the same data stream | **matched** |
| Evaluation | separate eval env, 50 deterministic episodes every N/10 steps + video | training episodic returns only, per this repo's SAC methodology; deterministic-eval *path* implemented (`eval_action`) but not scheduled | **deviation, deliberate** (verification compares training curves against our SAC baseline collected the same way) |
| Infrastructure | `torch.compile` + optional fp16 AMP (`use_amp=false` in the MuJoCo recipe), fused Adam on CUDA | neither | **deviation, deliberate** (performance only; AMP is off in the target recipe anyway) |
