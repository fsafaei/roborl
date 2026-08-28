# PPO — Proximal Policy Optimization (discrete actions)

Spec note per [algorithm-lifecycle.md](../algorithm-lifecycle.md) step 0.
Written before the implementation; the "implementation details" section is
the diffing checklist for verification debugging.

**Papers:** Schulman et al., *Proximal Policy Optimization Algorithms*
(2017) — the clipped surrogate objective. Schulman et al., *High-Dimensional
Continuous Control Using Generalized Advantage Estimation* (ICLR 2016) —
the advantage estimator every PPO uses. The real spec, though, is
Huang et al., *The 37 Implementation Details of Proximal Policy
Optimization* (ICLR Blog Track 2022): PPO's reported performance lives in
the implementation details, not the paper equations.
**Reference implementation:** CleanRL `ppo.py`
([docs](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy)) — the discrete,
classic-control variant these 13 core details describe.

## Objective

On-policy policy gradient with a trust region enforced by clipping. Collect
a fixed-size batch of rollout data under the current policy π_old, then take
several minibatch gradient epochs on the clipped surrogate

$$L^{CLIP}(\theta) = \mathbb{E}_t\Big[\min\big(r_t(\theta)\,\hat A_t,\ \mathrm{clip}(r_t(\theta),\,1-\epsilon,\,1+\epsilon)\,\hat A_t\big)\Big],\qquad r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\mathrm{old}}(a_t|s_t)},$$

which removes the incentive to move the ratio outside [1−ε, 1+ε] in the
direction the advantage points, letting the same batch be safely reused for
multiple epochs. The full loss adds a clipped value-function term and an
entropy bonus:

$$L = L^{CLIP}_{\mathrm{pg}} + c_v\,L^{V} - c_e\,\mathcal{H}[\pi_\theta],$$

minimized as `pg_loss - ent_coef * entropy + vf_coef * v_loss` over one
optimizer for both networks.

**Advantages** come from GAE(γ, λ), computed backward over the rollout:

$$\delta_t = r_t + \gamma\,V(s_{t+1})\,(1-d_{t+1}) - V(s_t),\qquad \hat A_t = \delta_t + \gamma\lambda\,(1-d_{t+1})\,\hat A_{t+1},$$

with $V(s_{T})$ bootstrapped from the critic on the observation after the
last rollout step. Value targets are $\hat R_t = \hat A_t + V(s_t)$ — the
"returns" PPO regresses the critic to are advantages plus the *old* values,
not Monte-Carlo returns.

## The implementation details that matter

The 13 core details from Huang et al., as CleanRL's `ppo.py` implements
them, in loop order:

1. **Vectorized architecture.** `num_envs` = 4 parallel envs stepped in
   lockstep; each rollout is `num_steps` = 128 steps × 4 envs = one batch of
   512 transitions. `global_step` advances by `num_envs` per vector step.
   Storage is `(num_steps, num_envs)`-shaped tensors, flattened to
   `(batch,)` only after GAE.
2. **Orthogonal initialization, zero biases.** Every hidden layer
   orthogonal with gain √2; the policy head gain 0.01 (near-uniform initial
   action distribution — this one measurably matters); the value head gain
   1.0. All biases 0.
3. **Adam with eps = 1e-5**, not the default 1e-8. One optimizer over all
   parameters of both networks.
4. **Learning-rate annealing.** Linear decay from 2.5e-4 to 0 over
   `num_iterations = total_timesteps // batch_size` iterations, set at the
   top of each iteration via `frac = 1 - (iteration-1)/num_iterations`.
5. **GAE as above**, with the done flags *offset by one*: `dones[t]` marks
   whether `obs[t]` began a new episode, so step t masks with
   `1 - dones[t+1]` (the flag of the *next* row), and the last row uses the
   `next_done` carried past the rollout's end.
6. **Minibatch updates over the shuffled full batch.** `update_epochs` = 4
   passes; each epoch reshuffles all 512 indices (`np.random.shuffle` — the
   global NumPy RNG, which `seed_everything` seeds) and slices 4 minibatches
   of 128. Every transition is used every epoch; nothing is dropped.
7. **Advantage normalization per minibatch** — `(A - mean) / (std + 1e-8)`
   inside the minibatch loop, after slicing, not over the whole batch.
8. **Clipped surrogate objective**, implemented sign-flipped as
   `max(-A·r, -A·clip(r, 1-ε, 1+ε))` and meaned; ε = 0.2.
9. **Value-loss clipping** (on by default, kept for reference parity even
   though Huang et al. find it doesn't help): the clipped branch is
   `v_old + clip(v_new - v_old, ±ε)`, the loss is
   `0.5 · max((v_new - R)², (v_clipped - R)²)` meaned. The unclipped
   fallback is `0.5 · MSE`.
10. **Entropy bonus** `ent_coef` = 0.01, from the Categorical's analytic
    entropy, meaned over the minibatch.
11. **Global gradient clipping** to L2-norm 0.5 over *all* parameters,
    every minibatch step.
12. **Debug variables.** `approx_kl = mean((r-1) - log r)` (the low-variance
    Schulman estimator), `old_approx_kl = mean(-log r)`, `clipfrac` = the
    fraction of ratios with `|r-1| > ε` averaged over all minibatches of the
    iteration, and `explained_variance = 1 - Var(R - V)/Var(R)` computed
    once per iteration over the whole batch (NaN when `Var(R) = 0`).
13. **Separate policy and value MLPs** (no shared trunk on classic
    control): each 64-64 with **tanh** activations — not ReLU, not 256-256;
    a shared trunk is an Atari-variant detail.

And the loop-mechanics details that live between those:

14. **π_old is not a second network.** Log-probs and values are recorded at
    rollout time (under `no_grad`); the ratio compares recomputed log-probs
    against those stored ones. First epoch, first minibatch: ratio ≡ 1,
    approx_kl ≡ 0 — a cheap invariant worth asserting in tests.
15. **Actions are Categorical samples from logits**; the stored action (not
    a fresh sample) is what gets its log-prob recomputed during updates
    (`get_action_and_value(obs, action)`).
16. **`target_kl` early stopping is off by default** (`None`). When set, it
    breaks out of the epoch loop after any epoch whose last minibatch
    `approx_kl` exceeds it.
17. **Truncation is treated as termination.** `next_done = terminated OR
    truncated`, no bootstrap through time limits — a known bias of the
    reference script (deliberately *not* fixed here: verification means
    matching the reference; see deviations). On CartPole the time limit is
    a return ceiling, so the bias is mild.
18. **No observation/reward normalization, no reward clipping** in `ppo.py`
    (those are the continuous-control variant's details).

## Hyperparameters (CleanRL defaults — verification uses these exactly)

| Hyperparameter | Value |
|---|---|
| total_timesteps | 500_000 |
| learning_rate | 2.5e-4 (Adam, eps 1e-5) |
| num_envs | 4 |
| num_steps | 128 (→ batch_size 512) |
| anneal_lr | True |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| num_minibatches | 4 (→ minibatch_size 128) |
| update_epochs | 4 |
| norm_adv | True |
| clip_coef | 0.2 |
| clip_vloss | True |
| ent_coef | 0.01 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |
| target_kl | None |

## Deviations from CleanRL (deliberate, semantics-preserving)

- **`SyncVectorEnv` with `autoreset_mode=SAME_STEP`.** CleanRL's script is
  written for pre-1.0 autoreset semantics: the step that ends an episode
  returns the reset observation, done = 1. Gymnasium 1.x defaults to
  NEXT_STEP autoreset, which would insert a phantom zero-reward transition
  per episode into the batch. `AutoresetMode.SAME_STEP` (Gymnasium ≥ 1.1)
  reproduces the reference data stream exactly. The final observation
  (surfaced in `info["final_obs"]`) is deliberately unused, matching the
  reference: with truncation folded into `done` (detail 17), nothing
  bootstraps from it.
- **Episode stats read from the 1.x dict-of-arrays info format**
  (`infos["final_info"]["episode"]` masked by `_episode`) instead of
  CleanRL's 0.29-era list-of-dicts scan. Same numbers, same
  `global_step` x-axis.
- **Shared roborl infrastructure** for config / seeding / device / env
  factory / W&B logging, per ADR 0003; metric names via
  `roborl.telemetry.metrics` constants.
- **Pure helpers factored out for unit-testing**: GAE
  (`compute_gae`), the clipped policy objective with its debug stats
  (`clipped_policy_loss`), the clipped value loss (`clipped_value_loss`),
  and `explained_variance`. Identical math, hand-computed fixtures per
  lifecycle step 2; the loop stays top-to-bottom.
- **Env seeding via the factory** (`reset(seed=seed+idx)` inside the thunk,
  plain `reset()` in the loop) rather than CleanRL's `reset(seed=seed)` on
  the vector env — same per-env seed sequence, and the action/observation
  spaces are seeded too.

## Telemetry (mirrors CleanRL's `ppo.py` exactly, logged once per iteration)

`charts/learning_rate`, `losses/value_loss`, `losses/policy_loss`,
`losses/entropy`, `losses/old_approx_kl`, `losses/approx_kl`,
`losses/clipfrac` (mean over the iteration's minibatches),
`losses/explained_variance`, `charts/SPS`, plus `charts/episodic_return` /
`charts/episodic_length` at every episode end. All against `global_step`.

Reading them: `clipfrac` ≈ 0 means updates too timid to matter, ≈ 1 means
wildly off-policy updates; `approx_kl` spiking means the clip isn't
containing the step (LR or advantage scale); `explained_variance` ≤ 0 means
the critic is useless — check GAE masking (detail 5) and the value target
definition before anything else. `losses/policy_loss` magnitude is
meaningless in PPO; watch trends. Details in
[docs/telemetry.md](../telemetry.md).

## Verification results (lifecycle steps 5–6) — PASS on all three referenced envs

Ran 2026-08-27 at commit `4ed2bee` (MountainCar-v0: 2026-08-28 at
`f9be429`), 5 seeds per env (seeds 1–5), 500k steps each, CleanRL's default
hyperparameters with zero overrides. Reference: 6 CleanRL seeds per env
from the `openrlbenchmark/cleanrl` W&B project.
Final performance is IQM over the last 10% of training with 95% stratified
bootstrap CIs; verdicts from `roborl benchmark compare`, reports committed
under `benchmarks/reports/ppo/`.

| Env | roborl IQM [95% CI] | CleanRL IQM [95% CI] | Verdict |
|---|---|---|---|
| CartPole-v1 | 488.9 [472.2, 498.5] | 495.4 [488.3, 498.8] | **PASS** |
| Acrobot-v1 | −84.0 [−84.9, −82.8] | −84.6 [−86.3, −83.7] | **PASS** |
| MountainCar-v0 | −200.0 [−200.0, −200.0] | −200.0 [−200.0, −200.0] | **PASS** |
| LunarLander-v3 | 26.7 [8.6, 40.7] | — none exist | **N/A** |

LunarLander (issue #3's third env) has **no CleanRL `ppo` reference at
all** — openrlbenchmark holds `ppo` runs only for CartPole-v1, Acrobot-v1,
and MountainCar-v0, with neither LunarLander-v2 nor -v3 present. Our five
tracked runs and their harness-computed stats are recorded in a hand-written
no-verdict report (`benchmarks/reports/ppo/LunarLander-v3/report.md`)
rather than given a verdict nothing supports. MountainCar-v0 (the
reference's actual third env) stands in as the third verified comparison —
a floor-match PASS: CleanRL's `ppo.py` never solves MountainCar at these
hyperparameters (sparse reward, no exploration mechanism), so both sides
sit exactly at the −200 time-limit floor. That certifies "we reproduce the
reference's behavior," which is precisely what verification claims — not
that PPO solves the env.

Sanity gate (step 4) passed: CartPole-v1 last-10 mean 295.9 at 60k steps
(random ≈ 22; the 500 cap gets touched soon after, with the oscillation
PPO is known for), locked in as a `slow`-marker test.
