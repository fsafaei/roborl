# SAC — Soft Actor-Critic

Spec note per [algorithm-lifecycle.md](../algorithm-lifecycle.md) step 0.
Written before the implementation; the "implementation details" section is
the diffing checklist for verification debugging.

**Papers:** Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy
Deep RL with a Stochastic Actor* (ICML 2018) — the original, with a state
value network and fixed temperature. Haarnoja et al., *Soft Actor-Critic
Algorithms and Applications* (2019) — the version everyone implements: no
value network, automatic temperature tuning.
**Reference implementation:** CleanRL `sac_continuous_action.py`
([docs](https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy)),
which follows the 2019 paper.

## Objective

Maximum-entropy RL: maximize expected return *plus* policy entropy at every
visited state,

$$J(\pi) = \sum_t \mathbb{E}_{(s_t,a_t)\sim\rho_\pi}\big[r(s_t,a_t) + \alpha\,\mathcal{H}(\pi(\cdot\,|\,s_t))\big].$$

The temperature α trades reward against entropy. Entropy in the objective
buys (a) exploration that anneals itself as the Q-landscape sharpens and
(b) robustness — the policy keeps probability mass on all near-optimal
actions instead of collapsing early. Off-policy: transitions come from a
replay buffer; the actor is trained on buffer states, not on-policy rollouts.

## Update equations (2019 version, as CleanRL implements it)

Networks: twin soft Q-functions $Q_{\phi_1}, Q_{\phi_2}$, their polyak
targets $Q_{\bar\phi_1}, Q_{\bar\phi_2}$, squashed-Gaussian actor
$\pi_\theta$, and a learned temperature α. No state-value network.

**Critic.** Sample a batch $(s, a, r, s', d)$; sample a *fresh* next action
$a' \sim \pi_\theta(\cdot|s')$ (not from the buffer). The TD target is

$$y = r + \gamma\,(1-d)\,\Big(\min_{i=1,2} Q_{\bar\phi_i}(s', a') - \alpha \log \pi_\theta(a'|s')\Big),$$

and each critic minimizes $\mathrm{MSE}(Q_{\phi_i}(s,a),\, y)$; the two
losses are summed into one backward pass over one shared optimizer.

**Actor.** Reparameterized (rsample) actions $a_\theta \sim \pi_\theta(\cdot|s)$:

$$L(\theta) = \mathbb{E}_s\big[\alpha \log \pi_\theta(a_\theta|s) - \min_{i} Q_{\phi_i}(s, a_\theta)\big].$$

Gradients flow through the action into the actor; the critics are *not*
detached but their optimizer never sees actor-loss gradients (separate
optimizers — the standard trick).

**Temperature.** With target entropy $\bar{\mathcal{H}} = -\dim(\mathcal{A})$:

$$L(\alpha) = \mathbb{E}_s\big[-\alpha\,(\log \pi_\theta(a|s) + \bar{\mathcal{H}})\big],$$

log π detached (CleanRL recomputes it under `no_grad`). Optimized in
log-space: the parameter is `log_alpha`, α = exp(log_alpha). Note CleanRL
multiplies by α itself (`log_alpha.exp()`), not by log α as some
implementations do — same fixed point, different gradient scale.

**Targets.** Polyak averaging every step:
$\bar\phi \leftarrow \tau\phi + (1-\tau)\bar\phi$, τ = 0.005.

## The implementation details that matter

1. **Squashed Gaussian + log-prob correction.** The actor outputs mean and
   log_std of a diagonal Gaussian; actions are `tanh`-squashed then affinely
   rescaled to the action bounds (`action_scale`, `action_bias` computed from
   the action space and stored as buffers). The change of variables adds a
   correction to the log-density, applied per dimension then summed:
   $\log\pi(a|s) = \log\mathcal{N}(u) - \sum_j \log\big(\text{scale}_j\,(1 - \tanh(u_j)^2) + 10^{-6}\big)$
   where $u$ is the pre-squash sample. The `1e-6` floor prevents
   `log(0)` at saturation. Forgetting the *scale* factor inside the log (not
   just `1 - tanh²`) is a classic bug that only bites on envs whose action
   range isn't [-1, 1].
2. **log_std parameterization.** Not clamped: `tanh`-squashed into
   [LOG_STD_MIN, LOG_STD_MAX] = [-5, 2] (SpinUp / Denis Yarats style):
   `log_std = min + 0.5·(max − min)·(tanh(raw) + 1)`. Keeps gradients alive
   at the bounds where a hard clamp would zero them.
3. **Twin critics, min in *both* places.** The `min` over the two target
   critics appears in the TD target *and* the `min` over the two online
   critics in the actor loss. One shared Adam over both critics' parameters.
4. **Delayed, compensated actor updates.** Critic updates every env step
   (after `learning_starts`); actor and α update only every
   `policy_frequency` = 2 steps, but then run *2 consecutive updates on the
   same batch* to compensate. Target networks update every step
   (`target_network_frequency` = 1).
5. **Entropy auto-tuning target.** $\bar{\mathcal{H}} = -\dim(\mathcal{A})$
   (e.g. −1 on Pendulum, −6 on HalfCheetah). `log_alpha` initialized to 0
   (α = 1), optimized with the *critic's* learning rate, not the actor's.
6. **Warmup.** Uniform-random actions (from the seeded action space) for the
   first `learning_starts` = 5000 steps; no gradient updates until then.
   The random-action phase must still write transitions to the buffer.
7. **Bootstrap through truncation, never through termination.** The stored
   done flag is `terminated` only. At a time-limit truncation the target
   still bootstraps ($1-d = 1$), which requires storing the *true* final
   observation as `next_obs` (CleanRL: `handle_timeout_termination=False`
   plus `final_observation` patching).
8. **Fresh next actions, no target actor.** Unlike TD3/DDPG there is no
   target policy network and no target-action smoothing noise; $a'$ comes
   from the *current* actor under `no_grad`.
9. **Evaluation action = squashed mean.** `tanh(mean)·scale + bias`, no
   sampling — this is what `eval/`-namespace metrics will use later.
10. **Rewards/dones flattened to 1-D** before the target computation;
    a stray `[batch, 1]` broadcast against `[batch]` silently produces a
    `[batch, batch]` target (the classic shape bug — assert shapes).
11. **Architecture.** All MLPs 256-256 ReLU. Critics take `cat([obs, act])`.
    No observation normalization, no reward scaling, no gradient clipping.
12. **Observations float32.** MuJoCo emits float64 in places and MPS has no
    float64 — cast at the buffer boundary.

## Hyperparameters (CleanRL defaults — verification uses these exactly)

| Hyperparameter | Value |
|---|---|
| total_timesteps | 1_000_000 |
| buffer_size | 1_000_000 |
| gamma | 0.99 |
| tau (polyak) | 0.005 |
| batch_size | 256 |
| learning_starts | 5_000 |
| policy_lr | 3e-4 |
| q_lr (also α's lr) | 1e-3 |
| policy_frequency | 2 |
| target_network_frequency | 1 |
| autotune / initial α | True / 1.0 (fixed-α fallback: 0.2) |

## Deviations from CleanRL (deliberate, semantics-preserving)

- **Single env with manual `reset()`** instead of
  `SyncVectorEnv(num_envs=1)`. CleanRL's script is written for pre-1.0
  autoreset semantics (`final_info` / `final_observation`); under our
  Gymnasium 1.x a vector env autoresets on the *next* step, which would burn
  a `global_step` per episode without storing a transition. A single env
  with an explicit reset reproduces the reference's data stream exactly:
  one stored transition per `global_step`, true final observation at episode
  end, and the following step starting from the reset observation.
- **Local replay buffer** (~40 lines, preallocated NumPy ring) instead of
  the stable-baselines3 `ReplayBuffer` — SB3 is not a dependency of this
  repo, and the buffer *is* curriculum (autoreset gotcha lives there).
- **Shared roborl infrastructure** for config / seeding / device / env
  factory / W&B logging, per ADR 0003; metric names via
  `roborl.telemetry.metrics` constants (CleanRL's exact `losses/...` names).
- **Two pure helpers factored out for unit-testing**: the squashed-Gaussian
  log-prob correction and the soft TD target. The loop stays top-to-bottom;
  the math gets hand-computed fixtures per lifecycle step 2. One numerical
  nit: the helper recomputes ``tanh(x_t)`` where CleanRL reuses a single
  ``y_t`` node, so gradient accumulation order differs by ~1 ulp in float32
  (verified equivalent to machine epsilon in float64) — same math, not
  bitwise-identical training trajectories.
- **Seeded warmup actions.** The random actions before ``learning_starts``
  come from the seeded ``env.action_space``; CleanRL samples its *unseeded*
  ``single_action_space`` there (a known reproducibility quirk of the
  reference script). Strictly more reproducible, no behavioral difference
  in expectation.

## Telemetry (mirrors CleanRL's SAC exactly, logged every 100 steps)

`losses/qf1_values`, `losses/qf2_values`, `losses/qf1_loss`,
`losses/qf2_loss`, `losses/qf_loss` (logged as the *mean* of the two, i.e.
summed loss / 2), `losses/actor_loss`, `losses/alpha`, `losses/alpha_loss`
(autotune only), plus the standard `charts/episodic_return`,
`charts/episodic_length`, `charts/SPS` at episode ends.

Reading them: `qf*_values` drifting to ±10³ on Pendulum-scale rewards means
target divergence; `alpha` should decay from 1.0 as the policy sharpens
(flat α ≈ 1 forever means the entropy target is never binding — check
`log_pi`); `actor_loss` is meaningless in magnitude but should trend down as
Q-values grow.

## Verification plan (lifecycle steps 5–6)

Env-version parity: CleanRL's public benchmark runs use `Hopper-v4`,
`HalfCheetah-v4`, `Walker2d-v4` (confirm at fetch time via
`roborl benchmark fetch --algo sac_continuous_action --env-id ...`).
≥5 seeds on Pendulum-class budgets; MuJoCo budget (1M steps × 3 envs) may
justify 3 seeds — decide and state in the report. Sanity gate before any
verification spend: solves Pendulum-v1 (random ≈ −1200, solved ≳ −200).
