# Telemetry

Telemetry is this repo's primary debugging instrument and its scientific
record. Every training run logs the same canonical metrics with the same
names, so any two runs — ours or CleanRL's — can be overlaid in one W&B
workspace.

## Conventions

- **Metric names mirror CleanRL exactly** for everything CleanRL logs
  (`charts/...`, `losses/...`). Our additions live in distinct namespaces so
  provenance stays obvious: `diagnostics/` (training internals) and `eval/`
  (deterministic-policy evaluation).
- **The x-axis is `global_step`** — environment steps — for every metric.
  Wall-clock comparisons use `charts/SPS` instead.
- Metric names are constants in `src/roborl/telemetry/metrics.py`. Algorithm
  code imports them; hand-typing a metric string in a training loop is a bug.
- Every run records the full config dataclass plus provenance: git SHA and
  dirty flag, Python/torch/gymnasium versions, resolved device, hostname.
  A result that can't be traced to a commit and config doesn't exist.
- Run identity: project `roborl`, group `{exp_name}-{env_id}` (so seeds of
  one experiment aggregate natively), name
  `{exp_name}-{env_id}-s{seed}-{timestamp}`.
- Offline-first: `WANDB_MODE=offline` for air-gapped runs, `track=False` for
  no telemetry at all. Tests and CI never talk to W&B.

## Reading the metrics

For each metric: what it measures, what healthy looks like, and what
specific failure modes look like. Algorithm-specific metrics get added to
this table as algorithms land.

| Metric | Measures | Healthy | Broken looks like |
|---|---|---|---|
| `charts/episodic_return` | Undiscounted return of each finished training episode | Noisy but trending up; variance shrinks as policy stabilizes | **Flat at floor**: no learning signal reaching the optimizer — wrong reward scale, dead gradients, broken replay, or the task is too hard as posed. **Collapse after progress**: instability — learning rate too high, value divergence, target-network staleness, or entropy collapse. These are different bugs; identify which shape you have before touching anything |
| `charts/episodic_length` | Steps per episode | Task-dependent: grows on survival tasks (CartPole), shrinks on reach-goal-fast tasks | Pinned at the env's time limit → agent never terminates naturally; check `terminated` vs `truncated` handling |
| `charts/SPS` | Environment steps per second, end to end | Roughly constant after warm-up | Sagging over time → logging/render/replay bottleneck, memory growth; sudden drops → video capture or eval episodes on the hot path |
| `charts/learning_rate` | Current optimizer LR | Follows the configured schedule exactly | Anything else → schedule unit confusion (updates vs env steps) |
| `losses/value_loss` | Value-function regression loss | Spikes when the data distribution shifts, then settles | Monotonic growth → diverging critic (LR, reward scale); exact zero → target leakage or dead network |
| `losses/policy_loss` | Policy-gradient surrogate | **Magnitude alone is meaningless in PPO** — watch trends and spikes, not the value | Persistent large spikes → advantage scale or ratio explosions |
| `losses/entropy` | Policy entropy | Decreases gradually as the policy commits | **Instant collapse** → premature determinism: entropy bonus too low, LR too high; exploration dies. **Never decreasing** → no learning signal: policy gradient isn't reaching the actor |
| `losses/approx_kl` | KL between old and new policy per update | Small and roughly stable | Spikes → steps too aggressive: LR too high or clipping not doing its job |
| `losses/old_approx_kl` | The naive KL estimator `mean(-log r)` (CleanRL logs both) | Tracks `approx_kl`; can go slightly negative (it's a higher-variance estimate) | Large divergence between the two estimators → ratio distribution has heavy tails: advantage scale or LR |
| `losses/clipfrac` | Fraction of PPO ratios clipped | Mid-range (a few % to ~30%) | ≈0 → updates too timid to matter; ≈1 → wildly off-policy updates, something upstream (advantages, LR) is wrong |
| `losses/explained_variance` | How much of return variance the value function explains | Climbing toward ~1 | ≤0 → the value function is useless: check bootstrapping (`terminated` vs `truncated`), return scale, and whether values and returns are even aligned |
| `losses/qf1_values`, `losses/qf2_values` | Mean Q-estimate of each SAC critic on replayed actions | Tracks the true return scale of the env (e.g. settles around the discounted-return magnitude); the two critics stay close | Runaway growth far beyond any achievable return → target divergence (check `terminated` vs `truncated` in the buffer, target-network updates); large persistent qf1↔qf2 gap → one critic broken |
| `losses/qf1_loss`, `losses/qf2_loss`, `losses/qf_loss` | Per-critic TD MSE and their mean (CleanRL convention: `qf_loss` = sum/2) | Noisy, roughly flat band after warm-up; scales with reward magnitude | Monotonic growth → diverging targets; exact zero → the critic is fitting its own output (target leakage) |
| `losses/actor_loss` | SAC policy objective `α·logπ − min Q` | Magnitude meaningless; should trend *down* as Q-values grow | Flat forever alongside flat returns → policy gradient not flowing (check reparameterized sampling, `detach` mistakes) |
| `losses/alpha` | SAC entropy temperature (auto-tuned) | Decays from 1.0 as the policy sharpens toward the entropy target | Pinned at 1.0 → entropy target never binding (check `log_pi` scale, tanh correction); collapse to ~0 early → premature determinism, exploration dies |
| `losses/alpha_loss` | Temperature objective `−α(logπ + H̄)` | Hovers near zero once entropy tracks the target | Large sustained magnitude → entropy far from target: policy saturated (tanh bounds) or target entropy wrong for the action dim |
| `diagnostics/grad_norm` | Global gradient norm before clipping (FlashSAC: critic grad norm, and there is deliberately **no** clipping) | Stable band | Spikes or NaN → instability: lower LR, check reward scale, verify clipping is applied. In FlashSAC, growth here means the weight/feature norm constraints are not holding — check `normalize_parameters()` runs after every optimiser step |
| `diagnostics/target_clamp_fraction` | FlashSAC: fraction of distributional-target atoms at/beyond the fixed [-5, 5] support *before* clamping — the single best FlashSAC health signal | ≈ 0 after warm-up (verified runs: exactly 0) | Above a few percent → adaptive reward scaling is failing to keep discounted returns inside the support; the critic saturates at a boundary atom while looking superficially healthy, and "it just does not learn" |
| `diagnostics/reward_scale` | FlashSAC: the reward normaliser's denominator `max(√var(G), G_max_seen/5)` | Settles to a task-dependent constant (e.g. ~200 on Walker2d, ~390 on HalfCheetah) | Continuous drift → return statistics not converging; sudden jumps → the G_max branch taking over after a return spike |
| `diagnostics/return_rms_var` | FlashSAC: running variance of the discounted-return stream feeding the reward scale | Grows during learning, then flattens | Frozen at ~0 → accumulator being reset every step (check the done flags feeding the normaliser) |
| `diagnostics/target_dist_entropy` | FlashSAC: entropy of the projected categorical target distribution | Moderate, slowly declining as the critic sharpens | Collapse to ~0 (a spike distribution) → over-confident critic; often co-occurs with clamping |
| `diagnostics/critic_feature_norm` | FlashSAC: L2 norm of the critic trunk's RMSNorm output | Pinned near √hidden (≈16 at width 256) — the paper's "feature norm bounded" claim, made visible | Departing from √d → RMSNorm weight escaped its `normalize_parameters()` projection |
| `diagnostics/noise_repeat_len` | FlashSAC: currently sampled exploration-noise run length | Averages near the truncated-Zeta mean (~2.1) | Stuck at 1 or at the max → Zeta sampler or repeat-state bug |
| `diagnostics/target_entropy` | FlashSAC: the entropy target `0.5·A·log(2πe·σ²)` (constant) | Exactly `-0.4782·A` for σ = 0.15 | Any other value → wrong action-dim or σ wiring |
| `diagnostics/param_norm` | Global parameter norm | Slow drift | Runaway growth → no regularization pressure and diverging updates |
| `eval/episodic_return` | Deterministic-policy return, evaluated every N steps | Tracks or exceeds training return | Large persistent train↔eval gap → exploration noise doing the work, or normalization statistics leaking between train and eval |
| `eval/episodic_length` | Deterministic-policy episode length | Task-dependent; on Fetch pinned at 50 (episodes only truncate) | Anything but 50 on Fetch → the eval env is not the Fetch time limit you think it is |
| `eval/success_rate` | HER: deterministic-policy final-step `is_success`, the headline number on goal-conditioned tasks | Reach → 1.0 within a few thousand steps; Push / PickAndPlace climb toward 0.9+ by 1M | Far below `diagnostics/success_rate` → eval plumbing (wrong env, wrong goal in the input) or an exploration-vs-mean gap |
| `diagnostics/success_rate` | HER: final-step `is_success` of each training episode — the task's real progress signal (return and success are near-affine on sparse Fetch) | Push / P&P: near 0 for the first ~50–150k steps, then climbing; Reach → 1 fast | **Flat at 0 at 300k on Push with HER on** → relabeling or reward recomputation is broken; check this before anything else |
| `diagnostics/her_virtual_fraction` | HER: realized share of relabeled rows in the minibatch | Exactly `k/(k+1)` (0.8 at k = 4), constant; 0 with HER off | Anything else → split arithmetic |
| `diagnostics/her_virtual_reward_zero_fraction` | HER: share of relabeled rewards equal to 0 (goal hit under the substitute goal) | ≈ 0.1–0.3 on Fetch; the analytic floor from own-successor hits alone is `H_T/T` ≈ 0.09 at T = 50 | ≈ 0 → the reward is recomputed on the wrong achieved goal (pre-step instead of post-step) or the future-index math excludes the own successor; ≈ 1 → degenerate goals (achieved ≡ desired) |
| `diagnostics/q_lower_bound_violation` | HER: fraction of the minibatch's `min(Q1, Q2)` below `−1/(1−γ)·1.05` (−21 at γ = 0.95) — the sparse-reward value floor. A diagnostic only; nothing is clipped | ≈ 0 after warm-up | Growing → value divergence: check done wiring (a relabeled success must not terminate), γ plumbing, the substituted goal being on both sides of the transition |

## How to read a W&B workspace

1. **Compare seeds within a group first.** Seeds of one experiment share a
   W&B group. If seeds disagree wildly, you don't have a result yet — you
   have variance. Fix determinism and seed count before comparing anything.
2. **Then compare groups against reference curves.** Fetch CleanRL's runs
   (`roborl benchmark fetch`) or add the `openrlbenchmark` project to your
   workspace; identical metric names mean the curves overlay directly.
3. Watch `charts/SPS` whenever you change infrastructure — performance
   regressions hide in telemetry code, wrappers, and logging frequency.
4. Videos (`--capture-video`) are logged periodically; on manipulation tasks
   they are often the fastest way to see *what* the policy actually does.
