# Debugging RL

RL bugs rarely crash; they silently produce a slightly worse policy. The
skills this document teaches — a fixed protocol and a checklist of the
classic bugs — are a core part of this repo's curriculum, not incidental
pain. Foundational reading: Andy Jones, *Debugging Reinforcement Learning*;
John Schulman, *The Nuts and Bolts of Deep RL Research*.

## The protocol

Work the steps in order. Resist the urge to tune hyperparameters first —
that's step 8, not step 1.

1. **Shrink it.** Tiny network, short budget, one fixed seed, smallest env
   that shows the symptom. A bug you can reproduce in 30 seconds is a bug
   you can fix.
2. **Assert shapes, dtypes, and ranges at every boundary.** Observations,
   actions, rewards, dones — at the env boundary, the buffer boundary, and
   the network boundary. Broadcasting bugs (`(N,) − (N,1)`) are the classic
   silent killer.
3. **Overfit test.** The algorithm must crush a trivial task (or memorize a
   handful of transitions) first. If it can't overfit, the update math is
   broken — no amount of tuning helps.
4. **Determinism check.** Two runs with the same seed on CPU must match
   exactly. If they don't, fix seeding before anything else — you cannot
   debug what you cannot reproduce. (`seed_everything` + the env factory
   give you this; see the determinism unit test for the pattern.)
5. **Diff against the reference implementation piece by piece.** Compute
   advantages, returns, ratios with both implementations on the same fixed
   batch; swap components in one at a time until the numbers diverge. This
   is why every algorithm here stays diffable against its CleanRL
   counterpart.
6. **Audit `terminated` vs `truncated`.** Bootstrap through time-limit
   truncation; never bootstrap through true termination. Getting this
   backwards poisons every value target and still "kind of works" — the
   worst failure mode.
7. **Audit action scaling and normalization leakage.** Is the tanh-squashed
   action actually reaching the env in its bounds? Do observation
   normalization statistics differ between train and eval?
8. **Revert to reference hyperparameters exactly.** Only after the math is
   verified do hyperparameters get to be the explanation. Start from the
   reference values, change one thing at a time.
9. **Write it up.** Every real investigation becomes
   `docs/lab-notebook/YYYY-MM-DD-<slug>.md`: symptom → hypothesis →
   experiment → fix. The notebook is curriculum; future-you is the student.

## Common-bugs checklist

Before deep debugging, check the classics:

- [ ] Off-by-one in bootstrapped targets (using `value(s_t)` where
      `value(s_{t+1})` belongs, or misaligned `dones`)
- [ ] Missing `detach()`/`no_grad()` on bootstrap targets — the critic
      chases itself
- [ ] Advantage normalization at the wrong granularity (per-minibatch vs
      per-batch)
- [ ] LR-schedule unit confusion: updates vs environment steps
- [ ] Replay transitions corrupted at autoreset boundaries — Gymnasium 1.x
      autoreset semantics changed: the step after termination returns the
      *reset* observation; storing `(obs_terminal, action, reward, obs_reset)`
      pairs poisons the buffer
- [ ] Missing tanh log-prob correction in squashed-Gaussian policies (SAC)
- [ ] Polyak vs hard target updates mixed up, or wrong τ
- [ ] Unseeded `action_space.sample()` — exploration ignores your seed
      (the env factory seeds it; anything constructed elsewhere must too)
- [ ] float64 observations reaching MPS — cast to float32 at the boundary
- [ ] Reward scale/clipping surprises: a wrapper clipping rewards you didn't
      know about, or returns so large the value loss dwarfs the policy loss

## When a verification run fails

An INVESTIGATE verdict from `roborl benchmark compare` triggers this
protocol. The outcome is either a fix (and a re-run) or a documented,
understood deviation — both end with a lab-notebook entry. See
[benchmarking.md](benchmarking.md) for the verdict policy.
