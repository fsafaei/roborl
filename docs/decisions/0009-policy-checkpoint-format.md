# ADR 0009 — Policy checkpoint format (`roborl.io`)

Status: accepted · Date: 2026-09-08

## Context

Until now a training loop built its actor, trained it, logged metrics, and
discarded the weights. That was fine while roborl's only output was a
verification verdict. It stops being fine when a policy has to leave the
process: evaluated later, handed to a benchmark, or deployed on a robot.
Two sibling repositories consume roborl policies, and the workspace
agreement between them is that **roborl owns one checkpoint format and the
consumers only ever call `load_policy`**.

Requirements that shaped the design:

1. Rebuildable without the training config or code path: the file must
   say which actor it is and how big it is.
2. The loaded policy's action must equal what the loop's own evaluation
   path would have produced, for every actor family: SAC's squashed mean,
   FlashSAC's `tanh(mean)` mapped back through `RescaleAction`, PPO's mean
   after the observation normaliser and `ClipAction`.
3. Safe to load. A checkpoint downloaded from anywhere must not be able to
   execute code on the robot's control machine.
4. Content-addressable, so a real-robot recording can name the exact policy
   that produced it.
5. Small, and Python 3.10 compatible like the rest of roborl.

## Decision

1. **One file, one dict, `weights_only`.** `save_policy` writes a `.pt`
   holding a plain dict of primitives and tensors (`format`,
   `format_version`, `spec`, `metadata`, `actor_state_dict`,
   `obs_normalizer`). `load_policy` reads it with
   `torch.load(weights_only=True)`, so only tensors and primitive containers
   are ever unpickled. Metadata is reduced to JSON primitives before writing.
2. **The spec is in the file.** `PolicySpec` = `algo`, `obs_dim`,
   `act_dim`, `action_low`, `action_high`, `arch` (the constructor arguments
   beyond the two sizes, exactly the keys `ARCH_KEYS[algo]`). `build_actor`
   rebuilds the actor from it, importing the algorithm package lazily so
   `roborl.io` never depends on an algorithm at import time while every loop
   imports `roborl.io`.
3. **One deterministic rule per actor family**, kept in
   `_deterministic_action` and mirrored from the loops; each is checked by a
   unit test against the actor's own evaluation call.
4. **Actor-only scope.** No critics, optimisers, buffers or RNG state. Resume
   is a different artefact with different lifetime and size; if it is ever
   needed it gets its own format version or module.
5. **Frozen observation normaliser for PPO.** Gymnasium's
   `NormalizeObservation` keeps its running statistics in the env wrapper,
   not in the model; env 0's statistics at save time are stored as an
   `ObsNormalizer` together with the `[-10, 10]` clip that follows it. It is
   required for `ppo_continuous` and forbidden for every other algorithm.
6. **Saved once, at the end of the run**, through one shared config field
   `save_policy_path` on `ExperimentConfig` (`--save-policy-path` on every
   subcommand). End-only keeps the loops diffable against CleanRL; periodic
   or best-so-far saving is a later decision. `demo` and discrete `ppo`
   reject the flag.
7. **Fail early.** `save_policy` test-loads the weights into an actor rebuilt
   from the spec and checks SAC-style rescaling buffers against the bounds
   before writing; `load_policy` repeats the checks. Both return the file's
   SHA-256.

## Consequences

- Positive: one small module (`src/roborl/io/policy.py`), no new
  dependencies, no change to how algorithms are written. Consumers can
  verify `spec.obs_dim`/`act_dim`/bounds against their own contracts
  before acting, and record `sha256` for provenance.
- Positive: a mismatch between a spec and an actor is an error at save
  time, in the training process, not a silent shape error on a robot.
- Negative: adding an algorithm means adding a branch in `build_actor`
  and `_deterministic_action`, its `ARCH_KEYS` entry, and a test case. That
  is deliberate: the rule is written where it can be read, not discovered
  through reflection.
- Negative: with `num_envs > 1`, PPO's per-env normaliser statistics differ
  slightly; env 0's are saved. The verification recipe uses one env.
- Negative: discrete PPO has no checkpoint rule yet; the flag fails fast
  there.
