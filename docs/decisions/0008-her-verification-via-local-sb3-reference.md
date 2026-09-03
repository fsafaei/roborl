# ADR 0008 — HER verification via local SB3 reference runs

Status: accepted · Date: 2026-09-03

## Context

ADR 0005 makes every `verified ✅` a mechanical verdict against a reference
run set. For Hindsight Experience Replay there is no CleanRL implementation,
and the only public HER results on the Fetch tasks — rl-baselines3-zoo's
trained agents and their reported scores — cannot serve as the verdict
source for two reasons that compound: they are **TQC** + HER, not SAC + HER
(a different critic, so a mismatch would be uninterpretable), and they ran
on **`Fetch*-v1`** under the dead mujoco-py stack, which neither installs
today nor matches the `-v4` envs current Gymnasium-Robotics ships. The
env-version parity rule of `docs/benchmarking.md` rules them out outright.

Stable-Baselines3's `HerReplayBuffer` is, however, the de-facto reference
implementation of the method, and SB3 supports Gymnasium 1.x and the `-v4`
Fetch envs. The reference can be *produced* rather than *fetched*.

## Decision

1. **Verdict source: SB3's SAC + `HerReplayBuffer`, executed locally.** A
   PEP 723 inline-metadata script,
   `benchmarks/references/sb3-her/run_sb3_her.py`, run with
   `uv run --script`, pins exact versions of `stable-baselines3`,
   `gymnasium-robotics`, `mujoco`, and `pandas` at authoring time and
   trains SB3's `SAC("MultiInputPolicy", …, replay_buffer_class=
   HerReplayBuffer)` with the same hyperparameters, on the same machine,
   the same env id and version, and the same budget as our `her-sac`
   runs — 5 seeds per env. `Monitor` writes per-episode CSVs;
   `to_curves.py` converts them to the harness format (`run_id,
   global_step, episodic_return`), and `roborl benchmark compare` consumes
   them through the already-implemented `manual-csv` path
   (`--reference <files>`). No new adapter code.
2. **Stable-Baselines3 never enters roborl's dependency tree.** It exists
   only in the reference runner's isolated script environment.
   `docs/benchmarking.md`'s adapter table renames the placeholder
   `sb3-zoo (planned)` to `sb3-local (implemented)` pointing at
   `benchmarks/references/sb3-her/`.
3. **Published zoo / Hugging Face numbers are context, never verdicts.**
   Reports may quote them in a clearly labelled context table with the
   TQC / `-v1` caveats attached.
4. **`gymnasium-robotics` becomes the optional extra `fetch`**
   (pattern: the existing `mujoco` extra), constrained with `mujoco<3.12`
   because mujoco 3.12.0 breaks every Fetch reset in Gymnasium-Robotics
   1.4.2 (a joint-type enum comparison regression, reproduced at
   preflight; 3.3.7–3.11.0 all pass). The constraint lives inside the
   `fetch` extra only, so the plain `mujoco` extra keeps floating.
5. **Blind-then-diff applies to the buffer source, not the library.**
   Pass A of `docs/algos/her.md` is implemented without opening
   `her_replay_buffer.py`; running SB3 as a black box for the reference
   runs is allowed in any phase.

## Consequences

- The verification claim is exactly "our HER+SAC matches SB3's HER+SAC,
  same machine, same env version, same hyperparameters" — not "matches the
  zoo's published numbers" and not "reproduces the 2017 paper".
- Both sides share every knob, including our shared deviations from SB3's
  defaults (`learning_starts` 1000), so parity holds by construction and
  the deviations are listed in each report rather than hidden.
- Producing the reference costs a full second campaign of compute
  (≈ 10.5M env steps at batch 2048 × 512³ MLPs); the reference runner,
  converter, and versions are committed so the reference is reproducible,
  and the curves are committed when small enough to be reasonable
  fixtures, else cached with SHA-stamped provenance in the report.
- A future algorithm without a CleanRL reference (e.g. for robosuite) can
  reuse the same pattern: an isolated reference runner producing
  harness-format curves, adjudicated through `compare`.
