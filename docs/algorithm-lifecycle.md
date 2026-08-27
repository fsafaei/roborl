# Algorithm lifecycle

The definition of done for every algorithm in this repo. An algorithm that
skips steps isn't done — it's a draft. The README status table
(`planned → in progress → implemented → verified ✅`) advances only through
these steps, and `verified ✅` requires the committed report of step 6.

## Steps

0. **Read, then specify.** Read the paper and the CleanRL implementation.
   Write a short spec note in `docs/algos/<algo>.md`: the objective, the
   update equations, and — most importantly — the implementation details
   that actually matter, in the spirit of *The 37 Implementation Details of
   PPO* (Huang et al., ICLR Blog Track 2022). Writing the spec first makes
   step 5's diffing tractable.
1. **Implement** in `src/roborl/algos/<algo>/` as **one readable
   top-to-bottom training loop**. Use the shared infrastructure (config,
   seeding, device, env factory, telemetry); keep algorithm math local to
   the file, even when that means duplicating a small helper another
   algorithm already has (rule of three — see ADR 0003). The demo
   (`src/roborl/demo.py`) is the structural template.
2. **Unit-test the math** against hand-computed fixtures: advantage
   estimation, return/target computation, loss values on a tiny fixed batch.
3. **Add the CI smoke test:** a few hundred steps on CPU, asserting the loop
   runs, logs, and doesn't NaN. Marker `smoke`.
4. **Sanity check:** solves a trivial env (CartPole for discrete, Pendulum
   for continuous) beyond random-agent level.
5. **Verification runs** per [benchmarking.md](benchmarking.md): ≥5 seeds
   (≥3 with a stated justification for expensive envs), the reference's
   exact hyperparameters (any deviation listed in the report), the same
   total-timestep budget, on the same env id and version the reference used.
   Tracked to W&B.
6. **Compare and report:** `roborl benchmark compare ... --out
   benchmarks/reports/<algo>/<env>/`; commit the report and figure.
7. **On INVESTIGATE:** run the [debugging protocol](debugging-rl.md); write
   the lab-notebook entry; fix and re-run, or document the understood
   deviation.
8. **Finalize docs:** complete `docs/algos/<algo>.md`, extend the telemetry
   table with algorithm-specific metrics, flip the README status row (with
   the report linked).
9. **PR** with this checklist filled in, CI green.
