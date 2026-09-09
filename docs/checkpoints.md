# Policy checkpoints

`roborl.io` owns the one checkpoint format in this ecosystem (ADR 0009). A
checkpoint is a **policy artefact**: the trained actor's weights, the spec
needed to rebuild it, a frozen observation normaliser where the algorithm
trained behind one, and provenance. It is what gets evaluated or deployed.
It is *not* a resume-training snapshot: no critics, optimisers, replay
buffers or RNG state.

## Saving

Every training subcommand takes `--save-policy-path`. The policy is written
once, when the run ends:

```bash
uv run roborl sac --env-id Pendulum-v1 --total-timesteps 20000 \
    --save-policy-path runs/sac-pendulum.pt
```

The end-of-run summary prints the path and the file's SHA-256. `runs/` is
gitignored. `demo` and discrete `ppo` reject the flag: a random agent has no
policy, and the discrete actor has no checkpoint rule yet.

## Loading

```python
from roborl.io import load_policy

policy = load_policy("runs/sac-pendulum.pt")  # device="cpu" by default; "cuda", "mps", "auto"
action = policy.act(obs)  # (act_dim,) float32 in the env's own units
policy.spec  # PolicySpec: algo, obs_dim, act_dim, action bounds, arch
policy.metadata  # provenance dict, see below
policy.sha256  # digest of the file that was loaded
```

`act` is deterministic (no sampling), accepts one observation `(obs_dim,)`
or a batch `(B, obs_dim)`, rejects the wrong size and non-finite values, and
returns actions inside `[action_low, action_high]`.

## What `act` computes

Each rule mirrors the algorithm's own evaluation path, so a loaded policy
behaves like the actor did in the loop's eval episodes.

| `algo` | Actor class | Deterministic action | Input preprocessing |
|---|---|---|---|
| `sac` | `algos.sac.sac.Actor` | `tanh(mean) · action_scale + action_bias` (buffers in the state dict) | none |
| `flashsac` | `algos.flashsac.networks.FlashSACActor` | `eval_action` = `tanh(mean)` in `[-1, 1]`, then the affine map onto the true bounds that `RescaleAction` applied in training | BatchNorm running stats travel with the weights |
| `ppo_continuous` | `algos.ppo.ppo_continuous.Agent` | `actor_mean`, clipped to the bounds like `ClipAction` | frozen `NormalizeObservation` statistics (env 0 at save time), then the `[-10, 10]` clip |

## File format, version 1

One `.pt` file written by `torch.save`, holding a plain dict:

| Key | Content |
|---|---|
| `format` | `"roborl.policy"` |
| `format_version` | `1` |
| `spec` | `PolicySpec` as a dict: `algo`, `obs_dim`, `act_dim`, `action_low`, `action_high`, `arch` |
| `metadata` | provenance, JSON primitives only |
| `actor_state_dict` | the actor's `state_dict()`, tensors on CPU |
| `obs_normalizer` | `null`, or `{mean, var, epsilon, clip}` for `ppo_continuous` |

`arch` holds the constructor arguments beyond `(obs_dim, act_dim)`:
`hidden`, `num_blocks`, `use_rmsnorm` for `flashsac`; nothing for `sac` and
`ppo_continuous`. Any change to what a key
means bumps `format_version`; a loader refuses versions it does not know.

## Provenance (`metadata`)

Written by `roborl.io.policy_metadata` in every training loop:
`roborl_version`, `git_sha`, `git_dirty`, `created_utc`, `python_version`,
`torch_version`, `gymnasium_version`, `resolved_device`, `env_id`, `seed`,
`global_step`, `exp_name`, `run_name`, and `config` (the full frozen config
as a dict). Nothing in it is needed to act; it is there so a result can be
traced back to the code and run that produced it (integrity rules in
`CLAUDE.md`).

## Integrity and safety

- **Content address.** `save_policy` returns the file's SHA-256 and
  `load_policy` recomputes it; downstream recordings store it next to the
  actions the policy produced.
- **Data, not code.** Files are read with `torch.load(weights_only=True)`.
  Only tensors and primitive containers are unpickled; a file carrying
  arbitrary Python objects is refused.
- **Fail at save time.** Before writing, the weights are loaded into an
  actor rebuilt from the spec (`strict=True`), and SAC-style rescaling
  buffers are checked against the spec's bounds. A spec that cannot
  reproduce the actor never reaches disk.
- **Fail at load time.** The same checks run on load, plus the
  normaliser-presence rule (required for `ppo_continuous`, forbidden
  otherwise).

## Adding an algorithm

Three places, one test: a branch in `build_actor` and in
`_deterministic_action` (`src/roborl/io/policy.py`), the actor's `arch`
keys in `ARCH_KEYS`, and a parametrised case in
`tests/unit/test_policy_io.py` that compares `act` against the loop's own
evaluation rule.
