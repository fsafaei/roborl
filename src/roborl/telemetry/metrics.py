"""Canonical metric names.

Algorithm code must import these constants instead of hand-typing metric
strings — a typo in a metric name silently splits a chart into two, and
grep-ability of the constants is how we keep the telemetry contract honest.

Names in ``charts/`` and ``losses/`` mirror CleanRL exactly so our curves
overlay 1:1 with reference runs in a single W&B workspace. Metrics we add
beyond CleanRL live in their own namespaces: ``diagnostics/`` for training
internals, ``eval/`` for deterministic-policy evaluation. The x-axis for
everything is ``global_step`` (environment steps). See ``docs/telemetry.md``
for how to read each metric.
"""

from __future__ import annotations

from typing import Final

GLOBAL_STEP: Final = "global_step"

# CleanRL-compatible (charts/)
EPISODIC_RETURN: Final = "charts/episodic_return"
EPISODIC_LENGTH: Final = "charts/episodic_length"
SPS: Final = "charts/SPS"
LEARNING_RATE: Final = "charts/learning_rate"

# CleanRL-compatible (losses/) — logged by algorithms as they land.
# SAC (mirrors sac_continuous_action.py; qf_loss is the mean of the two).
QF1_VALUES: Final = "losses/qf1_values"
QF2_VALUES: Final = "losses/qf2_values"
QF1_LOSS: Final = "losses/qf1_loss"
QF2_LOSS: Final = "losses/qf2_loss"
QF_LOSS: Final = "losses/qf_loss"
ACTOR_LOSS: Final = "losses/actor_loss"
ALPHA: Final = "losses/alpha"
ALPHA_LOSS: Final = "losses/alpha_loss"

VALUE_LOSS: Final = "losses/value_loss"
POLICY_LOSS: Final = "losses/policy_loss"
ENTROPY: Final = "losses/entropy"
APPROX_KL: Final = "losses/approx_kl"
OLD_APPROX_KL: Final = "losses/old_approx_kl"
CLIPFRAC: Final = "losses/clipfrac"
EXPLAINED_VARIANCE: Final = "losses/explained_variance"

# roborl additions (diagnostics/): training internals beyond CleanRL's set.
GRAD_NORM: Final = "diagnostics/grad_norm"
PARAM_NORM: Final = "diagnostics/param_norm"

# FlashSAC (see docs/algos/flashsac.md for what healthy looks like).
TARGET_CLAMP_FRACTION: Final = "diagnostics/target_clamp_fraction"
REWARD_SCALE: Final = "diagnostics/reward_scale"
RETURN_RMS_VAR: Final = "diagnostics/return_rms_var"
TARGET_DIST_ENTROPY: Final = "diagnostics/target_dist_entropy"
CRITIC_FEATURE_NORM: Final = "diagnostics/critic_feature_norm"
NOISE_REPEAT_LEN: Final = "diagnostics/noise_repeat_len"
TARGET_ENTROPY: Final = "diagnostics/target_entropy"

# roborl additions (eval/): deterministic-policy evaluation, distinct from
# the exploration-noised training return.
EVAL_EPISODIC_RETURN: Final = "eval/episodic_return"
EVAL_EPISODIC_LENGTH: Final = "eval/episodic_length"
