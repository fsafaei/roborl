"""HER + SAC (Andrychowicz et al., 2017 on Haarnoja et al., 2019) — one training loop.

This file is ``sac.py`` with exactly the deltas listed in
``docs/algos/her.md`` ("her_sac.py — deltas from sac.py"): goal-conditioned
dict observations flattened to ``concat(observation, desired_goal)``, the
episode-aware HER replay buffer (``buffer.py``) in place of the uniform
ring buffer, configurable MLP widths, the Fetch recipe hyperparameters, a
success-rate signal, and a periodic deterministic evaluation pass. Every
SAC update equation, the twin critics, temperature autotuning, polyak
targets, ``terminated``-only dones, and the single non-autoresetting env
with an explicit ``reset()`` are untouched — HER changes the data, not the
algorithm. Keep this file diffable against ``sac.py``.
"""

from __future__ import annotations

import csv
import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 — torch's universal alias; keeps CleanRL diffability
from torch import nn, optim

from roborl.algos.her.buffer import GoalStrategy, HerReplayBuffer
from roborl.algos.her.goals import (
    check_goal_env,
    episode_success,
    flatten_goal_obs,
    goal_space_dims,
)
from roborl.config import ExperimentConfig
from roborl.envs.factory import make_env
from roborl.telemetry import metrics
from roborl.telemetry.logger import RunLogger
from roborl.utils.device import resolve_device
from roborl.utils.seeding import seed_everything

LOG_STD_MAX = 2.0
LOG_STD_MIN = -5.0
Q_LOWER_BOUND_SLACK = 1.05
"""The lower-bound diagnostic fires below ``-1/(1-gamma)`` times this slack."""


@dataclass(frozen=True)
class HerSacConfig(ExperimentConfig):
    """HER + SAC hyperparameters — the rl-baselines3-zoo Fetch recipe (docs/algos/her.md).

    The SAC machinery is CleanRL's; the values are the zoo's TQC Fetch entry
    adapted to SAC (two critics, no quantile truncation), shared verbatim
    with the local SB3 reference runs (ADR 0008). Verification runs must not
    override any of them. Training defaults to ``track=True``: a run that
    isn't recorded didn't happen.
    """

    exp_name: str = "her-sac"
    env_id: str = "FetchPush-v4"
    total_timesteps: int = 1_000_000
    track: bool = True
    buffer_size: int = 1_000_000
    """Replay buffer capacity in transitions (episodes = this // max_episode_steps)."""
    gamma: float = 0.95
    """Discount factor — the Fetch recipe's 0.95, not SAC's 0.99 (value bound -20)."""
    tau: float = 0.05
    """Polyak averaging coefficient for target networks (10x SAC's default)."""
    batch_size: int = 2048
    """Minibatch size sampled (and relabeled) from the replay buffer."""
    learning_starts: int = 1_000
    """Uniform-random action steps before any gradient update (shared with the SB3 runs)."""
    policy_lr: float = 1e-3
    """Actor learning rate."""
    q_lr: float = 1e-3
    """Critic learning rate (also used for the temperature) — one rate everywhere."""
    policy_frequency: int = 2
    """Actor updates every N env steps (compensated: N updates when they run)."""
    target_network_frequency: int = 1
    """Target-network polyak update every N env steps."""
    alpha: float = 0.2
    """Fixed entropy temperature, used only when autotune is off."""
    autotune: bool = True
    """Learn the temperature against target entropy -dim(action space)."""
    net_arch: tuple[int, ...] = (512, 512, 512)
    """Hidden layer widths of the actor and both critics."""
    her_enabled: bool = True
    """Relabel goals at sample time. False = ablation rungs R0/R1: same storage, no relabeling."""
    her_strategy: GoalStrategy = "future"
    """Goal-selection strategy: future (own successor onward), final, or episode."""
    her_k: int = 4
    """SB3's n_sampled_goal: relabeled fraction of every batch is k / (k + 1)."""
    eval_interval: int = 10_000
    """Env steps between deterministic evaluation passes (0 disables evaluation)."""
    eval_episodes: int = 20
    """Episodes per evaluation pass, on a separate env seeded seed + 1000."""
    save_episodes: bool = False
    """Write per-episode returns (and successes) to runs/{run_name}*.csv for benchmark compare."""
    episode_dir: str = "runs"
    """Directory episode CSVs are written to when save_episodes is on."""


def _mlp(in_dim: int, hidden_sizes: tuple[int, ...]) -> tuple[nn.Sequential, int]:
    """ReLU MLP trunk over the given widths; returns it with its output width."""
    layers: list[nn.Module] = []
    width = in_dim
    for hidden in hidden_sizes:
        layers += [nn.Linear(width, hidden), nn.ReLU()]
        width = hidden
    return nn.Sequential(*layers), width


class SoftQNetwork(nn.Module):
    """Soft Q-function Q(s, a): ReLU MLP of configurable widths over concatenated obs and action."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        """Build the MLP for flat (goal-augmented) observation and action sizes."""
        super().__init__()
        self.trunk, width = _mlp(obs_dim + act_dim, hidden_sizes)
        self.fc_out = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Return Q-values with shape ``(batch, 1)``."""
        q: torch.Tensor = self.fc_out(self.trunk(torch.cat([x, a], 1)))
        return q


class Actor(nn.Module):
    """Squashed-Gaussian policy: tanh(N(mean, std)) rescaled to the action bounds."""

    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        """Build the policy MLP and store the action rescaling as buffers.

        Args:
            obs_dim: Flat (goal-augmented) observation size.
            act_dim: Flat action size.
            action_low: Lower action bounds, shape ``(act_dim,)``.
            action_high: Upper action bounds, shape ``(act_dim,)``.
            hidden_sizes: Hidden layer widths.
        """
        super().__init__()
        self.trunk, width = _mlp(obs_dim, hidden_sizes)
        self.fc_mean = nn.Linear(width, act_dim)
        self.fc_logstd = nn.Linear(width, act_dim)
        self.register_buffer(
            "action_scale", torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the Gaussian's ``(mean, log_std)``, each ``(batch, act_dim)``.

        log_std is tanh-squashed into [LOG_STD_MIN, LOG_STD_MAX] (SpinUp /
        Denis Yarats style) rather than clamped, keeping gradients alive at
        the bounds.
        """
        x = self.trunk(x)
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a reparameterized action.

        Returns:
            ``(action, log_prob, mean_action)``: the rsampled squashed action
            ``(batch, act_dim)``, its log-density ``(batch, 1)``, and the
            deterministic squashed mean action (what evaluation uses).
        """
        mean, log_std = self(x)
        x_t = torch.distributions.Normal(mean, log_std.exp()).rsample()
        action = torch.tanh(x_t) * self.action_scale + self.action_bias
        log_prob = squashed_gaussian_log_prob(x_t, mean, log_std, self.action_scale)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


def squashed_gaussian_log_prob(
    x_t: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor, action_scale: torch.Tensor
) -> torch.Tensor:
    """Log-density of a tanh-squashed, rescaled Gaussian action (duplicated from sac.py).

    Change of variables for ``a = tanh(u) * scale + bias`` with ``u ~ N(mean, std)``:
    ``log pi(a) = log N(u) - sum_j log(scale_j * (1 - tanh(u_j)^2) + 1e-6)``,
    summed over action dimensions.

    Args:
        x_t: Pre-squash samples ``u``, shape ``(batch, act_dim)``.
        mean: Gaussian means, same shape.
        log_std: Gaussian log standard deviations, same shape.
        action_scale: Per-dimension half-range of the action space.

    Returns:
        Log-probabilities with shape ``(batch, 1)``.
    """
    log_prob: torch.Tensor = torch.distributions.Normal(mean, log_std.exp()).log_prob(x_t)
    log_prob = log_prob - torch.log(action_scale * (1 - torch.tanh(x_t).pow(2)) + 1e-6)
    return log_prob.sum(1, keepdim=True)


def soft_td_target(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    alpha: float,
    min_q_next: torch.Tensor,
    next_log_pi: torch.Tensor,
) -> torch.Tensor:
    """TD target ``y = r + gamma * (1 - d) * (min Q_target(s', a') - alpha * log pi(a'|s'))``.

    Duplicated verbatim from ``sac.py`` (rule of three, ADR 0003).

    Args:
        rewards: Batch rewards, shape ``(batch,)`` — 1-D enforced, because a
            ``(batch, 1)`` here broadcasts into a ``(batch, batch)`` target.
        dones: True-termination flags, shape ``(batch,)``.
        gamma: Discount factor.
        alpha: Entropy temperature.
        min_q_next: Min over target critics at ``(s', a')``, shape ``(batch, 1)``.
        next_log_pi: Log-density of the fresh next action, shape ``(batch, 1)``.

    Returns:
        Targets with shape ``(batch,)``.

    Raises:
        ValueError: If ``rewards`` or ``dones`` is not 1-D.
    """
    if rewards.ndim != 1 or dones.ndim != 1:
        raise ValueError(
            f"rewards and dones must be 1-D, got shapes {tuple(rewards.shape)} "
            f"and {tuple(dones.shape)}."
        )
    soft_value = (min_q_next - alpha * next_log_pi).view(-1)
    return rewards + (1.0 - dones) * gamma * soft_value


def q_lower_bound_violation(min_q: torch.Tensor, gamma: float) -> float:
    """Fraction of Q predictions below the sparse-reward value floor ``-1/(1-gamma)`` (with slack).

    On Fetch every reward is in ``{-1, 0}`` and episodes never terminate, so
    no return can lie below ``-1/(1-gamma)`` (``-20`` at ``gamma = 0.95``).
    Predictions below ``-1/(1-gamma) * 1.05`` are a divergence signal — this
    is a diagnostic only; nothing is clipped (Baselines clipped, SB3 does not).

    Args:
        min_q: ``min(Q1, Q2)`` predictions on the minibatch, any shape.
        gamma: Discount factor.

    Returns:
        The violating fraction in ``[0, 1]``.
    """
    bound = -1.0 / (1.0 - gamma) * Q_LOWER_BOUND_SLACK
    return (min_q < bound).float().mean().item()


@dataclass(frozen=True)
class EvalResult:
    """One deterministic evaluation pass.

    Attributes:
        success_rate: Mean final-step ``is_success`` over the episodes.
        episodic_return: Mean undiscounted return.
        episodic_length: Mean episode length.
    """

    success_rate: float
    episodic_return: float
    episodic_length: float


def evaluate_policy(actor: Actor, env: gym.Env, episodes: int, device: torch.device) -> EvalResult:
    """Run deterministic (mean-action) episodes on a separate, already-seeded env.

    Touches neither the training env, the replay buffer, nor NumPy's global
    RNG — the eval env carries its own RNG stream from the factory.
    Episode success is the **final** step's ``is_success``.

    Args:
        actor: The policy; its squashed mean action is used.
        env: The evaluation env (factory-built, ``seed + 1000``).
        episodes: Number of episodes to run.
        device: Device the actor lives on.

    Returns:
        Means over the episodes.
    """
    successes: list[float] = []
    returns: list[float] = []
    lengths: list[int] = []
    for _ in range(episodes):
        obs_dict, _ = env.reset()
        done = False
        while not done:
            obs = torch.as_tensor(flatten_goal_obs(obs_dict), device=device).unsqueeze(0)
            with torch.no_grad():
                _, _, mean_action = actor.get_action(obs)
            obs_dict, _, terminated, truncated, info = env.step(
                mean_action.squeeze(0).cpu().numpy()
            )
            done = terminated or truncated
        successes.append(episode_success(info))
        returns.append(float(info["episode"]["r"]))
        lengths.append(int(info["episode"]["l"]))
    return EvalResult(
        success_rate=float(np.mean(successes)),
        episodic_return=float(np.mean(returns)),
        episodic_length=float(np.mean(lengths)),
    )


@dataclass(frozen=True)
class HerSacSummary:
    """What a HER+SAC run produced, for the console summary and for tests.

    Attributes:
        episodic_returns: Return of every completed training episode, in order.
        episodic_lengths: Length of every completed training episode, in order.
        episodic_successes: Final-step success (0/1) of every training episode.
        eval_steps: Global steps at which evaluation passes ran.
        eval_success_rates: Deterministic-policy success rate per pass.
        steps: Environment steps actually taken.
        sps: Average environment steps per second.
        wandb_url: The W&B run URL when tracking online, else None.
        episodes_csv: Path of the saved episode log, when enabled.
        success_csv: Path of the saved per-episode success log, when enabled.
    """

    episodic_returns: list[float] = field(default_factory=list)
    episodic_lengths: list[int] = field(default_factory=list)
    episodic_successes: list[float] = field(default_factory=list)
    eval_steps: list[int] = field(default_factory=list)
    eval_success_rates: list[float] = field(default_factory=list)
    steps: int = 0
    sps: float = 0.0
    wandb_url: str | None = None
    episodes_csv: str | None = None
    success_csv: str | None = None

    def render(self) -> str:
        """Format the end-of-run console summary."""
        lines = [f"her-sac finished: {self.steps} steps, {len(self.episodic_returns)} episodes"]
        if self.episodic_returns:
            tail = self.episodic_returns[-10:]
            tail_success = self.episodic_successes[-10:]
            lines.append(
                f"episodic return (last {len(tail)} episodes): "
                f"{float(np.mean(tail)):.1f} ± {float(np.std(tail)):.1f}; "
                f"success rate {float(np.mean(tail_success)):.2f}"
            )
        if self.eval_success_rates:
            lines.append(
                f"eval success rate (last pass, step {self.eval_steps[-1]}): "
                f"{self.eval_success_rates[-1]:.2f}"
            )
        lines.append(f"throughput: {self.sps:.0f} steps/s")
        lines.append(
            f"tracked at: {self.wandb_url}" if self.wandb_url else "telemetry: disabled or offline"
        )
        if self.episodes_csv:
            lines.append(f"episode log: {self.episodes_csv}")
        if self.success_csv:
            lines.append(f"success log: {self.success_csv}")
        return "\n".join(lines)


def run_her_sac(
    config: HerSacConfig,
    buffer_audit: Callable[[HerReplayBuffer], None] | None = None,
) -> HerSacSummary:
    """Train HER + SAC top to bottom; the structure mirrors ``sac.py``.

    Args:
        config: The experiment configuration.
        buffer_audit: Optional hook called with the replay buffer after
            training, before teardown — the smoke test uses it to audit the
            stored episodes (fixed horizon, no terminations, no autoreset
            leakage). Never set in production.

    Returns:
        A summary of the run.

    Raises:
        ValueError: If the environment violates the goal-env contract or has
            no registered ``max_episode_steps``.
    """
    seed_everything(config.seed)
    device = resolve_device(config.device)

    logger = RunLogger(config, resolved_device=str(device))
    logger.start()

    env = make_env(
        config.env_id,
        seed=config.seed,
        capture_video=config.capture_video,
        video_dir=f"{config.video_dir}/{config.run_name}",
    )()
    compute_reward = check_goal_env(env)  # also asserts the Box action space
    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.action_space.shape is not None
    obs_dim, goal_dim = goal_space_dims(env.observation_space)
    in_dim = obs_dim + goal_dim
    act_dim = int(np.prod(env.action_space.shape))
    max_episode_steps = env.spec.max_episode_steps if env.spec is not None else None
    if max_episode_steps is None:
        raise ValueError(
            f"{config.env_id} registers no max_episode_steps; the HER buffer is episode-major."
        )
    # A separate, identically wrapped env for deterministic evaluation; its
    # own RNG stream, no video, never touched by training.
    eval_env = make_env(config.env_id, seed=config.seed + 1000)()

    actor = Actor(in_dim, act_dim, env.action_space.low, env.action_space.high, config.net_arch).to(
        device
    )
    qf1 = SoftQNetwork(in_dim, act_dim, config.net_arch).to(device)
    qf2 = SoftQNetwork(in_dim, act_dim, config.net_arch).to(device)
    qf1_target = SoftQNetwork(in_dim, act_dim, config.net_arch).to(device)
    qf2_target = SoftQNetwork(in_dim, act_dim, config.net_arch).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=config.q_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=config.policy_lr)

    if config.autotune:
        target_entropy = -float(act_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=config.q_lr)
    else:
        alpha = config.alpha

    rb = HerReplayBuffer(
        config.buffer_size,
        max_episode_steps,
        (obs_dim,),
        (goal_dim,),
        env.action_space.shape,
        compute_reward,
        her_k=config.her_k,
        strategy=config.her_strategy,
        her_enabled=config.her_enabled,
    )
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    eval_steps: list[int] = []
    eval_success_rates: list[float] = []
    actor_loss_val = float("nan")
    alpha_loss_val = float("nan")
    start = time.perf_counter()

    obs_dict, _ = env.reset()
    obs = flatten_goal_obs(obs_dict)
    for global_step in range(config.total_timesteps):
        if global_step < config.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action_t, _, _ = actor.get_action(torch.as_tensor(obs, device=device).unsqueeze(0))
            action = action_t.squeeze(0).cpu().numpy()

        next_obs_dict, reward, terminated, truncated, info = env.step(action)
        next_obs = flatten_goal_obs(next_obs_dict)
        # next_obs is the true final observation at episode end (no autoreset),
        # and the stored done is `terminated` only — truncation bootstraps.
        # All three dict pieces go in raw; the buffer casts to float32.
        rb.add(
            obs_dict["observation"],
            obs_dict["achieved_goal"],
            obs_dict["desired_goal"],
            action,
            float(reward),
            next_obs_dict["observation"],
            next_obs_dict["achieved_goal"],
            terminated,
        )

        if terminated or truncated:
            rb.commit_episode()  # the episode becomes sampleable only now
            episode = info["episode"]
            returns.append(float(episode["r"]))
            lengths.append(int(episode["l"]))
            successes.append(episode_success(info))  # the FINAL step's flag
            logger.log(
                {
                    metrics.EPISODIC_RETURN: returns[-1],
                    metrics.EPISODIC_LENGTH: lengths[-1],
                    metrics.SUCCESS_RATE: successes[-1],
                },
                step=global_step,
            )
            obs_dict, _ = env.reset()
            obs = flatten_goal_obs(obs_dict)
        else:
            obs_dict = next_obs_dict
            obs = next_obs

        if config.eval_interval > 0 and (global_step + 1) % config.eval_interval == 0:
            result = evaluate_policy(actor, eval_env, config.eval_episodes, device)
            eval_steps.append(global_step + 1)
            eval_success_rates.append(result.success_rate)
            logger.log(
                {
                    metrics.EVAL_SUCCESS_RATE: result.success_rate,
                    metrics.EVAL_EPISODIC_RETURN: result.episodic_return,
                    metrics.EVAL_EPISODIC_LENGTH: result.episodic_length,
                },
                step=global_step,
            )

        if global_step > config.learning_starts:
            data = rb.sample(config.batch_size, device)
            with torch.no_grad():
                next_action, next_log_pi, _ = actor.get_action(data.next_observations)
                min_q_next = torch.min(
                    qf1_target(data.next_observations, next_action),
                    qf2_target(data.next_observations, next_action),
                )
                next_q_value = soft_td_target(
                    data.rewards, data.dones, config.gamma, alpha, min_q_next, next_log_pi
                )

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % config.policy_frequency == 0:
                # Delayed update, compensated: run policy_frequency updates.
                for _ in range(config.policy_frequency):
                    pi, log_pi, _ = actor.get_action(data.observations)
                    min_qf_pi = torch.min(qf1(data.observations, pi), qf2(data.observations, pi))
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    actor_loss_val = actor_loss.item()

                    if config.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()
                        alpha_loss_val = alpha_loss.item()

            if global_step % config.target_network_frequency == 0:
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters(), strict=True
                ):
                    target_param.data.copy_(
                        config.tau * param.data + (1 - config.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters(), strict=True
                ):
                    target_param.data.copy_(
                        config.tau * param.data + (1 - config.tau) * target_param.data
                    )

            if global_step % 100 == 0:
                virtual_fraction, virtual_zero_fraction = rb.last_sample_stats
                scalars = {
                    metrics.QF1_VALUES: qf1_a_values.mean().item(),
                    metrics.QF2_VALUES: qf2_a_values.mean().item(),
                    metrics.QF1_LOSS: qf1_loss.item(),
                    metrics.QF2_LOSS: qf2_loss.item(),
                    metrics.QF_LOSS: qf_loss.item() / 2.0,
                    metrics.ACTOR_LOSS: actor_loss_val,
                    metrics.ALPHA: alpha,
                    metrics.SPS: global_step / (time.perf_counter() - start),
                    metrics.HER_VIRTUAL_FRACTION: virtual_fraction,
                    metrics.HER_VIRTUAL_REWARD_ZERO_FRACTION: virtual_zero_fraction,
                    metrics.Q_LOWER_BOUND_VIOLATION: q_lower_bound_violation(
                        torch.min(qf1_a_values, qf2_a_values).detach(), config.gamma
                    ),
                }
                if config.autotune:
                    scalars[metrics.ALPHA_LOSS] = alpha_loss_val
                logger.log(scalars, step=global_step)

    elapsed = time.perf_counter() - start
    if buffer_audit is not None:
        buffer_audit(rb)
    episodes_csv = success_csv = None
    if config.save_episodes:
        episodes_csv, success_csv = _save_episode_log(config, returns, lengths, successes)
    summary = HerSacSummary(
        episodic_returns=returns,
        episodic_lengths=lengths,
        episodic_successes=successes,
        eval_steps=eval_steps,
        eval_success_rates=eval_success_rates,
        steps=config.total_timesteps,
        sps=config.total_timesteps / elapsed,
        wandb_url=logger.url,
        episodes_csv=episodes_csv,
        success_csv=success_csv,
    )
    logger.finish()
    eval_env.close()
    env.close()
    return summary


def _save_episode_log(
    config: HerSacConfig, returns: list[float], lengths: list[int], successes: list[float]
) -> tuple[str, str]:
    """Write per-episode curves in the benchmark compare input format, plus successes.

    ``{run_name}.csv`` is the 3-column format ``compare`` consumes;
    ``{run_name}-success.csv`` carries ``episodic_success`` (0/1) for the
    ablation figures. Fourth local copy of the writer — promotion to shared
    core is overdue under the rule of three and happens in its own PR.
    """
    out = Path(config.episode_dir)
    out.mkdir(parents=True, exist_ok=True)
    steps = list(itertools.accumulate(lengths))
    episodes_path = out / f"{config.run_name}.csv"
    with episodes_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "global_step", "episodic_return"])
        for step, episodic_return in zip(steps, returns, strict=True):
            writer.writerow([config.run_name, step, episodic_return])
    success_path = out / f"{config.run_name}-success.csv"
    with success_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "global_step", "episodic_success"])
        for step, success in zip(steps, successes, strict=True):
            writer.writerow([config.run_name, step, int(success)])
    return str(episodes_path), str(success_path)
