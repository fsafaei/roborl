"""Soft Actor-Critic (Haarnoja et al., 2018/2019) — one training loop, top to bottom.

Diffable against CleanRL's ``sac_continuous_action.py``; the spec note in
``docs/algos/sac.md`` lists the update equations, the implementation details
that matter, and every deliberate deviation from the reference. The largest
one: a single non-autoresetting env with an explicit ``reset()`` reproduces
CleanRL's pre-Gymnasium-1.0 data stream exactly (one stored transition per
step, true final observation at episode end) without the 1.x vector-env
autoreset pitfalls.
"""

from __future__ import annotations

import csv
import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 — torch's universal alias; keeps CleanRL diffability
from torch import nn, optim

from roborl.algos.sac.buffer import ReplayBuffer
from roborl.config import ExperimentConfig
from roborl.envs.factory import make_env
from roborl.telemetry import metrics
from roborl.telemetry.logger import RunLogger
from roborl.utils.device import resolve_device
from roborl.utils.seeding import seed_everything

LOG_STD_MAX = 2.0
LOG_STD_MIN = -5.0


@dataclass(frozen=True)
class SacConfig(ExperimentConfig):
    """SAC hyperparameters — CleanRL's defaults exactly (see docs/algos/sac.md).

    Verification runs must not override any algorithm hyperparameter; any
    deviation gets listed in the benchmark report. Unlike the demo, training
    defaults to ``track=True``: a run that isn't recorded didn't happen.
    """

    exp_name: str = "sac"
    env_id: str = "Pendulum-v1"
    total_timesteps: int = 1_000_000
    track: bool = True
    buffer_size: int = 1_000_000
    """Replay buffer capacity in transitions."""
    gamma: float = 0.99
    """Discount factor."""
    tau: float = 0.005
    """Polyak averaging coefficient for target networks."""
    batch_size: int = 256
    """Minibatch size sampled from the replay buffer."""
    learning_starts: int = 5_000
    """Uniform-random action steps before any gradient update."""
    policy_lr: float = 3e-4
    """Actor learning rate."""
    q_lr: float = 1e-3
    """Critic learning rate (also used for the temperature)."""
    policy_frequency: int = 2
    """Actor updates every N env steps (compensated: N updates when they run)."""
    target_network_frequency: int = 1
    """Target-network polyak update every N env steps."""
    alpha: float = 0.2
    """Fixed entropy temperature, used only when autotune is off."""
    autotune: bool = True
    """Learn the temperature against target entropy -dim(action space)."""
    save_episodes: bool = False
    """Write per-episode returns to runs/{run_name}.csv for benchmark compare."""
    episode_dir: str = "runs"
    """Directory episode CSVs are written to when save_episodes is on."""


class SoftQNetwork(nn.Module):
    """Soft Q-function Q(s, a): 256-256 ReLU MLP over concatenated obs and action."""

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        """Build the MLP for flat observation and action sizes."""
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Return Q-values with shape ``(batch, 1)``."""
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q: torch.Tensor = self.fc3(x)
        return q


class Actor(nn.Module):
    """Squashed-Gaussian policy: tanh(N(mean, std)) rescaled to the action bounds."""

    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def __init__(
        self, obs_dim: int, act_dim: int, action_low: np.ndarray, action_high: np.ndarray
    ) -> None:
        """Build the policy MLP and store the action rescaling as buffers.

        Args:
            obs_dim: Flat observation size.
            act_dim: Flat action size.
            action_low: Lower action bounds, shape ``(act_dim,)``.
            action_high: Upper action bounds, shape ``(act_dim,)``.
        """
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)
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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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
    """Log-density of a tanh-squashed, rescaled Gaussian action.

    Change of variables for ``a = tanh(u) * scale + bias`` with ``u ~ N(mean, std)``:
    ``log pi(a) = log N(u) - sum_j log(scale_j * (1 - tanh(u_j)^2) + 1e-6)``,
    summed over action dimensions. The 1e-6 floor prevents log(0) where tanh
    saturates; the scale factor inside the log matters on envs whose action
    range isn't [-1, 1].

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


@dataclass(frozen=True)
class SacSummary:
    """What a SAC run produced, for the console summary and for tests.

    Attributes:
        episodic_returns: Return of every completed episode, in order.
        episodic_lengths: Length of every completed episode, in order.
        steps: Environment steps actually taken.
        sps: Average environment steps per second.
        wandb_url: The W&B run URL when tracking online, else None.
        episodes_csv: Path of the saved episode log, when enabled.
    """

    episodic_returns: list[float] = field(default_factory=list)
    episodic_lengths: list[int] = field(default_factory=list)
    steps: int = 0
    sps: float = 0.0
    wandb_url: str | None = None
    episodes_csv: str | None = None

    def render(self) -> str:
        """Format the end-of-run console summary."""
        lines = [f"sac finished: {self.steps} steps, {len(self.episodic_returns)} episodes"]
        if self.episodic_returns:
            tail = self.episodic_returns[-10:]
            lines.append(
                f"episodic return (last {len(tail)} episodes): "
                f"{float(np.mean(tail)):.1f} ± {float(np.std(tail)):.1f}"
            )
        lines.append(f"throughput: {self.sps:.0f} steps/s")
        lines.append(
            f"tracked at: {self.wandb_url}" if self.wandb_url else "telemetry: disabled or offline"
        )
        if self.episodes_csv:
            lines.append(f"episode log: {self.episodes_csv}")
        return "\n".join(lines)


def run_sac(config: SacConfig) -> SacSummary:
    """Train SAC top to bottom; the structure mirrors CleanRL's script.

    Args:
        config: The experiment configuration.

    Returns:
        A summary of the run.

    Raises:
        ValueError: If the environment's action space is not a ``Box``.
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
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError(f"SAC needs a continuous (Box) action space; got {env.action_space}.")
    assert env.observation_space.shape is not None
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    actor = Actor(obs_dim, act_dim, env.action_space.low, env.action_space.high).to(device)
    qf1 = SoftQNetwork(obs_dim, act_dim).to(device)
    qf2 = SoftQNetwork(obs_dim, act_dim).to(device)
    qf1_target = SoftQNetwork(obs_dim, act_dim).to(device)
    qf2_target = SoftQNetwork(obs_dim, act_dim).to(device)
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

    rb = ReplayBuffer(config.buffer_size, env.observation_space.shape, env.action_space.shape)
    returns: list[float] = []
    lengths: list[int] = []
    actor_loss_val = float("nan")
    alpha_loss_val = float("nan")
    start = time.perf_counter()

    obs, _ = env.reset()
    obs = obs.astype(np.float32)
    for global_step in range(config.total_timesteps):
        if global_step < config.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action_t, _, _ = actor.get_action(torch.as_tensor(obs, device=device).unsqueeze(0))
            action = action_t.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = next_obs.astype(np.float32)
        # next_obs is the true final observation at episode end (no autoreset),
        # and the stored done is `terminated` only — truncation bootstraps.
        rb.add(obs, action, float(reward), next_obs, terminated)

        if terminated or truncated:
            episode = info["episode"]
            returns.append(float(episode["r"]))
            lengths.append(int(episode["l"]))
            logger.log(
                {
                    metrics.EPISODIC_RETURN: returns[-1],
                    metrics.EPISODIC_LENGTH: lengths[-1],
                },
                step=global_step,
            )
            obs, _ = env.reset()
            obs = obs.astype(np.float32)
        else:
            obs = next_obs

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
                scalars = {
                    metrics.QF1_VALUES: qf1_a_values.mean().item(),
                    metrics.QF2_VALUES: qf2_a_values.mean().item(),
                    metrics.QF1_LOSS: qf1_loss.item(),
                    metrics.QF2_LOSS: qf2_loss.item(),
                    metrics.QF_LOSS: qf_loss.item() / 2.0,
                    metrics.ACTOR_LOSS: actor_loss_val,
                    metrics.ALPHA: alpha,
                    metrics.SPS: global_step / (time.perf_counter() - start),
                }
                if config.autotune:
                    scalars[metrics.ALPHA_LOSS] = alpha_loss_val
                logger.log(scalars, step=global_step)

    elapsed = time.perf_counter() - start
    episodes_csv = _save_episode_log(config, returns, lengths) if config.save_episodes else None
    summary = SacSummary(
        episodic_returns=returns,
        episodic_lengths=lengths,
        steps=config.total_timesteps,
        sps=config.total_timesteps / elapsed,
        wandb_url=logger.url,
        episodes_csv=episodes_csv,
    )
    logger.finish()
    env.close()
    return summary


def _save_episode_log(config: SacConfig, returns: list[float], lengths: list[int]) -> str:
    """Write per-episode curves in the benchmark compare input format.

    Duplicated from the demo (second use — promotion to shared core waits
    for the third, per ADR 0003).
    """
    path = Path(config.episode_dir) / f"{config.run_name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "global_step", "episodic_return"])
        step_of_episode_end = itertools.accumulate(lengths)
        for step, episodic_return in zip(step_of_episode_end, returns, strict=True):
            writer.writerow([config.run_name, step, episodic_return])
    return str(path)
