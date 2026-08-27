"""Proximal Policy Optimization (Schulman et al., 2017) — one training loop, top to bottom.

Diffable against CleanRL's ``ppo.py`` (the discrete, classic-control
variant); the spec note in ``docs/algos/ppo.md`` lists the 13 core
implementation details, the loop mechanics between them, and every
deliberate deviation from the reference. The largest one:
``SyncVectorEnv(autoreset_mode=SAME_STEP)`` reproduces CleanRL's
pre-Gymnasium-1.0 data stream exactly (the step that ends an episode
returns the reset observation, done = 1) without the 1.x NEXT_STEP phantom
transition per episode.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical

from roborl.config import ExperimentConfig
from roborl.envs.factory import make_env
from roborl.telemetry import metrics
from roborl.telemetry.logger import RunLogger
from roborl.utils.device import resolve_device
from roborl.utils.seeding import seed_everything


@dataclass(frozen=True)
class PpoConfig(ExperimentConfig):
    """PPO hyperparameters — CleanRL's defaults exactly (see docs/algos/ppo.md).

    Verification runs must not override any algorithm hyperparameter; any
    deviation gets listed in the benchmark report. Unlike the demo, training
    defaults to ``track=True``: a run that isn't recorded didn't happen.
    """

    exp_name: str = "ppo"
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500_000
    track: bool = True
    learning_rate: float = 2.5e-4
    """Adam learning rate (eps 1e-5), linearly annealed when anneal_lr is on."""
    num_envs: int = 4
    """Parallel environments stepped in lockstep."""
    num_steps: int = 128
    """Rollout length per environment per iteration (batch = num_envs * num_steps)."""
    anneal_lr: bool = True
    """Linearly decay the learning rate to 0 over the run."""
    gamma: float = 0.99
    """Discount factor."""
    gae_lambda: float = 0.95
    """GAE bias/variance trade-off."""
    num_minibatches: int = 4
    """Minibatches per epoch (minibatch = batch / num_minibatches)."""
    update_epochs: int = 4
    """Gradient epochs over each rollout batch."""
    norm_adv: bool = True
    """Normalize advantages per minibatch."""
    clip_coef: float = 0.2
    """Surrogate (and value) clipping coefficient epsilon."""
    clip_vloss: bool = True
    """Use the clipped value loss (CleanRL default, kept for parity)."""
    ent_coef: float = 0.01
    """Entropy bonus coefficient."""
    vf_coef: float = 0.5
    """Value loss coefficient."""
    max_grad_norm: float = 0.5
    """Global gradient-norm clip over all parameters."""
    target_kl: float | None = None
    """Stop an iteration's epochs early when approx_kl exceeds this (off: None)."""
    save_episodes: bool = False
    """Write per-episode returns to runs/{run_name}.csv for benchmark compare."""
    episode_dir: str = "runs"
    """Directory episode CSVs are written to when save_episodes is on."""

    @property
    def batch_size(self) -> int:
        """Transitions per rollout: ``num_envs * num_steps``."""
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        """Transitions per gradient step: ``batch_size // num_minibatches``."""
        return self.batch_size // self.num_minibatches

    @property
    def num_iterations(self) -> int:
        """Rollout-update iterations: ``total_timesteps // batch_size``."""
        return self.total_timesteps // self.batch_size


def layer_init(
    layer: nn.Linear, std: float = float(np.sqrt(2)), bias_const: float = 0.0
) -> nn.Linear:
    """Orthogonal weight init with gain ``std``, constant bias (detail 2)."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Separate 64-64 tanh policy and value MLPs (details 2 and 13)."""

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        """Build both MLPs for a flat observation size and a discrete action count."""
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Return state values with shape ``(batch, 1)``."""
        value: torch.Tensor = self.critic(x)
        return value

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or score a given) action under the Categorical policy.

        Args:
            x: Observations, shape ``(batch, obs_dim)``.
            action: When given, score these actions instead of sampling —
                this is how updates recompute log-probs of rollout actions
                (detail 15).

        Returns:
            ``(action, log_prob, entropy, value)`` with shapes
            ``(batch,)``, ``(batch,)``, ``(batch,)``, ``(batch, 1)``.
        """
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation over a rollout (details 5 and the target).

    Backward recursion ``A_t = delta_t + gamma * lambda * (1 - d_{t+1}) * A_{t+1}``
    with ``delta_t = r_t + gamma * V(s_{t+1}) * (1 - d_{t+1}) - V(s_t)``.
    ``dones[t]`` flags whether ``obs[t]`` began a new episode, so step t masks
    with the *next* row's flag; the last row uses ``next_done``/``next_value``
    from past the rollout's end. Value targets are ``A_t + V(s_t)`` — old
    values, not Monte-Carlo returns.

    Args:
        rewards: Rollout rewards, shape ``(num_steps, num_envs)``.
        values: Rollout value estimates ``V(obs[t])``, same shape.
        dones: Episode-start flags for ``obs[t]``, same shape.
        next_value: ``V`` of the observation after the last step, ``(num_envs,)``.
        next_done: Done flag carried past the last step, ``(num_envs,)``.
        gamma: Discount factor.
        gae_lambda: GAE lambda.

    Returns:
        ``(advantages, returns)``, each ``(num_steps, num_envs)``.

    Raises:
        ValueError: If the rollout tensors are not 2-D with matching shapes.
    """
    if not (rewards.ndim == 2 and rewards.shape == values.shape == dones.shape):
        raise ValueError(
            "rewards, values, and dones must share a (num_steps, num_envs) shape; got "
            f"{tuple(rewards.shape)}, {tuple(values.shape)}, {tuple(dones.shape)}."
        )
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(next_value)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam
    return advantages, advantages + values


def clipped_policy_loss(
    new_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clipped surrogate objective and its debug statistics (details 8 and 12).

    Implemented sign-flipped as ``max(-A*r, -A*clip(r, 1-eps, 1+eps))``, meaned.
    Advantages arrive already normalized when norm_adv is on — normalization
    is the loop's job (per minibatch), not this helper's.

    Args:
        new_logprob: Recomputed log-probs of the rollout actions, ``(batch,)``.
        old_logprob: Log-probs recorded at rollout time, ``(batch,)``.
        advantages: Advantage estimates, ``(batch,)``.
        clip_coef: Clipping coefficient epsilon.

    Returns:
        ``(pg_loss, approx_kl, old_approx_kl, clipfrac)``: the scalar loss
        (with gradients) and the three detached scalar diagnostics —
        ``mean((r-1) - log r)``, ``mean(-log r)``, and the fraction of ratios
        with ``|r - 1| > eps``.
    """
    logratio = new_logprob - old_logprob
    ratio = logratio.exp()
    with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    return pg_loss, approx_kl, old_approx_kl, clipfrac


def clipped_value_loss(
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_coef: float,
) -> torch.Tensor:
    """Clipped value loss (detail 9): ``0.5 * max((V-R)^2, (V_clip-R)^2)``, meaned.

    The clipped branch is ``V_old + clip(V_new - V_old, -eps, +eps)`` — the
    value prediction may move at most ``eps`` from its rollout-time estimate.

    Args:
        new_values: Current value predictions, ``(batch,)``.
        old_values: Value estimates recorded at rollout time, ``(batch,)``.
        returns: Value targets (advantages + old values), ``(batch,)``.
        clip_coef: Clipping coefficient epsilon (shared with the policy).

    Returns:
        The scalar loss (with gradients).
    """
    v_loss_unclipped = (new_values - returns) ** 2
    v_clipped = old_values + torch.clamp(new_values - old_values, -clip_coef, clip_coef)
    v_loss_clipped = (v_clipped - returns) ** 2
    return 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()


def explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    """``1 - Var(R - V) / Var(R)``; NaN when the returns have zero variance.

    How much of the return variance the value function accounts for: 1 is a
    perfect critic, 0 is no better than predicting the mean, below 0 is
    worse than useless (detail 12).

    Args:
        values: Value predictions over the whole batch, ``(batch,)``.
        returns: Value targets over the whole batch, ``(batch,)``.

    Returns:
        The explained variance as a float.
    """
    var_returns = float(np.var(returns))
    if var_returns == 0:
        return float("nan")
    return 1.0 - float(np.var(returns - values)) / var_returns


@dataclass(frozen=True)
class PpoSummary:
    """What a PPO run produced, for the console summary and for tests.

    Attributes:
        episodic_returns: Return of every completed episode, in order.
        episodic_lengths: Length of every completed episode, in order.
        episode_end_steps: ``global_step`` at which each episode finished.
        steps: Environment steps actually taken (``num_iterations * batch_size``).
        sps: Average environment steps per second.
        wandb_url: The W&B run URL when tracking online, else None.
        episodes_csv: Path of the saved episode log, when enabled.
    """

    episodic_returns: list[float] = field(default_factory=list)
    episodic_lengths: list[int] = field(default_factory=list)
    episode_end_steps: list[int] = field(default_factory=list)
    steps: int = 0
    sps: float = 0.0
    wandb_url: str | None = None
    episodes_csv: str | None = None

    def render(self) -> str:
        """Format the end-of-run console summary."""
        lines = [f"ppo finished: {self.steps} steps, {len(self.episodic_returns)} episodes"]
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


def run_ppo(config: PpoConfig) -> PpoSummary:
    """Train PPO top to bottom; the structure mirrors CleanRL's script.

    Args:
        config: The experiment configuration.

    Returns:
        A summary of the run.

    Raises:
        ValueError: If the action space is not ``Discrete``, or the config's
            batch geometry doesn't divide evenly.
    """
    if config.batch_size % config.num_minibatches != 0:
        raise ValueError(
            f"batch_size {config.batch_size} must divide evenly into "
            f"num_minibatches {config.num_minibatches}."
        )
    if config.num_iterations == 0:
        raise ValueError(
            f"total_timesteps {config.total_timesteps} is smaller than one "
            f"batch ({config.batch_size}); nothing to run."
        )
    seed_everything(config.seed)
    device = resolve_device(config.device)

    logger = RunLogger(config, resolved_device=str(device))
    logger.start()

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                config.env_id,
                seed=config.seed,
                idx=idx,
                capture_video=config.capture_video,
                video_dir=f"{config.video_dir}/{config.run_name}",
            )
            for idx in range(config.num_envs)
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        raise ValueError(
            f"This PPO is discrete-action; got {envs.single_action_space}. "
            "The continuous variant is a separate lifecycle (issue #4)."
        )
    assert envs.single_observation_space.shape is not None
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    n_actions = int(envs.single_action_space.n)

    agent = Agent(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    obs = torch.zeros(
        (config.num_steps, config.num_envs, *envs.single_observation_space.shape), device=device
    )
    actions = torch.zeros((config.num_steps, config.num_envs), device=device)
    logprobs = torch.zeros((config.num_steps, config.num_envs), device=device)
    rewards = torch.zeros((config.num_steps, config.num_envs), device=device)
    dones = torch.zeros((config.num_steps, config.num_envs), device=device)
    values = torch.zeros((config.num_steps, config.num_envs), device=device)

    returns_log: list[float] = []
    lengths_log: list[int] = []
    end_steps_log: list[int] = []
    global_step = 0
    start = time.perf_counter()

    # Plain reset: the factory already seeded each sub-env's RNG stream.
    # (SyncVectorEnv's generics erase to Any — annotate what it hands back.)
    next_obs_np: np.ndarray
    reward: np.ndarray
    terminations: np.ndarray
    truncations: np.ndarray
    next_obs_np, _ = envs.reset()
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(config.num_envs, device=device)

    for iteration in range(1, config.num_iterations + 1):
        if config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / config.num_iterations
            optimizer.param_groups[0]["lr"] = frac * config.learning_rate

        for step in range(config.num_steps):
            global_step += config.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # SAME_STEP autoreset: next_obs is already the reset observation
            # where done; truncation folds into done (reference parity — no
            # bootstrap through time limits, detail 17).
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(next_done_np, dtype=torch.float32, device=device)

            if "final_info" in infos:
                final_info = infos["final_info"]
                episode = final_info["episode"]
                for env_returns, env_length in zip(
                    episode["r"][final_info["_episode"]],
                    episode["l"][final_info["_episode"]],
                    strict=True,
                ):
                    returns_log.append(float(env_returns))
                    lengths_log.append(int(env_length))
                    end_steps_log.append(global_step)
                    logger.log(
                        {
                            metrics.EPISODIC_RETURN: returns_log[-1],
                            metrics.EPISODIC_LENGTH: lengths_log[-1],
                        },
                        step=global_step,
                    )

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages, returns = compute_gae(
                rewards,
                values,
                dones,
                next_value.flatten(),
                next_done,
                config.gamma,
                config.gae_lambda,
            )

        b_obs = obs.reshape((-1, *envs.single_observation_space.shape))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(config.batch_size)
        clipfracs: list[float] = []
        for _epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for mb_start in range(0, config.batch_size, config.minibatch_size):
                mb_inds = b_inds[mb_start : mb_start + config.minibatch_size]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                mb_advantages = b_advantages[mb_inds]
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                pg_loss, approx_kl, old_approx_kl, clipfrac = clipped_policy_loss(
                    newlogprob, b_logprobs[mb_inds], mb_advantages, config.clip_coef
                )
                clipfracs.append(clipfrac.item())

                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    v_loss = clipped_value_loss(
                        newvalue, b_values[mb_inds], b_returns[mb_inds], config.clip_coef
                    )
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

            if config.target_kl is not None and approx_kl.item() > config.target_kl:
                break

        explained_var = explained_variance(b_values.cpu().numpy(), b_returns.cpu().numpy())

        logger.log(
            {
                metrics.LEARNING_RATE: optimizer.param_groups[0]["lr"],
                metrics.VALUE_LOSS: v_loss.item(),
                metrics.POLICY_LOSS: pg_loss.item(),
                metrics.ENTROPY: entropy_loss.item(),
                metrics.OLD_APPROX_KL: old_approx_kl.item(),
                metrics.APPROX_KL: approx_kl.item(),
                metrics.CLIPFRAC: float(np.mean(clipfracs)),
                metrics.EXPLAINED_VARIANCE: explained_var,
                metrics.SPS: global_step / (time.perf_counter() - start),
            },
            step=global_step,
        )

    elapsed = time.perf_counter() - start
    episodes_csv = (
        _save_episode_log(config, returns_log, end_steps_log) if config.save_episodes else None
    )
    summary = PpoSummary(
        episodic_returns=returns_log,
        episodic_lengths=lengths_log,
        episode_end_steps=end_steps_log,
        steps=global_step,
        sps=global_step / elapsed,
        wandb_url=logger.url,
        episodes_csv=episodes_csv,
    )
    logger.finish()
    envs.close()
    return summary


def _save_episode_log(config: PpoConfig, returns: list[float], end_steps: list[int]) -> str:
    """Write per-episode curves in the benchmark compare input format.

    Not the SAC/demo helper duplicated a third time: with vector envs the
    episode-end step is the recorded ``global_step``, not a cumulative sum
    of episode lengths (episodes interleave across envs). Promotion to
    shared core per ADR 0003 waits until the signatures actually converge.
    """
    path = Path(config.episode_dir) / f"{config.run_name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "global_step", "episodic_return"])
        for step, episodic_return in zip(end_steps, returns, strict=True):
            writer.writerow([config.run_name, step, episodic_return])
    return str(path)
