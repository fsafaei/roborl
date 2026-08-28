"""PPO for continuous actions — one training loop, top to bottom.

Diffable against CleanRL's ``ppo_continuous_action.py`` (the MuJoCo
variant); the continuous-actions section of ``docs/algos/ppo.md`` lists the
9 continuous-control implementation details on top of the 13 core ones and
every deliberate deviation from the reference. The clipped-objective math
(GAE, policy/value losses, explained variance) is identical to the discrete
variant and imported from the sibling module — same package, same
hand-computed fixtures. What is new here: the diagonal-Gaussian policy with
a state-independent log std, and the environment normalization stack
(observation/reward running normalization + clipping, action clipping).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal

from roborl.algos.ppo.ppo import (
    PpoConfig,
    PpoSummary,
    _save_episode_log,
    clipped_policy_loss,
    clipped_value_loss,
    compute_gae,
    explained_variance,
    layer_init,
)
from roborl.telemetry import metrics
from roborl.telemetry.logger import RunLogger
from roborl.utils.device import resolve_device
from roborl.utils.seeding import seed_everything


@dataclass(frozen=True)
class PpoContinuousConfig(PpoConfig):
    """PPO continuous hyperparameters — CleanRL's defaults exactly.

    Same schema as the discrete variant (the batch geometry properties come
    with it); only the defaults differ, matching
    ``ppo_continuous_action.py``: one env with long 2048-step rollouts, 32
    minibatches x 10 epochs per batch, lr 3e-4, and no entropy bonus.
    """

    exp_name: str = "ppo_continuous_action"
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 2048
    num_minibatches: int = 32
    update_epochs: int = 10
    ent_coef: float = 0.0


def make_continuous_env(
    env_id: str,
    seed: int,
    gamma: float,
    idx: int = 0,
    capture_video: bool = False,
    video_dir: str = "videos",
) -> Callable[[], gym.Env]:
    """Build a thunk creating one env with PPO's continuous-control wrappers.

    Local to this algorithm rather than in ``roborl.envs.factory`` (ADR
    0003): the normalization stack is a PPO-continuous implementation detail
    (details C5-C9 in the spec note), not shared infrastructure — SAC runs
    on raw observations and rewards. Wrapper order mirrors CleanRL exactly;
    ``RecordEpisodeStatistics`` sits *below* the reward normalization so
    ``charts/episodic_return`` stays in raw env units. Seeding follows the
    factory's convention: per-env ``seed + idx`` RNG streams via an initial
    ``reset(seed=...)``, action and observation spaces seeded likewise.

    Args:
        env_id: Gymnasium environment id, e.g. ``"Pendulum-v1"``.
        seed: Base seed; the env uses ``seed + idx``.
        gamma: Discount factor, needed by ``NormalizeReward``'s running
            discounted-return statistics.
        idx: Index of this env within a vector env (0 for a single env).
        capture_video: Record periodic episode videos (env 0 only).
        video_dir: Directory the video files are written to.

    Returns:
        A thunk returning the wrapped, seeded environment.
    """

    def thunk() -> gym.Env:
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, video_folder=video_dir)
        else:
            env = gym.make(env_id)
        if not isinstance(env.action_space, gym.spaces.Box):
            # Checked here, before ClipAction's own assert, for a clear error.
            raise ValueError(
                f"This PPO is continuous-action; got {env.action_space}. "
                "The discrete variant lives in roborl.algos.ppo.ppo."
            )
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10, 10), env.observation_space
        )
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(
            env, lambda reward: float(np.clip(float(reward), -10.0, 10.0))
        )
        env.reset(seed=seed + idx)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


class Agent(nn.Module):
    """Separate 64-64 tanh mean and value MLPs plus a state-independent log std.

    Details C1-C4: actions come from a diagonal Gaussian whose mean is the
    actor MLP's output (head gain 0.01) and whose log std is a free
    parameter initialized to zeros (sigma = 1 everywhere at init),
    independent of the observation.
    """

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        """Build both MLPs for a flat observation size and action dimension."""
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Return state values with shape ``(batch, 1)``."""
        value: torch.Tensor = self.critic(x)
        return value

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or score a given) action under the diagonal Gaussian policy.

        Log-prob and entropy are summed over the action dimensions —
        independent action components (detail C3). The stored (unclipped)
        rollout action is what gets scored during updates; clipping to the
        action-space bounds is the ``ClipAction`` wrapper's job (detail C5).

        Args:
            x: Observations, shape ``(batch, obs_dim)``.
            action: When given, score these actions instead of sampling.

        Returns:
            ``(action, log_prob, entropy, value)`` with shapes
            ``(batch, action_dim)``, ``(batch,)``, ``(batch,)``, ``(batch, 1)``.
        """
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        probs = Normal(action_mean, torch.exp(action_logstd))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def run_ppo_continuous(config: PpoContinuousConfig) -> PpoSummary:
    """Train continuous PPO top to bottom; mirrors CleanRL's script structure.

    Args:
        config: The experiment configuration.

    Returns:
        A summary of the run.

    Raises:
        ValueError: If the action space is not ``Box``, or the config's
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
            make_continuous_env(
                config.env_id,
                seed=config.seed,
                gamma=config.gamma,
                idx=idx,
                capture_video=config.capture_video,
                video_dir=f"{config.video_dir}/{config.run_name}",
            )
            for idx in range(config.num_envs)
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise ValueError(
            f"This PPO is continuous-action; got {envs.single_action_space}. "
            "The discrete variant lives in roborl.algos.ppo.ppo."
        )
    assert envs.single_observation_space.shape is not None
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_shape = envs.single_action_space.shape
    action_dim = int(np.prod(action_shape))

    agent = Agent(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    obs = torch.zeros(
        (config.num_steps, config.num_envs, *envs.single_observation_space.shape), device=device
    )
    actions = torch.zeros((config.num_steps, config.num_envs, *action_shape), device=device)
    logprobs = torch.zeros((config.num_steps, config.num_envs), device=device)
    rewards = torch.zeros((config.num_steps, config.num_envs), device=device)
    dones = torch.zeros((config.num_steps, config.num_envs), device=device)
    values = torch.zeros((config.num_steps, config.num_envs), device=device)

    returns_log: list[float] = []
    lengths_log: list[int] = []
    end_steps_log: list[int] = []
    global_step = 0
    start = time.perf_counter()

    # Plain reset: the thunk already seeded each sub-env's RNG stream.
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
        b_actions = actions.reshape((-1, *action_shape))
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
                    b_obs[mb_inds], b_actions[mb_inds]
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
