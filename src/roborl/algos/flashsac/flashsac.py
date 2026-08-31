"""FlashSAC (Kim et al., RSS 2026) — one training loop, top to bottom.

SAC plus six stability changes; the spec note in ``docs/algos/flashsac.md``
lists the update equations, the pitfall catalogue, and every judgement call
awaiting the Pass B diff against the reference implementation. Structure
mirrors ``sac.py``: a single non-autoresetting env with an explicit
``reset()`` keeps one stored transition per step and the true final
observation at episode end. The math lives in pure, fixture-tested modules
(``distrib``, ``rewards``, ``noise``); this file is the wiring, and the
wiring discipline is BatchNorm mode and gradient isolation:

- rollout and evaluation forward passes always run with ``training=False``;
- the critic loss and TD target each use a *single* cross-batch train-mode
  forward over ``cat([s, s'])``, then chunk;
- the critic inside the actor loss runs in eval mode with its *parameters*
  frozen (gradient flows through the action only) — deliberate asymmetry;
- ``alpha`` is read inside ``no_grad`` in the target; the temperature loss
  sees a detached entropy;
- every optimiser step is followed by its LR-schedule step and (actor /
  critic) ``normalize_parameters()``;
- the target EMA covers parameters only — the target critic's BatchNorm
  statistics are its own, advanced by its own train-mode forwards.
"""

from __future__ import annotations

import copy
import csv
import itertools
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import optim

from roborl.algos.flashsac.buffer import ReplayBuffer
from roborl.algos.flashsac.distrib import categorical_td_target, select_min_member
from roborl.algos.flashsac.networks import (
    FlashSACActor,
    FlashSACDoubleCritic,
    Temperature,
    entropy_target,
)
from roborl.algos.flashsac.noise import NoiseRepeater
from roborl.algos.flashsac.rewards import RewardNormalizer
from roborl.config import ExperimentConfig
from roborl.envs.factory import make_env
from roborl.telemetry import metrics
from roborl.telemetry.logger import RunLogger
from roborl.utils.device import resolve_device
from roborl.utils.seeding import seed_everything


@dataclass(frozen=True)
class FlashSacConfig(ExperimentConfig):
    """FlashSAC hyperparameters — the authors' CPU/MuJoCo recipe (docs/algos/flashsac.md).

    Verification runs must not override any algorithm hyperparameter; any
    deviation gets listed in the benchmark report. Training defaults to
    ``track=True``: a run that isn't recorded didn't happen.
    """

    exp_name: str = "flashsac"
    env_id: str = "Pendulum-v1"
    total_timesteps: int = 1_000_000
    track: bool = True
    buffer_size: int = 1_000_000
    """Replay buffer capacity in transitions."""
    gamma: float = 0.99
    """Discount factor."""
    tau: float = 0.01
    """Target EMA weight on the online network (twice SAC's 0.005)."""
    batch_size: int = 512
    """Minibatch size sampled from the replay buffer."""
    learning_starts: int = 10_000
    """Uniform-random action steps before any gradient update."""
    lr_init: float = 3e-4
    """Learning rate at update 0 (== peak: the warmup is effectively absent)."""
    lr_peak: float = 3e-4
    """Learning rate at the end of warmup, start of the cosine decay."""
    lr_end: float = 1.5e-4
    """Learning rate at the end of training."""
    warmup_rate: float = 1e-6
    """Warmup length as a fraction of the optimiser's total updates."""
    actor_update_period: int = 2
    """Actor + temperature update every N critic updates — a plain skip, not compensated."""
    actor_hidden: int = 128
    """Actor trunk width."""
    actor_blocks: int = 2
    """Actor residual block count."""
    critic_hidden: int = 256
    """Critic trunk width."""
    critic_blocks: int = 2
    """Critic residual block count."""
    n_atoms: int = 101
    """Categorical support size."""
    v_min: float = -5.0
    """Lowest support atom."""
    v_max: float = 5.0
    """Highest support atom."""
    g_max: float = 5.0
    """Reward-scaling bound: discounted returns are kept inside [-g_max, g_max]."""
    alpha_init: float = 0.01
    """Initial entropy temperature (not SAC's 1.0)."""
    sigma_tgt: float = 0.15
    """Target per-dimension action std defining the entropy target."""
    noise_zeta_mu: float = 2.0
    """Zeta exponent for exploration-noise run lengths."""
    noise_zeta_max: int = 16
    """Zeta truncation: maximum noise run length."""
    use_rmsnorm: bool = True
    """Terminal RMSNorm on both trunks. Off only on ablation-ladder rung 2."""
    use_distributional: bool = True
    """Categorical critic + adaptive reward scaling. Off on ladder rungs 2-3:
    scalar critic, MSE on raw rewards."""
    use_weight_norm: bool = True
    """Unit weight normalisation at init and after every optimiser step. Off on rungs 2-4."""
    use_flash_exploration: bool = True
    """Sigma-based entropy target + Zeta-repeated noise. Off on rungs 2-5, which use
    SAC's -dim(A) target and fresh per-step noise — pass --alpha-init 1.0 alongside."""
    save_episodes: bool = False
    """Write per-episode returns to runs/{run_name}.csv for benchmark compare."""
    episode_dir: str = "runs"
    """Directory episode CSVs are written to when save_episodes is on."""


def cosine_lr(
    update_step: int,
    total_updates: int,
    init: float,
    peak: float,
    end: float,
    warmup_rate: float,
) -> float:
    """Linear warmup then cosine decay, per *gradient update* of one optimiser.

    ``total_updates`` is the SHARED horizon — total env steps times the
    update-to-data ratio — for all three optimisers, per the reference.
    Each optimiser advances its own ``update_step`` through it, so the
    actor and temperature (stepping every ``actor_update_period``-th
    update) end mid-cosine rather than at ``end``.

    Args:
        update_step: Completed updates of this optimiser (0-based).
        total_updates: Total updates this optimiser will take.
        init: Learning rate at step 0.
        peak: Learning rate at the end of warmup.
        end: Learning rate at the end of the budget.
        warmup_rate: Warmup length as a fraction of ``total_updates``.

    Returns:
        The learning rate for this update.
    """
    warmup = int(warmup_rate * total_updates)
    if update_step < warmup:
        return init + (peak - init) * update_step / warmup
    progress = (update_step - warmup) / max(total_updates - warmup, 1)
    progress = min(progress, 1.0)
    return end + (peak - end) * 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass(frozen=True)
class FlashSacSummary:
    """What a FlashSAC run produced, for the console summary and for tests.

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
        lines = [f"flashsac finished: {self.steps} steps, {len(self.episodic_returns)} episodes"]
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


def run_flashsac(config: FlashSacConfig) -> FlashSacSummary:
    """Train FlashSAC top to bottom.

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

    env: gym.Env = make_env(
        config.env_id,
        seed=config.seed,
        capture_video=config.capture_video,
        video_dir=f"{config.video_dir}/{config.run_name}",
    )()
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError(f"FlashSAC needs a continuous (Box) action space; got {env.action_space}.")
    # The actor's log-prob has no action_scale correction, so actions must
    # live in [-1, 1]; RescaleAction maps the env's true bounds onto that.
    env = gym.wrappers.RescaleAction(env, min_action=np.float32(-1.0), max_action=np.float32(1.0))
    env.action_space.seed(config.seed)
    assert isinstance(env.action_space, gym.spaces.Box)
    assert np.all(env.action_space.low == -1.0) and np.all(env.action_space.high == 1.0)
    assert env.observation_space.shape is not None
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    actor = FlashSACActor(
        obs_dim,
        act_dim,
        config.actor_hidden,
        config.actor_blocks,
        use_rmsnorm=config.use_rmsnorm,
    ).to(device)
    critic = FlashSACDoubleCritic(
        obs_dim,
        act_dim,
        hidden=config.critic_hidden,
        num_blocks=config.critic_blocks,
        n_atoms=config.n_atoms,
        v_min=config.v_min,
        v_max=config.v_max,
        use_rmsnorm=config.use_rmsnorm,
        distributional=config.use_distributional,
    ).to(device)
    # Full copy at construction only; from here the EMA touches parameters
    # exclusively and the target's BatchNorm stats evolve on their own.
    target_critic = copy.deepcopy(critic)
    if config.use_weight_norm:
        # The reference normalizes at init too: orthogonal init only gives unit
        # rows where out_features <= in_features, so wide layers start off-manifold.
        actor.normalize_parameters()
        critic.normalize_parameters()
        target_critic.normalize_parameters()
    temperature = Temperature(config.alpha_init).to(device)
    h_target = (
        entropy_target(act_dim, config.sigma_tgt)
        if config.use_flash_exploration
        else -float(act_dim)
    )
    bin_values = critic.bin_values.view(-1) if config.use_distributional else None

    critic_opt = optim.Adam(critic.parameters(), lr=config.lr_peak)
    actor_opt = optim.Adam(actor.parameters(), lr=config.lr_peak)
    temp_opt = optim.Adam(temperature.parameters(), lr=config.lr_peak)
    # ONE shared schedule horizon (total env steps x UTD), per the reference;
    # each optimiser advances its own step counter through it. The actor and
    # temperature step every actor_update_period-th update, so they end
    # mid-cosine (~2.25e-4 at the defaults) — that is the reference's
    # behaviour, not a bug (Pass B diff, docs/algos/flashsac.md).
    schedule_total = config.total_timesteps

    rb = ReplayBuffer(config.buffer_size, env.observation_space.shape, env.action_space.shape)
    reward_normalizer = RewardNormalizer(config.gamma, num_envs=1, g_max=config.g_max)
    noise = NoiseRepeater(1, act_dim, mu=config.noise_zeta_mu, k_max=config.noise_zeta_max)

    returns: list[float] = []
    lengths: list[int] = []
    update_step = 0
    actor_step = 0
    actor_loss_val = float("nan")
    alpha_loss_val = float("nan")
    start = time.perf_counter()

    obs, _ = env.reset()
    obs = obs.astype(np.float32)
    for global_step in range(config.total_timesteps):
        if global_step < config.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():  # BN-eval on every rollout path
                obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
                mean, std = actor.get_mean_and_std(obs_t, training=False)
                if config.use_flash_exploration:
                    eps = noise.next().to(device)
                else:  # SAC-style: a fresh draw every step (== rsample)
                    eps = torch.randn_like(mean)
                action_t = torch.tanh(mean + std * eps)
            action = action_t.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = next_obs.astype(np.float32)
        # Raw reward into the buffer; terminated only (truncation bootstraps).
        rb.add(obs, action, float(reward), next_obs, terminated)
        # The reward statistics run on the collected stream, and their
        # accumulator resets on BOTH flags — unlike the TD target.
        reward_normalizer.update(float(reward), terminated, truncated)

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
            obs_all = torch.cat([data.observations, data.next_observations], dim=0)

            # --- actor + temperature FIRST, every actor_update_period-th
            # update; the critic update below then sees the updated actor
            # and temperature, exactly as the reference orders it ---
            if update_step % config.actor_update_period == 0:
                # Cross-batch so the actor's BatchNorm sees the same
                # mixture of current and next observations.
                a_all, logp_all_pi = actor(obs_all, training=True)
                a_pi = a_all.chunk(2, dim=0)[0]
                logp = logp_all_pi.chunk(2, dim=0)[0]
                # Freeze critic PARAMETERS (not no_grad): the actor gradient
                # must flow through a_pi into the critic. BN-eval here is the
                # deliberate asymmetry against the critic loss below.
                critic.requires_grad_(False)
                q_pi, _ = critic(data.observations, a_pi, training=False)
                critic.requires_grad_(True)
                actor_loss = (temperature().detach() * logp - q_pi.min(dim=0).values).mean()

                actor_lr = cosine_lr(
                    actor_step,
                    schedule_total,
                    config.lr_init,
                    config.lr_peak,
                    config.lr_end,
                    config.warmup_rate,
                )
                for group in actor_opt.param_groups:
                    group["lr"] = actor_lr
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()
                if config.use_weight_norm:
                    actor.normalize_parameters()
                actor_loss_val = actor_loss.item()

                entropy = -logp.mean().detach()
                alpha_loss = temperature() * (entropy - h_target)
                for group in temp_opt.param_groups:
                    group["lr"] = actor_lr
                temp_opt.zero_grad()
                alpha_loss.backward()
                temp_opt.step()
                alpha_loss_val = alpha_loss.item()
                actor_step += 1

            # --- TD target, everything under no_grad ---
            with torch.no_grad():
                a_next, logp_next = actor(data.next_observations, training=False)
                alpha = temperature()  # read INSIDE no_grad: pitfall 7
                # Cross-batch: ONE train-mode pass over both halves so
                # Q(s,a) and Q(s',a') share normalisation statistics.
                act_all = torch.cat([data.actions, a_next], dim=0)
                q_all, logp_all = target_critic(obs_all, act_all, training=True)
                q_next = q_all.chunk(2, dim=1)[1]
                if config.use_distributional:
                    assert bin_values is not None and logp_all is not None
                    r_norm = reward_normalizer.normalize(data.rewards)
                    ent_term = alpha * logp_next
                    logp_next_d = logp_all.chunk(2, dim=1)[1]
                    log_p = select_min_member(q_next, logp_next_d)
                    m, clamp_fraction = categorical_td_target(
                        log_p, r_norm, data.dones, ent_term, bin_values, config.gamma
                    )
                else:
                    # Ladder rungs 2-3: scalar clipped double-Q soft target
                    # on RAW rewards (no fixed support, no reward scaling).
                    min_q_next = q_next.min(dim=0).values
                    soft_value = min_q_next - alpha * logp_next
                    y = data.rewards + config.gamma * (1.0 - data.dones) * soft_value
                    assert y.shape == (config.batch_size,)

            # --- critic loss on the same concatenated batch ---
            q_pred_all, logp_pred_all = critic(obs_all, act_all, training=True)
            if config.use_distributional:
                assert logp_pred_all is not None
                log_p_sa = logp_pred_all.chunk(2, dim=1)[0]
                ce = -(m.unsqueeze(0) * log_p_sa).sum(dim=-1)
                assert ce.shape == (critic.ensemble_size, config.batch_size)
                critic_loss = ce.mean()
            else:
                q_sa = q_pred_all.chunk(2, dim=1)[0]
                critic_loss = ((q_sa - y.unsqueeze(0)) ** 2).mean()

            lr_now = cosine_lr(
                update_step,
                schedule_total,
                config.lr_init,
                config.lr_peak,
                config.lr_end,
                config.warmup_rate,
            )
            for group in critic_opt.param_groups:
                group["lr"] = lr_now
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
            if config.use_weight_norm:
                critic.normalize_parameters()

            # --- target EMA: parameters ONLY, never BatchNorm buffers ---
            with torch.no_grad():
                for p_t, p in zip(target_critic.parameters(), critic.parameters(), strict=True):
                    p_t.lerp_(p, config.tau)

            if update_step % 100 == 0:
                with torch.no_grad():
                    q_current = q_pred_all.chunk(2, dim=1)[0]
                    feature_norm = (
                        critic.features(data.observations, data.actions, training=False)
                        .norm(dim=-1)
                        .mean()
                    )
                    grad_norm = torch.norm(
                        torch.stack(
                            [p.grad.norm() for p in critic.parameters() if p.grad is not None]
                        )
                    )
                    param_norm = torch.norm(torch.stack([p.norm() for p in critic.parameters()]))
                scalars = {
                    metrics.QF1_VALUES: q_current[0].mean().item(),
                    metrics.QF2_VALUES: q_current[1].mean().item(),
                    metrics.QF_LOSS: critic_loss.item(),
                    metrics.ACTOR_LOSS: actor_loss_val,
                    metrics.ALPHA: alpha.item(),
                    metrics.ALPHA_LOSS: alpha_loss_val,
                    metrics.LEARNING_RATE: lr_now,
                    metrics.SPS: global_step / (time.perf_counter() - start),
                    metrics.CRITIC_FEATURE_NORM: feature_norm.item(),
                    metrics.GRAD_NORM: grad_norm.item(),
                    metrics.PARAM_NORM: param_norm.item(),
                    metrics.TARGET_ENTROPY: h_target,
                }
                if config.use_distributional:
                    with torch.no_grad():
                        target_entropy_dist = -(m * (m + 1e-12).log()).sum(dim=-1).mean()
                    scalars[metrics.TARGET_CLAMP_FRACTION] = clamp_fraction.item()
                    scalars[metrics.REWARD_SCALE] = reward_normalizer.denominator
                    scalars[metrics.RETURN_RMS_VAR] = reward_normalizer.rms.var
                    scalars[metrics.TARGET_DIST_ENTROPY] = target_entropy_dist.item()
                if config.use_flash_exploration:
                    scalars[metrics.NOISE_REPEAT_LEN] = float(noise.run_length)
                logger.log(scalars, step=global_step)
            update_step += 1

    elapsed = time.perf_counter() - start
    episodes_csv = _save_episode_log(config, returns, lengths) if config.save_episodes else None
    summary = FlashSacSummary(
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


def _save_episode_log(config: FlashSacConfig, returns: list[float], lengths: list[int]) -> str:
    """Write per-episode curves in the benchmark compare input format.

    Third local copy (demo, sac, here) — promotion to shared core is due
    under the rule of three and happens in its own PR, per ADR 0003.
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
