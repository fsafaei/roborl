"""Random-agent pipeline check — the one runnable "hello world".

This is deliberately the only "agent" in the repository: it proves the whole
stack (config → seeding → device → env factory → telemetry → summary) works
end to end without any learning algorithm, and it is the template every
future training script follows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from roborl.config import ExperimentConfig
from roborl.envs.factory import make_env
from roborl.telemetry import metrics
from roborl.telemetry.logger import RunLogger
from roborl.utils.device import resolve_device
from roborl.utils.seeding import seed_everything


@dataclass(frozen=True)
class DemoConfig(ExperimentConfig):
    """Configuration for the random-agent demo.

    Defaults to ``track=False`` so first contact needs no W&B account; pass
    ``--track`` to see the run in W&B. Future training scripts will default
    to tracking instead.
    """

    exp_name: str = "demo"
    env_id: str = "CartPole-v1"
    total_timesteps: int = 5_000


@dataclass(frozen=True)
class DemoSummary:
    """What a demo run produced, for the console summary and for tests.

    Attributes:
        episodic_returns: Return of every completed episode, in order.
        episodic_lengths: Length of every completed episode, in order.
        steps: Environment steps actually taken.
        sps: Average environment steps per second.
        wandb_url: The W&B run URL when tracking online, else None.
    """

    episodic_returns: list[float] = field(default_factory=list)
    episodic_lengths: list[int] = field(default_factory=list)
    steps: int = 0
    sps: float = 0.0
    wandb_url: str | None = None

    def render(self) -> str:
        """Format the end-of-run console summary."""
        lines = [f"demo finished: {self.steps} steps, {len(self.episodic_returns)} episodes"]
        if self.episodic_returns:
            mean = float(np.mean(self.episodic_returns))
            std = float(np.std(self.episodic_returns))
            lines.append(f"episodic return: {mean:.1f} ± {std:.1f}")
        lines.append(f"throughput: {self.sps:.0f} steps/s")
        lines.append(
            f"tracked at: {self.wandb_url}"
            if self.wandb_url
            else "telemetry: disabled (pass --track to log to W&B)"
        )
        return "\n".join(lines)


def run_demo(config: DemoConfig) -> DemoSummary:
    """Run a random agent through the full pipeline.

    Args:
        config: The demo configuration.

    Returns:
        A summary of the run.
    """
    seed_everything(config.seed)
    device = resolve_device(config.device)  # unused by the random agent; proves resolution

    logger = RunLogger(config, resolved_device=str(device))
    logger.start()

    video_dir = f"{config.video_dir}/{config.run_name}"
    env = make_env(
        config.env_id,
        seed=config.seed,
        capture_video=config.capture_video,
        video_dir=video_dir,
    )()

    returns: list[float] = []
    lengths: list[int] = []
    start = time.perf_counter()
    env.reset()
    for global_step in range(1, config.total_timesteps + 1):
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            episode = info["episode"]
            returns.append(float(episode["r"]))
            lengths.append(int(episode["l"]))
            sps = global_step / (time.perf_counter() - start)
            logger.log(
                {
                    metrics.EPISODIC_RETURN: returns[-1],
                    metrics.EPISODIC_LENGTH: lengths[-1],
                    metrics.SPS: sps,
                },
                step=global_step,
            )
            env.reset()

    elapsed = time.perf_counter() - start
    summary = DemoSummary(
        episodic_returns=returns,
        episodic_lengths=lengths,
        steps=config.total_timesteps,
        sps=config.total_timesteps / elapsed,
        wandb_url=logger.url,
    )
    logger.finish()
    env.close()
    return summary
