"""ExperimentConfig: run-name shape, grouping, dict round-trip, entity resolution."""

import re

import pytest

from roborl.config import ExperimentConfig


@pytest.mark.unit
def test_run_name_shape_and_stability() -> None:
    config = ExperimentConfig(exp_name="demo", env_id="CartPole-v1", seed=7)
    assert re.fullmatch(r"demo-CartPole-v1-s7-\d{8}T\d{6}", config.run_name)
    assert config.run_name == config.run_name  # cached, stable per instance


@pytest.mark.unit
def test_group_is_exp_and_env() -> None:
    config = ExperimentConfig(exp_name="demo", env_id="Acrobot-v1")
    assert config.group == "demo-Acrobot-v1"


@pytest.mark.unit
def test_to_dict_round_trip() -> None:
    config = ExperimentConfig(exp_name="demo", seed=3, total_timesteps=100)
    d = config.to_dict()
    rebuilt = ExperimentConfig(**{k: v for k, v in d.items() if k != "run_name"})
    assert rebuilt == config
    assert d["seed"] == 3 and d["total_timesteps"] == 100


@pytest.mark.unit
def test_wandb_entity_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    assert ExperimentConfig().resolved_wandb_entity is None
    monkeypatch.setenv("WANDB_ENTITY", "some-team")
    assert ExperimentConfig().resolved_wandb_entity == "some-team"
    assert ExperimentConfig(wandb_entity="explicit").resolved_wandb_entity == "explicit"
