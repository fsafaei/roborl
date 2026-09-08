"""roborl.io: spec validation, save/load round trip per actor, integrity and safety checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from roborl.io import (
    FORMAT,
    ObsNormalizer,
    PolicySpec,
    build_actor,
    load_policy,
    save_policy,
    sha256_of,
)


class _ArbitraryObject:
    """Arbitrary pickled code: ``weights_only`` loading must refuse it."""


SPECS: dict[str, PolicySpec] = {
    "sac": PolicySpec("sac", 5, 2, (-2.0, -1.0), (2.0, 1.0)),
    "flashsac": PolicySpec(
        "flashsac",
        5,
        2,
        (-2.0, -1.0),
        (2.0, 1.0),
        {"hidden": 16, "num_blocks": 1, "use_rmsnorm": True},
    ),
    "ppo_continuous": PolicySpec("ppo_continuous", 5, 2, (-2.0, -1.0), (2.0, 1.0)),
}


def _normalizer(spec: PolicySpec) -> ObsNormalizer | None:
    if spec.algo != "ppo_continuous":
        return None
    return ObsNormalizer(
        mean=np.arange(spec.obs_dim) * 0.1, var=np.ones(spec.obs_dim) * 2.0, epsilon=1e-8, clip=10.0
    )


def _reference_action(
    spec: PolicySpec, actor: Any, obs: np.ndarray, normalizer: ObsNormalizer | None
) -> np.ndarray:
    """What each training loop's own evaluation path computes, written out independently."""
    x = obs if normalizer is None else normalizer(obs)
    t = torch.as_tensor(x, dtype=torch.float32)
    low, high = np.array(spec.action_low), np.array(spec.action_high)
    with torch.no_grad():
        if spec.algo == "sac":
            out: np.ndarray = actor.get_action(t)[2].numpy()
            return out
        if spec.algo == "flashsac":
            unit: np.ndarray = actor.eval_action(t).numpy()
            rescaled: np.ndarray = low + (unit + 1.0) / 2.0 * (high - low)
            return rescaled
        mean: np.ndarray = actor.actor_mean(t).numpy()
        clipped: np.ndarray = np.clip(mean, low, high)
        return clipped


def _trained_looking_actor(spec: PolicySpec) -> torch.nn.Module:
    """A randomly initialised actor whose buffers are not at their defaults."""
    torch.manual_seed(0)
    actor = build_actor(spec)
    if spec.algo == "flashsac":
        with torch.no_grad():  # a training-mode pass moves the BatchNorm running stats
            actor(torch.randn(32, spec.obs_dim) * 3 + 1, training=True)
    return actor


@pytest.mark.unit
@pytest.mark.parametrize("algo", sorted(SPECS))
def test_round_trip_reproduces_the_actor(tmp_path: Path, algo: str) -> None:
    spec = SPECS[algo]
    actor = _trained_looking_actor(spec)
    normalizer = _normalizer(spec)
    path = tmp_path / f"{algo}.pt"
    sha = save_policy(
        path, actor, spec, metadata={"note": "test", "seed": 1}, obs_normalizer=normalizer
    )
    assert sha == sha256_of(path)

    policy = load_policy(path)
    assert policy.sha256 == sha
    assert policy.spec == spec
    assert policy.metadata == {"note": "test", "seed": 1}
    assert (policy.obs_normalizer is None) == (normalizer is None)

    obs = np.random.default_rng(0).normal(size=(6, spec.obs_dim)).astype(np.float32)
    expected = _reference_action(spec, actor, obs, normalizer)
    np.testing.assert_allclose(policy.act(obs), expected, atol=1e-6)
    np.testing.assert_allclose(policy.act(obs[0]), expected[0], atol=1e-6)
    np.testing.assert_allclose(policy(obs[0]), policy.act(obs[0]))  # deterministic, callable
    single = policy.act(obs[0])
    assert single.shape == (spec.act_dim,) and single.dtype == np.float32
    assert np.all(single >= np.array(spec.action_low)) and np.all(
        single <= np.array(spec.action_high)
    )
    assert not any(p.requires_grad for p in policy.actor.parameters())


@pytest.mark.unit
def test_ppo_normalizer_is_applied_and_clipped(tmp_path: Path) -> None:
    spec = SPECS["ppo_continuous"]
    actor = _trained_looking_actor(spec)
    normalizer = ObsNormalizer(mean=np.zeros(5), var=np.full(5, 1e-4), epsilon=1e-8, clip=10.0)
    save_policy(tmp_path / "p.pt", actor, spec, metadata={}, obs_normalizer=normalizer)
    policy = load_policy(tmp_path / "p.pt")
    assert policy.obs_normalizer is not None and policy.obs_normalizer.clip == 10.0
    huge = np.full(5, 1e3, dtype=np.float32)
    np.testing.assert_allclose(policy.obs_normalizer(huge), np.full(5, 10.0))  # clipped
    saturated = policy.act(huge)  # every feature hits the clip: the same input as 10 * ones
    np.testing.assert_allclose(saturated, policy.act(np.full(5, 1e2, dtype=np.float32)), atol=1e-6)


@pytest.mark.unit
def test_spec_validation() -> None:
    good = SPECS["sac"]
    with pytest.raises(ValueError, match="unknown algo"):
        PolicySpec("dqn", 4, 2, (-1.0, -1.0), (1.0, 1.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="length act_dim"):
        PolicySpec("sac", 4, 2, (-1.0,), (1.0, 1.0))
    with pytest.raises(ValueError, match="low < high"):
        PolicySpec("sac", 4, 2, (-1.0, 1.0), (1.0, 1.0))
    with pytest.raises(ValueError, match="finite"):
        PolicySpec("sac", 4, 2, (-1.0, -np.inf), (1.0, 1.0))
    with pytest.raises(ValueError, match="positive"):
        PolicySpec("sac", 0, 2, (-1.0, -1.0), (1.0, 1.0))
    with pytest.raises(ValueError, match="arch"):
        PolicySpec("sac", 4, 2, (-1.0, -1.0), (1.0, 1.0), {"hidden": 8})
    with pytest.raises(ValueError, match="arch"):
        PolicySpec("flashsac", 4, 2, (-1.0, -1.0), (1.0, 1.0))
    assert PolicySpec.from_dict(good.to_dict()) == good
    listy = PolicySpec("sac", 4, 2, [-1, -1], [1, 1])  # type: ignore[arg-type]
    assert listy.action_low == (-1.0, -1.0) and listy.action_high == (1.0, 1.0)
    with pytest.raises(ValueError, match="malformed"):
        PolicySpec.from_dict({"algo": "sac"})


@pytest.mark.unit
def test_obs_normalizer_validation() -> None:
    with pytest.raises(ValueError):
        ObsNormalizer(mean=np.zeros(3), var=np.ones(2), epsilon=1e-8)
    with pytest.raises(ValueError):
        ObsNormalizer(mean=np.zeros(3), var=-np.ones(3), epsilon=1e-8)
    with pytest.raises(ValueError):
        ObsNormalizer(mean=np.zeros(3), var=np.ones(3), epsilon=0.0)
    n = ObsNormalizer(mean=np.ones(3), var=np.full(3, 4.0), epsilon=1e-12)
    np.testing.assert_allclose(n(np.array([3.0, 3.0, 3.0])), [1.0, 1.0, 1.0], atol=1e-6)


@pytest.mark.unit
def test_save_refuses_mismatches(tmp_path: Path) -> None:
    sac_actor = build_actor(SPECS["sac"])
    with pytest.raises(ValueError, match="do not fit"):  # spec says obs_dim 6, actor has 5
        save_policy(
            tmp_path / "x.pt",
            sac_actor,
            PolicySpec("sac", 6, 2, (-2.0, -1.0), (2.0, 1.0)),
            metadata={},
        )
    with pytest.raises(ValueError, match="action_scale"):  # buffers disagree with the bounds
        save_policy(
            tmp_path / "x.pt",
            sac_actor,
            PolicySpec("sac", 5, 2, (-1.0, -1.0), (1.0, 1.0)),
            metadata={},
        )
    with pytest.raises(ValueError, match="obs_normalizer"):  # SAC never has a normaliser
        save_policy(
            tmp_path / "x.pt",
            sac_actor,
            SPECS["sac"],
            metadata={},
            obs_normalizer=_normalizer(SPECS["ppo_continuous"]),
        )
    ppo_actor = build_actor(SPECS["ppo_continuous"])
    with pytest.raises(ValueError, match="obs_normalizer"):  # PPO always has one
        save_policy(tmp_path / "x.pt", ppo_actor, SPECS["ppo_continuous"], metadata={})
    wrong_size = ObsNormalizer(mean=np.zeros(4), var=np.ones(4), epsilon=1e-8)
    with pytest.raises(ValueError, match="obs_normalizer"):
        save_policy(
            tmp_path / "x.pt",
            ppo_actor,
            SPECS["ppo_continuous"],
            metadata={},
            obs_normalizer=wrong_size,
        )
    assert not (tmp_path / "x.pt").exists()  # nothing was written by any refused save


@pytest.mark.unit
def test_load_refuses_foreign_and_tampered_files(tmp_path: Path) -> None:
    other = tmp_path / "other.pt"
    torch.save({"format": "something.else", "weights": torch.zeros(2)}, other)
    with pytest.raises(ValueError, match="not a roborl policy"):
        load_policy(other)

    pickled = tmp_path / "pickled.pt"
    torch.save({"format": FORMAT, "obj": _ArbitraryObject()}, pickled)
    with pytest.raises(ValueError, match="not a roborl policy"):
        load_policy(pickled)

    (tmp_path / "garbage.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="not a roborl policy"):
        load_policy(tmp_path / "garbage.pt")

    spec = SPECS["sac"]
    good = tmp_path / "good.pt"
    save_policy(good, build_actor(spec), spec, metadata={})
    payload = torch.load(good, weights_only=True)
    payload["format_version"] = 99
    torch.save(payload, tmp_path / "future.pt")
    with pytest.raises(ValueError, match="format version"):
        load_policy(tmp_path / "future.pt")
    payload = torch.load(good, weights_only=True)
    payload["spec"]["act_dim"] = 3
    payload["spec"]["action_low"] = [-1.0, -1.0, -1.0]
    payload["spec"]["action_high"] = [1.0, 1.0, 1.0]
    torch.save(payload, tmp_path / "edited.pt")
    with pytest.raises(ValueError, match="do not fit"):
        load_policy(tmp_path / "edited.pt")


@pytest.mark.unit
def test_metadata_is_reduced_to_primitives(tmp_path: Path) -> None:
    spec = SPECS["sac"]
    meta = {
        "arr": np.arange(3),
        "scalar": np.float32(1.5),
        "path": Path("/x/y"),
        "nested": {"t": (1, 2)},
    }
    save_policy(tmp_path / "m.pt", build_actor(spec), spec, metadata=meta)
    loaded = load_policy(tmp_path / "m.pt").metadata
    assert loaded == {"arr": [0, 1, 2], "scalar": 1.5, "path": "/x/y", "nested": {"t": [1, 2]}}


@pytest.mark.unit
def test_act_rejects_bad_observations(tmp_path: Path) -> None:
    spec = SPECS["sac"]
    save_policy(tmp_path / "s.pt", build_actor(spec), spec, metadata={})
    policy = load_policy(tmp_path / "s.pt")
    with pytest.raises(ValueError, match="size"):
        policy.act(np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError, match="size"):
        policy.act(np.zeros((2, 3, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="NaN"):
        policy.act(np.array([np.nan, 0, 0, 0, 0], dtype=np.float32))
