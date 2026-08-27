"""Fetch adapters: local loading, curve splitting, offline cache hits."""

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas", reason="benchmark extra not installed")

from roborl.benchmark.fetch import (  # noqa: E402  (import valid only after the skip guard)
    OpenRLBenchmarkAdapter,
    load_runs,
    to_curves,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "reference_synthetic.parquet"


@pytest.mark.unit
def test_load_runs_parquet_fixture() -> None:
    frame = load_runs([FIXTURE])
    assert list(frame.columns) == ["run_id", "global_step", "episodic_return"]
    assert frame["run_id"].nunique() == 3


@pytest.mark.unit
def test_load_runs_csv(tmp_path: Path) -> None:
    csv = tmp_path / "run.csv"
    csv.write_text("run_id,global_step,episodic_return\nr1,10,1.5\nr1,20,2.5\n")
    frame = load_runs([csv])
    assert len(frame) == 2


@pytest.mark.unit
def test_load_runs_missing_columns_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("step,reward\n1,2\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_runs([bad])


@pytest.mark.unit
def test_to_curves_splits_and_sorts(tmp_path: Path) -> None:
    csv = tmp_path / "runs.csv"
    csv.write_text("run_id,global_step,episodic_return\nb,20,2.0\nb,10,1.0\na,5,0.5\n")
    curves = to_curves(load_runs([csv]))
    assert len(curves) == 2
    steps_b, values_b = curves[1]
    assert steps_b.tolist() == [10.0, 20.0]  # sorted by step
    assert values_b.tolist() == [1.0, 2.0]


@pytest.mark.unit
def test_adapter_cache_hit_stays_offline(tmp_path: Path) -> None:
    adapter = OpenRLBenchmarkAdapter(cache_dir=tmp_path)
    cache = adapter.cache_path("dqn", "CartPole-v1")
    cache.parent.mkdir(parents=True)
    pd.read_parquet(FIXTURE).to_parquet(cache, index=False)
    # A cache hit must return without touching wandb or the network.
    frame = adapter.fetch("dqn", "CartPole-v1")
    assert frame["run_id"].nunique() == 3
