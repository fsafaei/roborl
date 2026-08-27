"""Comparison report: verdict policy and rendered artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from roborl.benchmark.report import decide_verdict, run_compare

FIXTURE = Path(__file__).parent.parent / "fixtures" / "reference_synthetic.parquet"


@pytest.mark.unit
class TestVerdictPolicy:
    def test_overlapping_cis_pass(self) -> None:
        assert decide_verdict(100.0, (90.0, 110.0), 105.0, (95.0, 115.0), n_reference=5) == "PASS"

    def test_disjoint_cis_investigate(self) -> None:
        verdict = decide_verdict(50.0, (45.0, 55.0), 100.0, (95.0, 105.0), n_reference=5)
        assert verdict == "INVESTIGATE"

    def test_few_reference_runs_use_ratio_rule(self) -> None:
        # CIs disjoint, but with n_reference < 3 the 90%-of-reference rule applies.
        assert decide_verdict(95.0, (94.0, 96.0), 100.0, (100.0, 100.0), n_reference=2) == "PASS"
        verdict = decide_verdict(80.0, (79.0, 81.0), 100.0, (100.0, 100.0), n_reference=2)
        assert verdict == "INVESTIGATE"


def _write_runs(path: Path, offset: float) -> Path:
    rng = np.random.default_rng(42)
    frames = []
    for i in range(3):
        steps = np.arange(200, 10_001, 200)
        returns = offset + 500 * (1 - np.exp(-steps / 3000)) + rng.normal(0, 10, steps.size)
        frames.append(
            pd.DataFrame({"run_id": f"ours-{i}", "global_step": steps, "episodic_return": returns})
        )
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


@pytest.mark.unit
def test_compare_similar_runs_passes(tmp_path: Path) -> None:
    ours = _write_runs(tmp_path / "ours.csv", offset=0.0)
    result = run_compare(
        ours_paths=[ours],
        reference_paths=[FIXTURE],
        algo="demo",
        env_id="Synthetic-v0",
        out_dir=tmp_path / "out",
    )
    assert result.verdict == "PASS"
    assert result.report_path.exists() and result.figure_path.exists()
    report = result.report_path.read_text()
    assert "PASS" in report and "curves.png" in report and "Synthetic-v0" in report


@pytest.mark.unit
def test_compare_weak_runs_investigate(tmp_path: Path) -> None:
    ours = _write_runs(tmp_path / "ours.csv", offset=-300.0)
    result = run_compare(
        ours_paths=[ours],
        reference_paths=[FIXTURE],
        algo="demo",
        env_id="Synthetic-v0",
        out_dir=tmp_path / "out",
    )
    assert result.verdict == "INVESTIGATE"
