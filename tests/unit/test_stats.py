"""Statistics reproduce hand-computed values on tiny fixtures."""

import numpy as np
import pytest

from roborl.benchmark.stats import (
    align_curves,
    final_scores,
    iqm,
    iqm_curve_with_ci,
    stratified_bootstrap_ci,
)


@pytest.mark.unit
class TestIQM:
    def test_hand_computed_eight_values(self) -> None:
        # n=8, trim floor(0.25*8)=2 per end -> mean of [3,4,5,6] = 4.5
        assert iqm(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)) == 4.5

    def test_hand_computed_four_values(self) -> None:
        # n=4, trim 1 per end -> mean of [2,3] = 2.5
        assert iqm(np.array([4.0, 1.0, 3.0, 2.0])) == 2.5

    def test_outlier_robustness(self) -> None:
        # The outlier 1000 falls in the trimmed top quartile.
        assert iqm(np.array([1, 2, 3, 4, 5, 6, 7, 1000], dtype=float)) == 4.5

    def test_single_value(self) -> None:
        assert iqm(np.array([7.0])) == 7.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            iqm(np.array([]))


@pytest.mark.unit
class TestBootstrapCI:
    def test_constant_data_gives_degenerate_ci(self) -> None:
        assert stratified_bootstrap_ci(np.full(5, 3.0)) == (3.0, 3.0)

    def test_deterministic_given_seed(self) -> None:
        scores = np.array([10.0, 12.0, 9.0, 14.0, 11.0])
        assert stratified_bootstrap_ci(scores, seed=7) == stratified_bootstrap_ci(scores, seed=7)

    def test_ci_brackets_point_estimate(self) -> None:
        rng = np.random.default_rng(1)
        scores = rng.normal(100, 10, size=10)
        lo, hi = stratified_bootstrap_ci(scores)
        assert lo <= iqm(scores) <= hi
        assert lo < hi

    def test_stratified_shape_two_tasks(self) -> None:
        matrix = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        lo, hi = stratified_bootstrap_ci(matrix)
        assert lo <= hi

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            stratified_bootstrap_ci(np.array([]))


@pytest.mark.unit
class TestCurves:
    def test_align_linear_is_exact(self) -> None:
        curves = [(np.array([0.0, 10.0]), np.array([0.0, 100.0]))]
        aligned = align_curves(curves, np.array([0.0, 5.0, 10.0]))
        assert aligned.tolist() == [[0.0, 50.0, 100.0]]

    def test_align_clamps_outside_range(self) -> None:
        curves = [(np.array([5.0, 10.0]), np.array([1.0, 2.0]))]
        aligned = align_curves(curves, np.array([0.0, 20.0]))
        assert aligned.tolist() == [[1.0, 2.0]]

    def test_iqm_curve_on_identical_runs(self) -> None:
        run = (np.array([0.0, 10.0]), np.array([0.0, 100.0]))
        grid = np.array([0.0, 5.0, 10.0])
        point, lo, hi = iqm_curve_with_ci([run, run, run, run], grid)
        assert point.tolist() == [0.0, 50.0, 100.0]
        assert lo.tolist() == point.tolist() == hi.tolist()

    def test_final_scores_hand_computed(self) -> None:
        # Steps 0..100; last 10% is steps >= 90 -> values [90, 100] -> mean 95.
        steps = np.arange(0.0, 101.0, 10.0)
        (score,) = final_scores([(steps, steps.copy())], last_fraction=0.1)
        assert score == 95.0
