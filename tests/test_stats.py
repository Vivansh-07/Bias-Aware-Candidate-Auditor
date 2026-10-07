"""Decision primitives must reproduce the numbers quoted in the project report."""
import numpy as np
import pytest

from cfa.stats import (abstention_gate, bh_adjust, by_adjust, by_rank_one_cutoff, calibration_gate,
                       clopper_pearson, holm_adjust)


@pytest.mark.parametrize("k, n, upper", [
    (0, 30, 0.1157),    # V2: zero alarms in 30 runs -> ~11.6%
    (17, 500, 0.0539),  # V6 failed default condition
    (16, 500, 0.0514),  # V5 mean shift 0.75
    (44, 500, 0.1163),  # V5 mean shift 1.50
])
def test_clopper_pearson_upper_matches_report(k, n, upper):
    assert clopper_pearson(k, n)[1] == pytest.approx(upper, abs=6e-4)


@pytest.mark.parametrize("k, n, lower", [
    (494, 500, 0.9741),  # V6 random-label abstention
    (294, 300, 0.9570),  # V3 milestone 4 abstention
    (490, 500, 0.9635),  # V5 abstention
])
def test_clopper_pearson_lower_matches_report(k, n, lower):
    assert clopper_pearson(k, n)[0] == pytest.approx(lower, abs=6e-4)


def test_v6_default_condition_fails_gate():
    assert not calibration_gate(17, 500)["passed"]
    assert calibration_gate(5, 500)["passed"]
    assert abstention_gate(494, 500)["passed"]


def test_by_rank_one_cutoff_for_26_blocks():
    # Report: with 26 blocks the BY rank-one cutoff is ~0.000499 < 1/2000
    cut = by_rank_one_cutoff(26)
    assert cut == pytest.approx(0.000499, abs=2e-6)
    assert cut < 1 / 2000


def test_adjustments_are_monotone_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.random(40) ** 3
    for adj in (by_adjust, bh_adjust, holm_adjust):
        q = adj(p)
        assert np.all(q >= p - 1e-15)
        assert np.all(q <= 1.0)
        order = np.argsort(p)
        assert np.all(np.diff(q[order]) >= -1e-12)
    assert np.all(by_adjust(p) >= bh_adjust(p) - 1e-15)


def test_single_pvalue_unchanged_by_by():
    assert by_adjust([0.03])[0] == pytest.approx(0.03)
