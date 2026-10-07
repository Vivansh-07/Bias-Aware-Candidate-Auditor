"""Multiple-testing corrections and exact binomial intervals.

These are the decision primitives used by every auditor version:
- Benjamini-Yekutieli (BY) across feature blocks (primary gate, V3-V6)
- Bonferroni within a block (V6)
- Holm (V4 comparator) and Benjamini-Hochberg (secondary diagnostic)
- Clopper-Pearson exact two-sided intervals for alarm / abstention gates
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _as_array(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError("p-values must be a 1-D array")
    return p


def bh_adjust(p) -> np.ndarray:
    """Benjamini-Hochberg step-up adjusted p-values."""
    return _step_up(_as_array(p), dependence_factor=1.0)


def by_adjust(p) -> np.ndarray:
    """Benjamini-Yekutieli adjusted p-values (valid under arbitrary dependence)."""
    p = _as_array(p)
    m = len(p)
    c_m = np.sum(1.0 / np.arange(1, m + 1)) if m else 1.0
    return _step_up(p, dependence_factor=c_m)


def _step_up(p: np.ndarray, dependence_factor: float) -> np.ndarray:
    m = len(p)
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    ranked = p[order] * m * dependence_factor / np.arange(1, m + 1)
    # enforce monotonicity from the largest rank downwards
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    return out


def holm_adjust(p) -> np.ndarray:
    """Holm step-down adjusted p-values (familywise error control)."""
    p = _as_array(p)
    m = len(p)
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    ranked = p[order] * (m - np.arange(m))
    ranked = np.maximum.accumulate(ranked)
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    return out


def by_rank_one_cutoff(m: int, alpha: float = 0.05) -> float:
    """Smallest raw p-value a single isolated block must beat under BY."""
    c_m = np.sum(1.0 / np.arange(1, m + 1))
    return alpha / (m * c_m)


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    a = (1.0 - level) / 2.0
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1.0 - a, k + 1, n - k))
    return lo, hi


def calibration_gate(alarms: int, runs: int, threshold: float = 0.05) -> dict:
    """Clean-data gate: the exact upper 95% endpoint must lie below `threshold`."""
    lo, hi = clopper_pearson(alarms, runs)
    return {
        "alarms": int(alarms),
        "runs": int(runs),
        "rate": alarms / runs if runs else float("nan"),
        "ci_low": lo,
        "ci_high": hi,
        "passed": bool(hi < threshold),
    }


def abstention_gate(abstentions: int, runs: int, threshold: float = 0.90) -> dict:
    """Random-label gate: the exact lower 95% endpoint must reach `threshold`."""
    lo, hi = clopper_pearson(abstentions, runs)
    return {
        "abstentions": int(abstentions),
        "runs": int(runs),
        "rate": abstentions / runs if runs else float("nan"),
        "ci_low": lo,
        "ci_high": hi,
        "passed": bool(lo >= threshold),
    }
