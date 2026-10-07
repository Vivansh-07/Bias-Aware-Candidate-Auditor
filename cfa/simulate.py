"""Synthetic generator matching the project's experimental family.

Columns
  x1..x5  legitimate predictors (Gaussian)
  a       binary group (the corruption target; the auditor is NOT told it is sensitive)
  p       noisy proxy of `a` (flipped with probability `proxy_noise`)
  z1..zk  irrelevant nuisance columns
  y       observed label (corrupted in the audit source when corruption > 0)
  source  0 = trusted reference, 1 = audit
  y_clean the clean label (hidden from the auditor; used for evaluation only)

Corruption meaning: an eligible positive label (a = 1, y = 1) in the AUDIT
source is flipped to 0 with probability `corruption`. Shift moves the audit
mean of x1 without changing the label equation (a clean population change).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

BETA = np.array([0.8, -0.6, 0.5, 0.4, -0.3])


@dataclass(frozen=True)
class Scenario:
    n_audit: int = 5000
    n_trusted: int = 1667
    prevalence: float = 0.25
    proxy_noise: float = 0.10
    n_noise: int = 5
    group_effect: float = 0.3
    corruption: float = 0.0
    shift: float = 0.0
    nonlinear: bool = False
    missing: float = 0.0
    correlated_nuisance: bool = False
    random_labels: bool = False
    intercept: float = -0.3

    def as_dict(self) -> dict:
        return asdict(self)


FEATURES_BASE = ["x1", "x2", "x3", "x4", "x5", "a", "p"]


def feature_names(sc: Scenario) -> list[str]:
    return FEATURES_BASE + [f"z{k + 1}" for k in range(sc.n_noise)]


def _sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


def _draw_source(n: int, sc: Scenario, audit: bool, rng: np.random.Generator) -> pd.DataFrame:
    X = rng.standard_normal((n, 5))
    if audit and sc.shift:
        X[:, 0] += sc.shift
    a = (rng.random(n) < sc.prevalence).astype(int)
    p = np.where(rng.random(n) < sc.proxy_noise, 1 - a, a)
    if sc.correlated_nuisance:
        Z = np.column_stack([0.7 * X[:, k % 5] + np.sqrt(1 - 0.49) * rng.standard_normal(n)
                             for k in range(sc.n_noise)]) if sc.n_noise else np.empty((n, 0))
    else:
        Z = rng.standard_normal((n, sc.n_noise))

    logit = sc.intercept + X @ BETA + sc.group_effect * a
    if sc.nonlinear:
        logit = logit + 0.6 * (X[:, 0] ** 2 - 1) + 0.5 * X[:, 1] * X[:, 2]
    if sc.random_labels:
        y_clean = (rng.random(n) < 0.45).astype(int)
    else:
        y_clean = (rng.random(n) < _sigmoid(logit)).astype(int)

    y = y_clean.copy()
    if audit and sc.corruption > 0:
        eligible = (a == 1) & (y == 1)
        y[eligible & (rng.random(n) < sc.corruption)] = 0

    df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4", "x5"])
    df["a"] = a
    df["p"] = p
    for k in range(sc.n_noise):
        df[f"z{k + 1}"] = Z[:, k]
    if sc.missing > 0:
        for c in ["x1", "x2", "x3", "x4", "x5"] + [f"z{k + 1}" for k in range(sc.n_noise)]:
            df.loc[rng.random(n) < sc.missing, c] = np.nan
    df["y"] = y
    df["source"] = int(audit)
    df["y_clean"] = y_clean
    return df


def generate(sc: Scenario, seed: int = 0) -> pd.DataFrame:
    """One dataset: trusted rows (source=0) followed by audit rows (source=1)."""
    rng = np.random.default_rng(seed)
    trusted = _draw_source(sc.n_trusted, sc, audit=False, rng=rng)
    audit = _draw_source(sc.n_audit, sc, audit=True, rng=rng)
    return pd.concat([trusted, audit], ignore_index=True)


def generate_test(sc: Scenario, n: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Independent clean test sample from the audit population (for mitigation)."""
    rng = np.random.default_rng(seed + 10_000_019)
    clean = replace(sc, corruption=0.0)
    return _draw_source(n, clean, audit=True, rng=rng)


DEFAULT = Scenario()

# The calibration conditions of the V5/V6 control studies (no corruption).
CALIBRATION_CONDITIONS: dict[str, Scenario] = {
    "default": DEFAULT,
    "proxy_noise_0": replace(DEFAULT, proxy_noise=0.0),
    "proxy_noise_30": replace(DEFAULT, proxy_noise=0.30),
    "proxy_noise_50": replace(DEFAULT, proxy_noise=0.50),
    "n_audit_1000": replace(DEFAULT, n_audit=1000, n_trusted=333),
    "n_audit_10000": replace(DEFAULT, n_audit=10000, n_trusted=3333),
    "trusted_833": replace(DEFAULT, n_trusted=833),
    "trusted_3333": replace(DEFAULT, n_trusted=3333),
    "prevalence_10": replace(DEFAULT, prevalence=0.10),
    "prevalence_50": replace(DEFAULT, prevalence=0.50),
    "noise_cols_20": replace(DEFAULT, n_noise=20),
    "nonlinear": replace(DEFAULT, nonlinear=True),
    "missing_30": replace(DEFAULT, missing=0.30),
    "correlated_nuisance": replace(DEFAULT, correlated_nuisance=True),
    "shift_0.75": replace(DEFAULT, shift=0.75),
    "shift_1.50": replace(DEFAULT, shift=1.50),
    "shift_1.50_nonlinear": replace(DEFAULT, shift=1.50, nonlinear=True),
    "shift_1.50_corr_nuisance": replace(DEFAULT, shift=1.50, correlated_nuisance=True),
    "shift_1.50_noise_20": replace(DEFAULT, shift=1.50, n_noise=20),
    "shift_1.50_prev_10": replace(DEFAULT, shift=1.50, prevalence=0.10),
    "random_labels": replace(DEFAULT, random_labels=True),
}

# Demo presets for the web app.
PRESETS: dict[str, Scenario] = {
    "Matched clean control": DEFAULT,
    "Label corruption (30%)": replace(DEFAULT, corruption=0.30),
    "Strong corruption (40%, n=10k)": replace(DEFAULT, corruption=0.40, n_audit=10000, n_trusted=3333),
    "Clean population shift (1.5)": replace(DEFAULT, shift=1.50),
    "Shift + corruption": replace(DEFAULT, shift=1.50, corruption=0.40, n_audit=10000, n_trusted=3333),
    "Random labels": replace(DEFAULT, random_labels=True),
}
