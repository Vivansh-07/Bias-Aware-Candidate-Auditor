"""Real-data (semi-synthetic) transfer.

A real dataset is split record-disjointly into
  test     clean labels, held out for downstream fairness evaluation
  trusted  source = 0, clean reference labels
  audit    source = 1, optionally with injected label corruption and/or a
           clean population shift (biased sampling on a covariate that leaves
           P(Y | X) unchanged)
Real feature distributions and correlations are preserved; only the corruption
mechanism is controlled, so recovery can be scored against a known truth.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

GERMAN_CODES = {
    "checking_account": {"A11": "<0 DM", "A12": "0-200 DM", "A13": ">=200 DM", "A14": "no account"},
    "credit_history": {"A30": "no credits", "A31": "all paid (this bank)", "A32": "existing paid",
                       "A33": "past delay", "A34": "critical account"},
    "purpose": {"A40": "car (new)", "A41": "car (used)", "A42": "furniture", "A43": "radio/TV",
                "A44": "appliances", "A45": "repairs", "A46": "education", "A47": "vacation",
                "A48": "retraining", "A49": "business", "A410": "other"},
    "savings_account": {"A61": "<100 DM", "A62": "100-500 DM", "A63": "500-1000 DM",
                        "A64": ">=1000 DM", "A65": "unknown/none"},
    "employment": {"A71": "unemployed", "A72": "<1 yr", "A73": "1-4 yrs", "A74": "4-7 yrs", "A75": ">=7 yrs"},
    "personal_status": {"A91": "male: divorced/separated", "A92": "female: div/sep/married",
                        "A93": "male: single", "A94": "male: married/widowed", "A95": "female: single"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {"A121": "real estate", "A122": "savings/insurance", "A123": "car/other", "A124": "none"},
    "other_installments": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "free"},
    "job": {"A171": "unskilled non-resident", "A172": "unskilled resident", "A173": "skilled",
            "A174": "management/self-employed"},
    "telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}


@dataclass(frozen=True)
class DatasetInfo:
    file: str
    label: str
    description: str
    target_feature: str          # default corruption attribute
    target_values: tuple         # default corrupted group
    flip_from: int
    flip_to: int
    shift_feature: str
    label_meaning: str
    drop: tuple = field(default_factory=tuple)


DATASETS: dict[str, DatasetInfo] = {
    "COMPAS": DatasetInfo(
        "compas_clean.csv", "two_year_recid",
        "ProPublica COMPAS, Broward County (6,172 defendants, ProPublica filtering).",
        "race", ("African-American",), 0, 1, "age",
        "1 = re-arrested within two years. Injected corruption records some non-recidivists "
        "in the target group as recidivists (e.g. differential policing)."),
    "German Credit": DatasetInfo(
        "german_credit.csv", "credit_risk",
        "UCI Statlog German Credit (1,000 applicants, codes decoded).",
        "personal_status", ("female: div/sep/married",), 0, 1, "age",
        "1 = bad credit risk. Injected corruption marks some good-risk applicants in the "
        "target group as bad risk."),
    "Heart Disease": DatasetInfo(
        "heart_disease.csv", "target",
        "UCI Cleveland Heart Disease (303 patients). Deliberately small: shows support-based abstention.",
        "sex", (0,), 1, 0, "age",
        "1 = disease present. Injected corruption under-records disease in the target group."),
}


def load_dataset(name: str) -> pd.DataFrame:
    info = DATASETS[name]
    df = pd.read_csv(os.path.join(DATA_DIR, info.file), encoding="utf-8-sig")
    if name == "German Credit":
        for col, mapping in GERMAN_CODES.items():
            if col in df:
                df[col] = df[col].map(mapping).fillna(df[col])
    return df.drop(columns=list(info.drop), errors="ignore")


@dataclass
class CorruptionSpec:
    feature: str
    values: tuple
    flip_from: int
    flip_to: int
    rate: float

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.feature].isin(self.values).to_numpy()


@dataclass
class ShiftSpec:
    feature: str
    strength: float  # log-odds of audit selection per SD of the feature


def make_semisynthetic(df: pd.DataFrame, label: str, test_frac: float = 0.25,
                       trusted_frac: float = 0.30, audit_frac: float = 0.70,
                       corruption: CorruptionSpec | None = None, shift: ShiftSpec | None = None,
                       seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (analysis frame with y / y_clean / source, clean test frame)."""
    rng = np.random.default_rng(seed)
    df = df.dropna(subset=[label]).reset_index(drop=True)
    y_all = df[label].astype(int).to_numpy()
    idx = rng.permutation(len(df))
    n_test = int(round(test_frac * len(df)))
    test_idx, pool = idx[:n_test], idx[n_test:]
    n_trusted = int(round(trusted_frac * len(pool)))
    trusted_idx, rest = pool[:n_trusted], pool[n_trusted:]
    n_audit = int(round(audit_frac * len(rest)))
    if shift is not None and shift.strength != 0:
        x = df.loc[rest, shift.feature].astype(float)
        z = ((x - x.mean()) / (x.std() or 1.0)).fillna(0).to_numpy()
        w = np.exp(shift.strength * z)
        audit_idx = rng.choice(rest, size=n_audit, replace=False, p=w / w.sum())
    else:
        audit_idx = rng.choice(rest, size=n_audit, replace=False)

    trusted = df.loc[trusted_idx].copy()
    trusted["source"] = 0
    audit = df.loc[audit_idx].copy()
    audit["source"] = 1
    out = pd.concat([trusted, audit], ignore_index=True)
    out["y_clean"] = np.concatenate([y_all[trusted_idx], y_all[audit_idx]])
    y = out["y_clean"].to_numpy().copy()
    if corruption is not None and corruption.rate > 0:
        eligible = (out["source"].to_numpy() == 1) & corruption.mask(out) & (y == corruption.flip_from)
        flip = eligible & (rng.random(len(out)) < corruption.rate)
        y[flip] = corruption.flip_to
    out["y"] = y
    if label != "y":
        out = out.drop(columns=[label])

    test = df.loc[test_idx].copy()
    test["y"] = y_all[test_idx]
    test["y_clean"] = test["y"]
    if label != "y":
        test = test.drop(columns=[label])
    return out, test.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("y", "y_clean", "source")]
