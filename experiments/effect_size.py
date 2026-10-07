"""Does the V6 per-group effect size recover the injected label gap?

For each dataset we compare theta_hat for group a = 1 (Eq. 3 of the paper) with
the realised injected gap: the mean of (y - y_clean) over audit records with
a = 1, i.e. minus the share of those records whose label was flipped.

Usage:  python experiments/effect_size.py --reps 300
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cfa.simulate import DEFAULT, feature_names, generate  # noqa: E402
from cfa.v6 import AuditConfig, audit_v6  # noqa: E402


def one(c: float, rep: int) -> dict:
    seed = 11_000_000 + int(c * 100) * 10_000 + rep
    sc = replace(DEFAULT, corruption=c)
    df = generate(sc, seed=seed)
    res = audit_v6(df, feature_names(sc), config=AuditConfig(seed=seed))
    g = res.groups[(res.groups["feature"] == "a") & (res.groups["group"] == "a=1")].iloc[0]
    audit_a1 = df[(df["source"] == 1) & (df["a"] == 1)]
    truth = float((audit_a1["y"] - audit_a1["y_clean"]).mean())
    return {"corruption": c, "rep": rep, "theta_hat": g["label_gap"], "lo": g["label_gap_lo"],
            "hi": g["label_gap_hi"], "truth": truth, "flagged_a": int(res.selects("a"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()
    rows = Parallel(n_jobs=args.jobs)(delayed(one)(c, r) for c in (0.0, 0.2, 0.3, 0.4) for r in range(args.reps))
    runs = pd.DataFrame(rows)
    runs["covered"] = (runs["lo"] <= runs["truth"]) & (runs["truth"] <= runs["hi"])
    summary = runs.groupby("corruption").agg(
        reps=("rep", "size"), mean_truth=("truth", "mean"), mean_theta=("theta_hat", "mean"),
        sd_theta=("theta_hat", "std"), coverage=("covered", "mean"),
        mean_abs_error=("theta_hat", lambda s: float(np.mean(np.abs(s - runs.loc[s.index, "truth"]))))).reset_index()
    out = os.path.join(ROOT, "results")
    runs.to_csv(os.path.join(out, "effect_size_runs.csv"), index=False)
    summary.to_csv(os.path.join(out, "effect_size_summary.csv"), index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
