"""Semi-synthetic real-data transfer study.

Each repetition draws a fresh record-disjoint split (test / trusted / audit) of
a real dataset, optionally injects corruption into the audit labels and/or
applies a clean population shift, then runs V6 and V5 on the same data.
When V6 flags (the mitigation gate), the audit-guided reweighting is compared
with the baseline and the clean-label oracle on the held-out clean test set.

Usage
  python experiments/run_realdata.py --reps 200 --jobs 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cfa.mitigation import audit_guided_weights, evaluate_strategies  # noqa: E402
from cfa.realdata import DATASETS, CorruptionSpec, ShiftSpec, feature_columns, load_dataset, make_semisynthetic  # noqa: E402
from cfa.stats import calibration_gate, clopper_pearson  # noqa: E402
from cfa.v5 import audit_v5  # noqa: E402
from cfa.v6 import AuditConfig, audit_v6  # noqa: E402

CONDITIONS = {
    "COMPAS": [("clean", 0.0, 0.0), ("shift_age_1.0", 0.0, 1.0), ("shift_age_1.5", 0.0, 1.5),
               ("corrupt_10", 0.10, 0.0), ("corrupt_20", 0.20, 0.0), ("corrupt_30", 0.30, 0.0),
               ("corrupt_30_shift_1.0", 0.30, 1.0)],
    "German Credit": [("clean", 0.0, 0.0), ("shift_age_1.0", 0.0, 1.0),
                      ("corrupt_30", 0.30, 0.0), ("corrupt_50", 0.50, 0.0)],
}


def one_rep(dataset: str, cond_idx: int, cond: str, rate: float, shift: float, rep: int, B: int):
    info = DATASETS[dataset]
    raw = load_dataset(dataset)
    seed = 3_000_000 + 100_000 * list(DATASETS).index(dataset) + 10_000 * cond_idx + rep
    corr = CorruptionSpec(info.target_feature, info.target_values, info.flip_from, info.flip_to, rate) if rate else None
    sh = ShiftSpec(info.shift_feature, shift) if shift else None
    data, test = make_semisynthetic(raw, info.label, corruption=corr, shift=sh, seed=seed)
    feats = feature_columns(data)
    cfg = AuditConfig(seed=seed)
    r6 = audit_v6(data, feats, config=cfg)
    r5 = audit_v5(data, feats, config=cfg, B=B)
    runs = []
    for r in (r6, r5):
        runs.append({"dataset": dataset, "condition": cond, "rep": rep, "seed": seed, "method": r.method,
                     "corruption": rate, "shift": shift, "alarm": int(r.flagged),
                     "target_selected": int(r.selects(info.target_feature)),
                     "off_target_selected": int(any(info.target_feature not in b for b in r.selected_blocks)),
                     "selected": json.dumps(r.selected_blocks)})
    mit = []
    if rate > 0 and r6.flagged:
        wr = audit_guided_weights(data, feats, r6, seed=seed)
        ev = evaluate_strategies(data, test, feats, r6, corr.mask(test), weights=wr, seed=seed)
        ev["dataset"], ev["condition"], ev["rep"], ev["ess"] = dataset, cond, rep, wr.ess_ratio
        ev["weights_accepted"] = wr.accepted
        ev["target_flagged"] = int(r6.selects(info.target_feature))
        mit = ev.to_dict(orient="records")
    return runs, mit


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (ds, cond, m), g in runs.groupby(["dataset", "condition", "method"], sort=False):
        n = len(g)
        if g["corruption"].iloc[0] == 0:
            gate = calibration_gate(int(g["alarm"].sum()), n)
            out.append({"dataset": ds, "condition": cond, "method": m, "endpoint": "clean_alarm",
                        "count": gate["alarms"], "runs": n, "rate": gate["rate"], "ci_low": gate["ci_low"],
                        "ci_high": gate["ci_high"], "passed": gate["passed"]})
        else:
            k = int(g["target_selected"].sum())
            lo, hi = clopper_pearson(k, n)
            out.append({"dataset": ds, "condition": cond, "method": m, "endpoint": "target_recall",
                        "count": k, "runs": n, "rate": k / n, "ci_low": lo, "ci_high": hi,
                        "passed": bool(k / n >= 0.8), "off_target_rate": float(g["off_target_selected"].mean())})
    return pd.DataFrame(out)


def summarise_mitigation(mit: pd.DataFrame) -> pd.DataFrame:
    if mit.empty:
        return mit
    keep = ["Baseline (audit labels)", "Audit-guided reweighting", "Drop flagged block",
            "Trusted reference only", "Oracle (clean audit labels)"]
    mit = mit[mit["strategy"].isin(keep)]
    agg = mit.groupby(["dataset", "condition", "strategy", "model"], sort=False).agg(
        reps=("rep", "nunique"), auc=("auc", "mean"), eo_gap=("eo_gap", "mean"), dp_gap=("dp_gap", "mean"),
        group_pred_bias=("group_pred_bias", "mean"), auc_change=("auc_change", "mean"),
        eo_gap_change=("eo_gap_change", "mean")).reset_index()
    rw = mit[mit["strategy"] == "Audit-guided reweighting"].copy()
    rw["meets_criteria"] = (rw["eo_gap_change"] < 0) & (rw["auc_change"] >= -0.02) & (rw["ess"] >= 0.7)
    crit = rw.groupby(["dataset", "condition", "model"])["meets_criteria"].mean().rename("share_meeting_criteria")
    return agg.merge(crit.reset_index(), on=["dataset", "condition", "model"], how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--B", type=int, default=1999)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    tasks = [(ds, i, c, rate, sh, r) for ds, conds in CONDITIONS.items()
             for i, (c, rate, sh) in enumerate(conds) for r in range(args.reps)]
    print(f"real-data study: {len(tasks)} splits")
    t0 = time.time()
    res = Parallel(n_jobs=args.jobs, verbose=5)(delayed(one_rep)(*t, args.B) for t in tasks)
    runs = pd.DataFrame([row for rr, _ in res for row in rr])
    mit = pd.DataFrame([row for _, mm in res for row in mm])
    os.makedirs(args.out, exist_ok=True)
    runs.to_csv(os.path.join(args.out, "realdata_runs.csv"), index=False)
    mit.to_csv(os.path.join(args.out, "mitigation_runs.csv"), index=False)
    s = summarise(runs)
    s.to_csv(os.path.join(args.out, "realdata_summary.csv"), index=False)
    ms = summarise_mitigation(mit)
    ms.to_csv(os.path.join(args.out, "mitigation_summary.csv"), index=False)
    meta = {"reps": args.reps, "B": args.B, "splits": len(tasks), "elapsed_s": round(time.time() - t0, 1),
            "conditions": {k: [c[0] for c in v] for k, v in CONDITIONS.items()}}
    with open(os.path.join(args.out, "realdata_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    pd.set_option("display.width", 220)
    print(s.to_string(index=False))
    print(ms.round(3).to_string(index=False))
    print(meta)


if __name__ == "__main__":
    main()
