"""Monte Carlo studies for the reimplemented auditors.

calibration : clean (no-corruption) conditions -> alarm rate with exact 95% CI,
              gate = upper endpoint < 5% in every condition; random-label
              condition is scored as abstention (gate = lower endpoint >= 90%).
power       : corruption grid -> recall of the block containing the injected
              feature `a`, plus off-target selections.

Every method is applied to the SAME dataset per repetition (paired design).

Usage
  python experiments/run_study.py --study calibration --reps 500 --jobs 10
  python experiments/run_study.py --study power --reps 200 --jobs 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cfa.simulate import CALIBRATION_CONDITIONS, DEFAULT, feature_names, generate  # noqa: E402
from cfa.stats import abstention_gate, calibration_gate  # noqa: E402
from cfa.v5 import audit_v5  # noqa: E402
from cfa.v6 import AuditConfig, audit_v6  # noqa: E402

POWER_CONDITIONS = {
    f"c{int(c * 100)}_noise{int(pn * 100)}": replace(DEFAULT, corruption=c, proxy_noise=pn)
    for c in (0.10, 0.20, 0.30, 0.40) for pn in (0.0, 0.10, 0.30)
}
POWER_CONDITIONS.update({
    "c30_shift150": replace(DEFAULT, corruption=0.30, shift=1.50),
    "c30_n10000": replace(DEFAULT, corruption=0.30, n_audit=10000, n_trusted=3333),
    "c20_n10000": replace(DEFAULT, corruption=0.20, n_audit=10000, n_trusted=3333),
})

STUDY_OFFSET = {"calibration": 7_000_000, "power": 9_000_000}


def one_run(study: str, cond_idx: int, cond: str, sc, rep: int, methods: list[str], B: int) -> list[dict]:
    seed = STUDY_OFFSET[study] + 10_000 * cond_idx + rep
    df = generate(sc, seed=seed)
    feats = feature_names(sc)
    cfg = AuditConfig(seed=seed)
    rows = []
    for m in methods:
        t = time.time()
        res = audit_v6(df, feats, config=cfg) if m == "v6" else audit_v5(df, feats, config=cfg, B=B)
        sel = [b for b in res.selected_blocks]
        rows.append({
            "study": study, "condition": cond, "rep": rep, "seed": seed, "method": m.upper(),
            "alarm": int(res.flagged), "n_selected": len(sel),
            "target_selected": int(res.selects("a")),
            "off_target_selected": int(any("a" not in b for b in sel)),
            "selected": json.dumps(sel), "seconds": round(time.time() - t, 3),
        })
    return rows


def summarise(runs: pd.DataFrame, study: str) -> pd.DataFrame:
    out = []
    for (cond, method), g in runs.groupby(["condition", "method"], sort=False):
        n = len(g)
        if study == "calibration":
            if cond == "random_labels":
                gate = abstention_gate(int(n - g["alarm"].sum()), n)
                out.append({"condition": cond, "method": method, "endpoint": "abstention",
                            "count": gate["abstentions"], "runs": n, "rate": gate["rate"],
                            "ci_low": gate["ci_low"], "ci_high": gate["ci_high"], "passed": gate["passed"]})
            else:
                gate = calibration_gate(int(g["alarm"].sum()), n)
                out.append({"condition": cond, "method": method, "endpoint": "clean_alarm",
                            "count": gate["alarms"], "runs": n, "rate": gate["rate"],
                            "ci_low": gate["ci_low"], "ci_high": gate["ci_high"], "passed": gate["passed"]})
        else:
            k = int(g["target_selected"].sum())
            from cfa.stats import clopper_pearson
            lo, hi = clopper_pearson(k, n)
            out.append({"condition": cond, "method": method, "endpoint": "target_recall",
                        "count": k, "runs": n, "rate": k / n, "ci_low": lo, "ci_high": hi,
                        "passed": bool(k / n >= 0.80),
                        "off_target_rate": float(g["off_target_selected"].mean())})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["calibration", "power"], required=True)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--methods", default="v6,v5")
    ap.add_argument("--conditions", default="all")
    ap.add_argument("--B", type=int, default=1999)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    pool = CALIBRATION_CONDITIONS if args.study == "calibration" else POWER_CONDITIONS
    names = list(pool) if args.conditions == "all" else args.conditions.split(",")
    methods = args.methods.split(",")
    tasks = [(i, c, pool[c], r) for i, c in enumerate(pool) if c in names for r in range(args.reps)]
    print(f"{args.study}: {len(names)} conditions x {args.reps} reps x {methods} -> {len(tasks)} datasets")

    t0 = time.time()
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(one_run)(args.study, i, c, sc, r, methods, args.B) for i, c, sc, r in tasks)
    runs = pd.DataFrame([row for rows in results for row in rows])
    os.makedirs(args.out, exist_ok=True)
    runs.to_csv(os.path.join(args.out, f"{args.study}_runs.csv"), index=False)
    summary = summarise(runs, args.study)
    summary.to_csv(os.path.join(args.out, f"{args.study}_summary.csv"), index=False)
    meta = {"study": args.study, "reps": args.reps, "methods": methods, "conditions": names,
            "B": args.B, "datasets": len(tasks), "elapsed_s": round(time.time() - t0, 1)}
    with open(os.path.join(args.out, f"{args.study}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print(meta)


if __name__ == "__main__":
    main()
