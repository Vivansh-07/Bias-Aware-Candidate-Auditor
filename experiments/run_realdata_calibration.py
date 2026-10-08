"""Pre-registered real-data calibration study v2 (see protocols/realdata_calibration_v2.md).

Usage:  python experiments/run_realdata_calibration.py --reps 1000 --jobs 10
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cfa.realdata import DATASETS, ShiftSpec, feature_columns, load_dataset, make_semisynthetic  # noqa: E402
from cfa.stats import calibration_gate  # noqa: E402
from cfa.v5 import audit_v5  # noqa: E402
from cfa.v6 import AuditConfig, audit_v6  # noqa: E402

CONDITIONS = {
    "COMPAS": [("clean", 0.0), ("shift_age_1.0", 1.0), ("shift_age_1.5", 1.5)],
    "German Credit": [("clean", 0.0), ("shift_age_1.0", 1.0)],
}
SEED_BASE = 5_000_000


def one_split(dataset: str, cond_idx: int, cond: str, shift: float, rep: int, B: int,
              seed_base: int = SEED_BASE) -> list[dict]:
    info = DATASETS[dataset]
    seed = seed_base + 100_000 * list(DATASETS).index(dataset) + 10_000 * cond_idx + rep
    sh = ShiftSpec(info.shift_feature, shift) if shift else None
    data, _ = make_semisynthetic(load_dataset(dataset), info.label, shift=sh, seed=seed)
    feats = feature_columns(data)
    cfg = AuditConfig(seed=seed)
    rows = []
    for res in (audit_v6(data, feats, config=cfg), audit_v5(data, feats, config=cfg, B=B)):
        rows.append({"dataset": dataset, "condition": cond, "rep": rep, "seed": seed, "method": res.method,
                     "alarm": int(res.flagged), "selected": json.dumps(res.selected_blocks)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--B", type=int, default=1999)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--seed-base", type=int, default=SEED_BASE, help="only change for code smoke tests")
    args = ap.parse_args()
    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    tasks = [(ds, i, c, sh, r) for ds, conds in CONDITIONS.items()
             for i, (c, sh) in enumerate(conds) for r in range(args.reps)]
    print(f"real-data calibration v2: {len(tasks)} splits at commit {commit[:8]}")
    t0 = time.time()
    out = Parallel(n_jobs=args.jobs, verbose=5)(delayed(one_split)(*t, args.B, args.seed_base) for t in tasks)
    runs = pd.DataFrame([row for rows in out for row in rows])
    runs.to_csv(os.path.join(args.out, "realdata_cal_v2_runs.csv"), index=False)

    summary = []
    for (ds, cond), g in runs.groupby(["dataset", "condition"], sort=False):
        w = g.pivot(index="rep", columns="method", values="alarm")
        b = int(((w["V5"] == 1) & (w["V6"] == 0)).sum())
        c = int(((w["V5"] == 0) & (w["V6"] == 1)).sum())
        p_mcnemar = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        for m in ("V6", "V5"):
            gate = calibration_gate(int(w[m].sum()), len(w))
            summary.append({"dataset": ds, "condition": cond, "method": m, **gate,
                            "v5_only": b, "v6_only": c, "p_mcnemar": p_mcnemar})
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(args.out, "realdata_cal_v2_summary.csv"), index=False)
    meta = {"protocol": "experiments/protocols/realdata_calibration_v2.md", "code_commit": commit,
            "reps": args.reps, "B": args.B, "splits": len(tasks), "elapsed_s": round(time.time() - t0, 1),
            "v6_gate_all_conditions": bool(summary[summary.method == "V6"]["passed"].all())}
    with open(os.path.join(args.out, "realdata_cal_v2_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    pd.set_option("display.width", 220)
    print(summary.to_string(index=False))
    print(meta)


if __name__ == "__main__":
    main()
