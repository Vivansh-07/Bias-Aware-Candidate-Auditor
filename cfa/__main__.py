"""Command-line auditor.

Audit a CSV that already has a trusted-vs-audit source column:
    python -m cfa --csv my.csv --label outcome --source source

Run a built-in semi-synthetic demonstration:
    python -m cfa --dataset COMPAS --corruption 0.3
    python -m cfa --synthetic "Label corruption (30%)"
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .realdata import DATASETS, CorruptionSpec, feature_columns, load_dataset, make_semisynthetic
from .simulate import PRESETS, feature_names, generate
from .v5 import audit_v5
from .v6 import AuditConfig, audit_v6


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cfa", description="Candidate feature auditor (V6)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="CSV with features, a binary label and a 0/1 source column")
    src.add_argument("--dataset", choices=list(DATASETS), help="built-in dataset (semi-synthetic split)")
    src.add_argument("--synthetic", choices=list(PRESETS), help="synthetic preset")
    ap.add_argument("--label", help="label column (CSV mode)")
    ap.add_argument("--source", default="source", help="source column, 0 = trusted, 1 = audit (CSV mode)")
    ap.add_argument("--corruption", type=float, default=0.0, help="injected corruption rate (dataset mode)")
    ap.add_argument("--method", choices=["v6", "v5"], default="v6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="write the full audit report to this path")
    args = ap.parse_args(argv)

    if args.csv:
        if not args.label:
            ap.error("--label is required with --csv")
        df = pd.read_csv(args.csv)
        vals = sorted(df[args.label].dropna().unique())
        if len(vals) != 2:
            ap.error("label must be binary")
        df["y"] = (df[args.label] == vals[1]).astype(int)
        df = df.drop(columns=[args.label]).rename(columns={args.source: "source"})
        feats = feature_columns(df)
    elif args.dataset:
        info = DATASETS[args.dataset]
        corr = CorruptionSpec(info.target_feature, info.target_values, info.flip_from, info.flip_to,
                              args.corruption) if args.corruption > 0 else None
        df, _ = make_semisynthetic(load_dataset(args.dataset), info.label, corruption=corr, seed=args.seed)
        feats = feature_columns(df)
    else:
        sc = PRESETS[args.synthetic]
        df = generate(sc, seed=args.seed)
        feats = feature_names(sc)

    cfg = AuditConfig(seed=args.seed)
    res = audit_v6(df, feats, config=cfg) if args.method == "v6" else audit_v5(df, feats, config=cfg)
    cols = ["exploratory_rank", "block", "supported", "p_block", "q_BY", "selected"]
    print(f"\n{res.method} decision: {res.decision}\n")
    print(res.blocks[cols].to_string(index=False))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res.to_dict(), fh, indent=2, default=str)
        print(f"\nFull report written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
