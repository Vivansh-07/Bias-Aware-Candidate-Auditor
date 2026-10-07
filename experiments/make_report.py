"""Build paper-ready figures and docs/RESULTS.md from the saved study outputs.

Usage:  python experiments/make_report.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "docs", "figures")
COLORS = {"V6": "#2a78d6", "V5": "#eb6834"}   # categorical slots 1-2 (validated adjacent pair)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _read(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def pct(x):
    return f"{100 * x:.2f}%"


def fig_calibration(cal: pd.DataFrame) -> str:
    clean = cal[cal["endpoint"] == "clean_alarm"].copy()
    conds = list(dict.fromkeys(clean["condition"]))
    fig, ax = plt.subplots(figsize=(6.2, 0.28 * len(conds) + 1.0))
    for k, m in enumerate(["V6", "V5"]):
        g = clean[clean["method"] == m].set_index("condition").loc[conds]
        y = np.arange(len(conds)) + (-0.17 if m == "V6" else 0.17)
        ax.errorbar(g["rate"], y, xerr=[g["rate"] - g["ci_low"], g["ci_high"] - g["rate"]], fmt="o",
                    ms=4, color=COLORS[m], ecolor=COLORS[m], elinewidth=1.4, capsize=0, label=m)
    ax.axvline(0.05, color=INK, lw=1, ls=(0, (4, 3)))
    ax.text(0.051, -0.9, "5% gate", color=INK, fontsize=8, va="bottom")
    ax.set_yticks(np.arange(len(conds)), conds)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Clean alarm rate with exact two-sided 95% CI (500 datasets per condition)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    path = os.path.join(FIG, "fig_calibration.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_power(pw: pd.DataFrame) -> str:
    grid = pw[pw["condition"].str.match(r"c\d+_noise\d+$")].copy()
    grid["c"] = grid["condition"].str.extract(r"c(\d+)_").astype(int)
    grid["noise"] = grid["condition"].str.extract(r"noise(\d+)").astype(int)
    noises = sorted(grid["noise"].unique())
    fig, axes = plt.subplots(1, len(noises), figsize=(6.6, 2.3), sharey=True)
    for ax, n in zip(axes, noises):
        for m in ["V6", "V5"]:
            g = grid[(grid["noise"] == n) & (grid["method"] == m)].sort_values("c")
            ax.plot(g["c"], g["rate"], "-o", color=COLORS[m], lw=2, ms=4, label=m)
            ax.fill_between(g["c"], g["ci_low"], g["ci_high"], color=COLORS[m], alpha=0.12, lw=0)
        ax.axhline(0.8, color=INK, lw=1, ls=(0, (4, 3)))
        ax.set_title(f"proxy noise {n}%", fontsize=9, color=INK)
        ax.set_xticks(sorted(grid["c"].unique()), [f"{c}%" for c in sorted(grid["c"].unique())])
        ax.set_xlabel("corruption rate")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_ylabel("target-block recall")
    axes[0].set_ylim(0, 1.02)
    axes[-1].legend(loc="lower right")
    path = os.path.join(FIG, "fig_power.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_realdata(rd: pd.DataFrame, dataset: str) -> str:
    d = rd[rd["dataset"] == dataset].copy()
    parts = [("clean_alarm", "Clean / shift: alarm rate (gate 5%)", 0.05, ""),
             ("target_recall", "Corruption: recall (target 80%)", 0.80, "")]
    sizes = [max(1, d[d["endpoint"] == e]["condition"].nunique()) for e, *_ in parts]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.42 * max(sizes) + 1.1))
    for ax, (endpoint, title, line, line_label) in zip(axes, parts):
        sub = d[d["endpoint"] == endpoint]
        conds = list(dict.fromkeys(sub["condition"]))
        for m in ["V6", "V5"]:
            g = sub[sub["method"] == m].set_index("condition").loc[conds]
            y = np.arange(len(conds)) + (-0.19 if m == "V6" else 0.19)
            ax.barh(y, g["rate"], height=0.36, color=COLORS[m], label=m)
            ax.errorbar(g["rate"], y, xerr=[g["rate"] - g["ci_low"], g["ci_high"] - g["rate"]],
                        fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
        ax.axvline(line, color=INK, lw=1, ls=(0, (4, 3)))
        ax.set_yticks(np.arange(len(conds)), conds)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("rate with exact 95% CI")
        ax.grid(axis="y", visible=False)
        if endpoint == "target_recall":
            ax.set_xlim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    path = os.path.join(FIG, f"fig_realdata_{dataset.replace(' ', '_').lower()}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_mitigation(ms: pd.DataFrame, dataset: str, cond: str) -> str | None:
    d = ms[(ms["dataset"] == dataset) & (ms["condition"] == cond)]
    if d.empty:
        return None
    order = ["Baseline (audit labels)", "Audit-guided reweighting", "Drop flagged block",
             "Trusted reference only", "Oracle (clean audit labels)"]
    order = [o for o in order if o in set(d["strategy"])]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=True)
    for ax, metric, title in [(axes[0], "group_pred_bias", "prediction bias for corrupted group"),
                              (axes[1], "eo_gap", "equalized-odds gap")]:
        for k, (model, color) in enumerate([("Logistic regression", "#2a78d6"), ("Random forest", "#1baf7a")]):
            g = d[d["model"] == model].set_index("strategy").loc[order]
            y = np.arange(len(order)) + (-0.18 if k == 0 else 0.18)
            ax.barh(y, g[metric], height=0.34, color=color, label=model)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(title, fontsize=9, color=INK)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(order)), order)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.6, -0.08))
    fig.tight_layout()
    path = os.path.join(FIG, f"fig_mitigation_{dataset.replace(' ', '_').lower()}_{cond}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    os.makedirs(FIG, exist_ok=True)
    cal, pw = _read("calibration_summary.csv"), _read("power_summary.csv")
    rd, ms = _read("realdata_summary.csv"), _read("mitigation_summary.csv")
    out = ["# Results of the reimplemented auditors",
           "",
           "Auto-generated by `experiments/make_report.py` from the CSV files in `results/`. "
           "These are results of this repository's independent reimplementation of the V5 and V6 "
           "auditors from the project's written method; they are **not** the archived V1–V6 study "
           "numbers and should be reported as a separate study.",
           "",
           "**Protocol notes (for honest reporting).** The V5/V6 code, gates and condition list were "
           "written before the main runs. A 100-repetition development pilot on four conditions (default, "
           "shift 1.50, nonlinear, shift 1.50 + nonlinear) was run first. After the pilot, two trusted-sample-size "
           "conditions (833 and 3,333) were added, and no auditor code was changed. Seeds are deterministic "
           "(`experiments/run_study.py`), so the pilot reused some main-study streams for those conditions. "
           "The condition set was chosen by this project, and these runs are not a locked, independent "
           "confirmation study.",
           ""]
    if cal is not None:
        meta = json.load(open(os.path.join(RES, "calibration_meta.json")))
        fig_calibration(cal)
        clean = cal[cal["endpoint"] == "clean_alarm"]
        out += ["## 1. Calibration (no corruption)", "",
                f"{meta['datasets']:,} synthetic datasets ({meta['reps']} per condition), every method on the same data, "
                f"V5 with B = {meta['B']} permutations. Gate: exact two-sided 95% upper bound < 5% in every clean "
                "condition; random-label abstention lower bound ≥ 90%.", ""]
        for m, g in clean.groupby("method"):
            out.append(f"- **{m}**: {int(g['passed'].sum())}/{len(g)} clean conditions pass; "
                       f"alarm rates {pct(g['rate'].min())}–{pct(g['rate'].max())}; "
                       f"largest upper bound {pct(g['ci_high'].max())}.")
        rl = cal[cal["endpoint"] == "abstention"]
        for _, r in rl.iterrows():
            out.append(f"- **{r['method']}** random labels: abstained {r['count']}/{r['runs']} "
                       f"(lower bound {pct(r['ci_low'])}) → {'pass' if r['passed'] else 'fail'}.")
        t = cal.copy()
        t["rate"] = t["rate"].map(pct)
        t["95% CI"] = [f"[{pct(a)}, {pct(b)}]" for a, b in zip(cal["ci_low"], cal["ci_high"])]
        t["gate"] = t["passed"].map({True: "pass", False: "FAIL"})
        out += ["", "![calibration](figures/fig_calibration.png)", "",
                md_table(t[["condition", "method", "endpoint", "count", "runs", "rate", "95% CI", "gate"]]), ""]
    if pw is not None:
        fig_power(pw)
        t = pw.copy()
        t["recall"] = t["rate"].map(pct)
        t["95% CI"] = [f"[{pct(a)}, {pct(b)}]" for a, b in zip(pw["ci_low"], pw["ci_high"])]
        t["off-target"] = pw["off_target_rate"].map(pct)
        t["≥80%"] = t["passed"].map({True: "yes", False: "no"})
        out += ["## 2. Power (corruption injected)", "",
                "Recall = share of datasets in which the block containing the injected group `a` is selected. "
                "Off-target = share in which some other block is also selected. Exploratory: the protocol "
                "requires every calibration gate to pass before power is interpreted.", "",
                "![power](figures/fig_power.png)", "",
                md_table(t[["condition", "method", "count", "runs", "recall", "95% CI", "≥80%", "off-target"]]), ""]
    if rd is not None:
        meta = json.load(open(os.path.join(RES, "realdata_meta.json")))
        out += ["## 3. Real-data transfer (semi-synthetic)", "",
                f"{meta['splits']:,} record-disjoint random splits ({meta['reps']} per condition). Clean and shift rows "
                "report alarm rates; corrupt rows report recall of the corrupted attribute.", ""]
        for ds in rd["dataset"].unique():
            fig_realdata(rd, ds)
            slug = ds.replace(" ", "_").lower()
            t = rd[rd["dataset"] == ds].copy()
            t["rate"] = t["rate"].map(pct)
            t["95% CI"] = [f"[{pct(a)}, {pct(b)}]" for a, b in zip(rd.loc[t.index, "ci_low"], rd.loc[t.index, "ci_high"])]
            out += [f"### {ds}", "", f"![{ds}](figures/fig_realdata_{slug}.png)", "",
                    md_table(t[["condition", "method", "endpoint", "count", "runs", "rate", "95% CI"]]), ""]
    if ms is not None and not ms.empty:
        out += ["## 4. Audit-guided mitigation (only in splits where V6 flagged)", "",
                "Means over flagged splits; fairness measured on clean held-out test labels for the corrupted "
                "group. `share_meeting_criteria` = fraction of splits where reweighting lowered the EO gap, lost "
                "≤ 0.02 AUC and kept ESS ≥ 70%.", ""]
        for (ds, cond), _ in ms.groupby(["dataset", "condition"], sort=False):
            p = fig_mitigation(ms, ds, cond)
            t = ms[(ms["dataset"] == ds) & (ms["condition"] == cond)].copy()
            t = t[["strategy", "model", "reps", "auc", "eo_gap", "dp_gap", "group_pred_bias",
                   "auc_change", "eo_gap_change", "share_meeting_criteria"]].round(3)
            out += [f"### {ds} · {cond}", ""]
            if p:
                out += [f"![mitigation](figures/{os.path.basename(p)})", ""]
            out += [md_table(t.fillna("")), ""]
    with open(os.path.join(ROOT, "docs", "RESULTS.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print("wrote docs/RESULTS.md and", len(os.listdir(FIG)), "figures")


if __name__ == "__main__":
    main()
