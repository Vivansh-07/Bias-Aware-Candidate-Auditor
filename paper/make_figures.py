"""Vector figures sized for an IEEE two-column paper (column width 3.5 in).

Usage:  python paper/make_figures.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(HERE, "figures")
COLORS = {"V6": "#2a78d6", "V5": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL_W = 3.45

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7.5,
    "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False, "axes.linewidth": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})

PRETTY = {
    "default": "default", "proxy_noise_0": "proxy noise 0%", "proxy_noise_30": "proxy noise 30%",
    "proxy_noise_50": "proxy noise 50%", "n_audit_1000": "audit n = 1,000", "n_audit_10000": "audit n = 10,000",
    "trusted_833": "trusted n = 833", "trusted_3333": "trusted n = 3,333", "prevalence_10": "prevalence 10%",
    "prevalence_50": "prevalence 50%", "noise_cols_20": "20 noise columns", "nonlinear": "nonlinear labels",
    "missing_30": "30% missingness", "correlated_nuisance": "correlated nuisance",
    "shift_0.75": "shift 0.75", "shift_1.50": "shift 1.50", "shift_1.50_nonlinear": "shift 1.50 + nonlinear",
    "shift_1.50_corr_nuisance": "shift 1.50 + corr. nuisance", "shift_1.50_noise_20": "shift 1.50 + 20 noise",
    "shift_1.50_prev_10": "shift 1.50 + prev. 10%",
}


def calibration():
    cal = pd.read_csv(os.path.join(RES, "calibration_summary.csv"))
    clean = cal[cal["endpoint"] == "clean_alarm"]
    conds = list(dict.fromkeys(clean["condition"]))
    cap = 0.12
    fig, ax = plt.subplots(figsize=(COL_W, 3.05))
    for m, off in (("V6", -0.17), ("V5", 0.17)):
        g = clean[clean["method"] == m].set_index("condition").loc[conds]
        y = np.arange(len(conds)) + off
        rate, lo, hi = g["rate"].clip(upper=cap), g["ci_low"].clip(upper=cap), g["ci_high"].clip(upper=cap)
        ax.errorbar(rate, y, xerr=[rate - lo, hi - rate], fmt="o", ms=2.6, color=COLORS[m],
                    elinewidth=1.1, capsize=0, label=m)
        for yi, r, h in zip(y, g["rate"], g["ci_high"]):
            if h > cap:
                ax.annotate(f"{r:.1%} →", (cap, yi), xytext=(-2, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=6.5, color=COLORS[m], fontweight="bold")
    ax.axvline(0.05, color=INK, lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(np.arange(len(conds)), [PRETTY.get(c, c) for c in conds])
    ax.set_ylim(len(conds) - 0.5, -0.6)
    ax.set_xlim(0, cap)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("clean alarm rate, exact 95% CI (500 datasets each)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", handletextpad=0.2)
    fig.savefig(os.path.join(OUT, "fig_calibration.pdf"))
    plt.close(fig)


def power():
    pw = pd.read_csv(os.path.join(RES, "power_summary.csv"))
    grid = pw[pw["condition"].str.match(r"c\d+_noise\d+$")].copy()
    grid["c"] = grid["condition"].str.extract(r"c(\d+)_", expand=False).astype(int)
    grid["noise"] = grid["condition"].str.extract(r"noise(\d+)", expand=False).astype(int)
    noises = sorted(grid["noise"].unique())
    fig, axes = plt.subplots(1, len(noises), figsize=(COL_W, 1.45), sharey=True)
    for ax, n in zip(axes, noises):
        for m in ("V6", "V5"):
            g = grid[(grid["noise"] == n) & (grid["method"] == m)].sort_values("c")
            ax.fill_between(g["c"], g["ci_low"], g["ci_high"], color=COLORS[m], alpha=0.15, lw=0)
            ax.plot(g["c"], g["rate"], "-o", color=COLORS[m], lw=1.3, ms=2.6, label=m)
        ax.axhline(0.8, color=INK, lw=0.8, ls=(0, (3, 2)))
        ax.set_title(f"proxy noise {n}%", pad=2)
        ax.set_xticks([10, 20, 30, 40], ["10", "20", "30", "40"])
        ax.set_xlabel("corruption (%)", labelpad=1)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0].set_ylabel("target-block recall")
    axes[0].set_ylim(0, 1.03)
    axes[0].legend(loc="lower right", handlelength=1.2, borderaxespad=0.1)
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(os.path.join(OUT, "fig_power.pdf"))
    plt.close(fig)


def realdata():
    rd = pd.read_csv(os.path.join(RES, "realdata_summary.csv"))
    d = rd[rd["dataset"] == "COMPAS"]
    v2_path = os.path.join(RES, "realdata_cal_v2_summary.csv")
    if os.path.exists(v2_path):
        # panel (a) from the pre-registered 1,000-split calibration study
        v2 = pd.read_csv(v2_path)
        v2 = v2[v2["dataset"] == "COMPAS"].assign(endpoint="clean_alarm")
        d = pd.concat([v2, d[d["endpoint"] != "clean_alarm"]], ignore_index=True)
    label = {"clean": "clean", "shift_age_1.0": "age shift 1.0", "shift_age_1.5": "age shift 1.5",
             "corrupt_10": "10%", "corrupt_20": "20%", "corrupt_30": "30%", "corrupt_30_shift_1.0": "30% + shift"}
    parts = [("clean_alarm", "(a) clean / shift: alarm rate", 0.05, 0.065),
             ("target_recall", "(b) corruption: recall of race", 0.80, 1.0)]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.55), gridspec_kw={"width_ratios": [1, 1.25]})
    for ax, (endpoint, title, line, xmax) in zip(axes, parts):
        sub = d[d["endpoint"] == endpoint]
        conds = list(dict.fromkeys(sub["condition"]))
        for m, off in (("V6", -0.19), ("V5", 0.19)):
            g = sub[sub["method"] == m].set_index("condition").loc[conds]
            y = np.arange(len(conds)) + off
            ax.barh(y, g["rate"], height=0.36, color=COLORS[m], label=m)
            ax.errorbar(g["rate"], y, xerr=[g["rate"] - g["ci_low"], g["ci_high"] - g["rate"]],
                        fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.2)
        ax.axvline(line, color=INK, lw=0.8, ls=(0, (3, 2)))
        ax.set_yticks(np.arange(len(conds)), [label[c] for c in conds])
        ax.set_ylim(len(conds) - 0.5, -0.5)
        ax.set_xlim(0, xmax)
        ax.set_xticks([0, 0.02, 0.04, 0.06] if endpoint == "clean_alarm" else [0, 0.5, 1.0])
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_title(title, pad=2)
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower right", handlelength=1.0, borderaxespad=0.1)
    fig.subplots_adjust(wspace=0.55)
    fig.savefig(os.path.join(OUT, "fig_realdata.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    calibration()
    power()
    realdata()
    print("figures written to", OUT)
