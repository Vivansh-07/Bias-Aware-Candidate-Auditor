"""Candidate Feature Auditor - interactive working model (DSE3170 PBL-3).

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import os
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from cfa.mitigation import audit_guided_weights, evaluate_strategies
from cfa.realdata import (DATASETS, CorruptionSpec, ShiftSpec, feature_columns, load_dataset,
                          make_semisynthetic)
from cfa.simulate import CALIBRATION_CONDITIONS, PRESETS, Scenario, feature_names, generate, generate_test
from cfa.stats import abstention_gate, calibration_gate
from cfa.v5 import audit_v5
from cfa.v6 import AuditConfig, AuditResult, audit_v6

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")

st.set_page_config(page_title="Candidate Feature Auditor", page_icon="🔎", layout="wide")


# =====================================================================  helpers
def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "–"
    if p >= 0.999:
        return "1"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def decision_banner(res: AuditResult, label: str):
    if res.flagged:
        blocks = "; ".join("{" + ", ".join(b) + "}" for b in res.selected_blocks)
        st.warning(f"🚩 **{label}: flag for human review →** {blocks}")
        st.caption("A flag is an investigation prompt for an association with unusual source–label "
                   "behaviour. It is not proof of bias, not a causal finding and not an instruction to delete data.")
    else:
        st.info(f"✋ **{label}: abstain.** No statistically supported candidate detected.")
        st.caption("Abstention means the support and significance gates were not met. "
                   "It does not certify that the data are free of bias.")


def block_table(res: AuditResult) -> pd.DataFrame:
    b = res.blocks.copy()
    out = pd.DataFrame({"Rank (exploratory)": b["exploratory_rank"], "Block": b["block"]})
    if res.method == "V6":
        out["Contrasts"] = b["n_contrasts"]
        out["Strongest contrast"] = b["best_contrast"]
        out["Raw p (min)"] = b["p_min_raw"].map(fmt_p)
        out["Bonferroni p"] = b["p_block"].map(fmt_p)
    else:
        out["Coherent score T"] = b["T_block"].round(4)
        out["Permutation p"] = b["p_block"].map(fmt_p)
    out["BY q"] = b["q_BY"].map(fmt_p)
    out["Supported"] = b["supported"].map({True: "yes", False: "no"})
    out["Selected"] = b["selected"].map({True: "🚩 FLAG", False: "–"})
    return out


def block_chart(res: AuditResult):
    b = res.blocks.copy()
    raw = b["p_min_raw"] if res.method == "V6" else b["p_block"]
    b["evidence"] = -np.log10(raw.clip(lower=1e-16))
    b["status"] = np.where(b["selected"], "flagged (BY q < 0.05)",
                           np.where(b["supported"], "not significant after correction", "unsupported"))
    bars = alt.Chart(b).mark_bar().encode(
        x=alt.X("evidence:Q", title="Raw evidence, −log10(uncorrected p)"),
        y=alt.Y("block:N", sort="-x", title=None),
        color=alt.Color("status:N", scale=alt.Scale(
            domain=["flagged (BY q < 0.05)", "not significant after correction", "unsupported"],
            range=["#e4572e", "#8d99ae", "#d9d9d9"]), legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["block", alt.Tooltip("evidence:Q", format=".2f"), alt.Tooltip("q_BY:Q", format=".2e"), "supported"],
    )
    line = pd.DataFrame({"x": [-np.log10(0.05)], "label": ["naive p = 0.05"]})
    rule = alt.Chart(line).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
    text = alt.Chart(line).mark_text(align="left", dx=4, dy=-6, fontSize=10).encode(
        x="x:Q", y=alt.value(0), text="label:N")
    return (bars + rule + text).properties(
        height=130 + 26 * len(b),
        title=f"{res.method}: ranking vs detection")


def group_chart(res: AuditResult, feature: str):
    if res.groups.empty:
        return None
    g = res.groups[res.groups["feature"] == feature].copy()
    if g.empty:
        return None
    if res.method == "V6":
        g["value"] = g["label_gap"]
        title = "Adjusted label gap, audit − trusted (95% CI)"
        base = alt.Chart(g).encode(y=alt.Y("group:N", title=None, sort=None))
        bars = base.mark_bar().encode(
            x=alt.X("value:Q", title=title),
            color=alt.condition("datum.value < 0", alt.value("#3a86ff"), alt.value("#e4572e")),
            tooltip=["group", "n_trusted", "n_audit", alt.Tooltip("label_gap:Q", format=".3f"),
                     alt.Tooltip("trusted_rate:Q", format=".3f"), alt.Tooltip("audit_rate:Q", format=".3f")])
        err = base.mark_errorbar().encode(x=alt.X("label_gap_lo:Q", title=title), x2="label_gap_hi:Q")
        chart = bars + err
    else:
        g["value"] = g["coherent_diff"]
        chart = alt.Chart(g).mark_bar().encode(
            y=alt.Y("group:N", title=None, sort=None),
            x=alt.X("value:Q", title="Coherent difference Σ w(s)·d(s,g)"),
            color=alt.condition("datum.value < 0", alt.value("#3a86ff"), alt.value("#e4572e")),
            tooltip=["group", "n_trusted", "n_audit", alt.Tooltip("value:Q", format=".3f")])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule().encode(x="x:Q")
    return (chart + zero).properties(height=110 + 42 * len(g), title=f"{res.method} · {feature}")


def render_result(res: AuditResult, features: list[str], key: str):
    decision_banner(res, res.method)
    d = res.diagnostics
    c = st.columns(4)
    c[0].metric("Feature blocks", d["n_blocks"])
    c[1].metric("Trusted analysis rows", d["n_analysis_trusted"])
    c[2].metric("Audit rows", d["n_analysis_audit"])
    if res.method == "V6":
        c[3].metric("Source-model AUC", f"{d['source_model_auc']:.3f}",
                    help="How well features predict trusted vs audit membership. 0.5 = no population "
                         "difference; higher = population shift that V6 adjusts for.")
    else:
        c[3].metric("Min attainable p", f"{d['min_attainable_p']:.4f}",
                    help="With B permutations the smallest p-value is 1/(B+1).")
    st.altair_chart(block_chart(res), use_container_width=True)
    st.dataframe(block_table(res), hide_index=True, width="stretch")
    default = res.lead_feature(res.selected_blocks[0]) if res.flagged else res.lead_feature(res.blocks.iloc[0]["members"])
    feat = st.selectbox("Inspect candidate groups for feature", features,
                        index=features.index(default) if default in features else 0, key=f"feat_{key}")
    ch = group_chart(res, feat)
    if ch is not None:
        st.altair_chart(ch, use_container_width=True)
    with st.expander("Raw tables (contrasts / groups / diagnostics)"):
        st.dataframe(res.contrasts, width="stretch")
        st.dataframe(res.groups, width="stretch")
        st.json({k: v for k, v in d.items() if k != "association"})
    st.download_button("Download audit report (JSON)", json.dumps(res.to_dict(), indent=2, default=str),
                       file_name=f"audit_{res.method}_{key}.json", mime="application/json", key=f"dl_{key}")


def render_mitigation(data, test, feats, res: AuditResult, protected, protected_label: str, key: str):
    st.subheader("Stage 3 · Audit-guided mitigation (separate from detection)")
    st.caption(f"Fairness is measured on a held-out test set with **clean** labels, for the group "
               f"**{protected_label}** versus everyone else. Proposed acceptance criteria from the report: "
               f"EO gap reduced, AUC loss ≤ 0.02, effective sample size ≥ 70%.")
    if not res.flagged:
        st.info("Mitigation gate closed: the auditor abstained, so no intervention is proposed. "
                "You can still compare the baseline with the trusted-only and oracle models.")
    if st.button("Train & evaluate downstream models", key=f"mit_{key}"):
        with st.spinner("Training logistic regression and random forest for each strategy…"):
            wr = audit_guided_weights(data, feats, res) if res.flagged else None
            ev = evaluate_strategies(data, test, feats, res, protected, weights=wr)
        st.session_state[f"mit_{key}_out"] = (wr, ev)
    if f"mit_{key}_out" in st.session_state:
        wr, ev = st.session_state[f"mit_{key}_out"]
        if wr is not None:
            c = st.columns(3)
            c[0].metric("Weighted feature", wr.feature)
            c[1].metric("Effective sample size", f"{wr.ess_ratio:.0%}")
            c[2].metric("Weights accepted", "yes" if wr.accepted else "no", help=wr.reason or None)
            st.dataframe(wr.table.round(3), hide_index=True, width="stretch")
        show = ev[["strategy", "model", "auc", "accuracy", "eo_gap", "dp_gap", "group_pred_bias",
                   "auc_change", "eo_gap_change"]].copy()
        show.columns = ["Strategy", "Model", "AUC", "Accuracy", "EO gap", "DP gap",
                        "Over/under-prediction for group", "ΔAUC vs baseline", "ΔEO gap vs baseline"]
        st.dataframe(show.round(3), hide_index=True, width="stretch")
        long = ev.melt(id_vars=["strategy", "model"], value_vars=["eo_gap", "group_pred_bias"],
                       var_name="metric", value_name="value")
        long["metric"] = long["metric"].map({"eo_gap": "Equalized-odds gap", "group_pred_bias": "Prediction bias for group"})
        chart = alt.Chart(long).mark_bar().encode(
            x=alt.X("value:Q", title=None), y=alt.Y("strategy:N", title=None, sort=None),
            color=alt.Color("model:N", legend=alt.Legend(orient="bottom", title=None)),
            yOffset="model:N", tooltip=["strategy", "model", alt.Tooltip("value:Q", format=".3f")],
        ).properties(height=260).facet(column=alt.Column("metric:N", title=None))
        st.altair_chart(chart, use_container_width=True)
        st.caption("Lower is better for both metrics (prediction bias closer to 0). "
                   "Oracle uses the clean audit labels, which a real auditor never has.")


# =====================================================================  pages
def page_overview():
    st.title("🔎 Candidate Feature Auditor")
    st.markdown("#### Bias-Aware AI Pipelines with Fairness Detection: a working model of the V6 auditor")
    st.markdown(
        "Machine-learning datasets can carry **group-dependent label errors** even when nobody has said "
        "which columns are sensitive. This tool audits a dataset **before any model is trained**: it flags "
        "a feature block only when the evidence for unusual label behaviour is stronger than noise, and "
        "otherwise **abstains**.")
    c = st.columns(3)
    c[0].markdown("**1 · Ranking**  \nWhich feature looks most unusual?  \n_Every dataset has a top feature, even pure noise._")
    c[1].markdown("**2 · Detection**  \nIs that evidence stronger than chance?  \n_Cross-fitted residual test + Bonferroni + BY._")
    c[2].markdown("**3 · Mitigation**  \nDoes acting on the flag improve fairness?  \n_Evaluated separately, on clean test labels._")

    st.subheader("How V6 works")
    st.graphviz_chart("""
    digraph {
      rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2f7", fontname="Helvetica", fontsize=11];
      T [label="Trusted reference\\n(clean labels, S=0)"]; A [label="Audit sample\\n(labels under review, S=1)"];
      D [label="Design subset (25%)\\nblocks @ assoc ≥ 0.60\\ngroup boundaries", fillcolor="#fff4e0"];
      X [label="Analysis set\\n2 folds per source"];
      M [label="Cross-fitted logistic models\\ne(X)=P(S=1|X)\\nm(X)=P(Y=1|X)"];
      R [label="Residual product\\nr=(S−e)(Y−m)"];
      C [label="Group contrasts of mean r\\nBonferroni within block\\nBY across blocks"];
      F [label="FLAG block\\nfor human review", fillcolor="#fde2dc"]; N [label="ABSTAIN\\nno supported candidate", fillcolor="#e3f2e1"];
      T -> D; T -> X; A -> X; D -> X [style=dashed, label="geometry"]; X -> M -> R -> C; C -> F [label="q < 0.05"]; C -> N [label="otherwise"];
    }""")
    st.markdown(
        "- **Source model** `e(X)` explains away clean population differences between trusted and audit data.\n"
        "- **Label model** `m(X)` explains legitimate feature–label relationships.\n"
        "- Whatever group-linked source–label association **remains** is a signal worth reviewing.\n"
        "- **Cross-fitting**: no record is predicted by a model that trained on it.")
    st.subheader("Try it")
    st.markdown(
        "- **🧪 Synthetic Lab**: the paper's generator. Run a clean control, a corruption case and a clean "
        "population shift, and compare V6 with the V5 permutation auditor it replaced.\n"
        "- **🌍 Real-Data Audit**: COMPAS, German Credit, Heart Disease or your own CSV, with injected "
        "corruption or shift (semi-synthetic transfer), followed by audit-guided mitigation.\n"
        "- **📊 Calibration**: Monte-Carlo alarm rates with exact confidence bounds, live and from saved studies.")


def page_synthetic():
    st.header("🧪 Synthetic Lab")
    st.caption("Generator: legitimate predictors x1–x5, binary group `a` (the corruption target, never labelled "
               "as sensitive), noisy proxy `p`, irrelevant columns z1…zk. Corruption flips positive audit labels "
               "in group a=1 to 0; shift moves the audit mean of x1 without changing the label rule.")
    preset = st.selectbox("Scenario preset", list(PRESETS), index=0)
    b = PRESETS[preset]
    k = preset
    with st.expander("Customise scenario", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        n_audit = c1.number_input("Audit records", 500, 30000, b.n_audit, 500, key=f"na{k}")
        n_trusted = c1.number_input("Trusted records", 200, 10000, b.n_trusted, 100, key=f"nt{k}")
        corruption = c2.slider("Corruption rate", 0.0, 0.6, b.corruption, 0.05, key=f"c{k}")
        shift = c2.slider("Population shift on x1", 0.0, 2.5, b.shift, 0.25, key=f"s{k}")
        prevalence = c3.slider("Group prevalence", 0.05, 0.6, b.prevalence, 0.05, key=f"pv{k}")
        proxy_noise = c3.slider("Proxy noise", 0.0, 0.5, b.proxy_noise, 0.05, key=f"pn{k}")
        n_noise = c4.slider("Irrelevant columns", 0, 30, b.n_noise, 1, key=f"nn{k}")
        missing = c4.slider("Missingness (numeric)", 0.0, 0.4, b.missing, 0.05, key=f"m{k}")
        o1, o2, o3 = st.columns(3)
        nonlinear = o1.checkbox("Nonlinear label rule", b.nonlinear, key=f"nl{k}")
        corr_nuis = o2.checkbox("Correlated nuisance columns", b.correlated_nuisance, key=f"cn{k}")
        random_labels = o3.checkbox("Random labels", b.random_labels, key=f"rl{k}")
    sc = Scenario(n_audit=int(n_audit), n_trusted=int(n_trusted), prevalence=prevalence, proxy_noise=proxy_noise,
                  n_noise=int(n_noise), corruption=corruption, shift=shift, nonlinear=nonlinear, missing=missing,
                  correlated_nuisance=corr_nuis, random_labels=random_labels)
    c1, c2, c3 = st.columns([1, 1, 2])
    seed = c1.number_input("Seed", 0, 10**6, 11)
    run_v5 = c2.checkbox("Compare with V5", True)
    B = c3.select_slider("V5 permutations B", [199, 499, 999, 1999], 1999)
    if st.button("▶ Run audit", type="primary"):
        with st.spinner("Generating data and auditing…"):
            df = generate(sc, seed=int(seed))
            feats = feature_names(sc)
            cfg = AuditConfig(seed=int(seed))
            t = time.time()
            r6 = audit_v6(df, feats, config=cfg)
            t6 = time.time() - t
            r5 = audit_v5(df, feats, config=cfg, B=B) if run_v5 else None
        st.session_state["syn"] = dict(sc=sc, seed=int(seed), df=df, feats=feats, r6=r6, r5=r5, t6=t6)

    if "syn" not in st.session_state:
        st.stop()
    S = st.session_state["syn"]
    df, feats, r6, r5, sc = S["df"], S["feats"], S["r6"], S["r5"], S["sc"]
    with st.expander("Data preview"):
        st.dataframe(df.drop(columns=["y_clean"]).head(12), width="stretch")
        rates = df.groupby("source")["y"].mean().rename({0: "trusted", 1: "audit"})
        st.write("Observed positive rate by source:", rates.round(3).to_dict())
    cols = st.columns(2 if r5 is not None else 1)
    with cols[0]:
        render_result(r6, feats, "syn6")
    if r5 is not None:
        with cols[1]:
            render_result(r5, feats, "syn5")
    with st.expander("🔍 Reveal ground truth"):
        truth = []
        if sc.corruption > 0:
            truth.append(f"Corruption: {sc.corruption:.0%} of positive audit labels in group **a = 1** flipped to 0 "
                         f"(proxy `p` noise {sc.proxy_noise:.0%}).")
        if sc.shift:
            truth.append(f"Clean population shift: audit mean of **x1** moved by {sc.shift} (label rule unchanged).")
        if sc.random_labels:
            truth.append("Labels are pure noise; any flag is a false alarm.")
        if not truth:
            truth.append("Matched clean control: no corruption and no shift; any flag is a false alarm.")
        for t in truth:
            st.markdown("- " + t)
        for r in [r for r in (r6, r5) if r is not None]:
            if sc.corruption > 0:
                st.markdown(f"**{r.method}**: target block {'✅ recovered' if r.selects('a') else '❌ missed'}"
                            + (" · ⚠ extra blocks flagged" if any('a' not in b for b in r.selected_blocks) else ""))
            else:
                st.markdown(f"**{r.method}**: {'⚠ false alarm' if r.flagged else '✅ correctly abstained'}")
    test = generate_test(sc, n=5000, seed=S["seed"])
    render_mitigation(df, test, feats, r6, (test["a"] == 1).to_numpy(), "a = 1", "syn")


def page_real():
    st.header("🌍 Real-Data Audit")
    st.caption("Semi-synthetic transfer: real features and correlations, record-disjoint trusted / audit / test "
               "splits, and a controlled corruption mechanism so recovery can be scored. You can also upload a "
               "CSV that already has a trusted-vs-audit `source` column.")
    choice = st.selectbox("Dataset", list(DATASETS) + ["Upload CSV"])
    existing_source = False
    if choice == "Upload CSV":
        up = st.file_uploader("CSV file", type="csv")
        if up is None:
            st.stop()
        raw = pd.read_csv(up)
        label = st.selectbox("Binary label column", raw.columns)
        if raw[label].nunique() != 2:
            st.error("Label column must be binary.")
            st.stop()
        vals = sorted(raw[label].unique())
        raw[label] = (raw[label] == vals[1]).astype(int)
        existing_source = "source" in raw.columns and st.checkbox(
            "Use existing `source` column (0 = trusted, 1 = audit) instead of a semi-synthetic split", True)
        info = None
    else:
        info = DATASETS[choice]
        raw = load_dataset(choice)
        label = info.label
        st.markdown(f"**{info.description}**  \nLabel `{label}`: {info.label_meaning}")
    with st.expander("Data preview"):
        st.dataframe(raw.head(15), width="stretch")

    feats_all = [c for c in raw.columns if c not in (label, "source", "y", "y_clean")]
    seed = st.number_input("Seed", 0, 10**6, 3)
    corr = shift = None
    if not existing_source:
        c1, c2 = st.columns(2)
        test_frac = c1.slider("Held-out clean test fraction", 0.1, 0.4, 0.25, 0.05)
        trusted_frac = c2.slider("Trusted fraction of remaining rows", 0.1, 0.6, 0.30, 0.05)
        st.markdown("**Inject label corruption into the audit source**")
        c1, c2, c3, c4 = st.columns(4)
        tf = info.target_feature if info else feats_all[0]
        cfeat = c1.selectbox("Corrupted attribute", feats_all, index=feats_all.index(tf) if tf in feats_all else 0)
        levels = sorted(raw[cfeat].dropna().unique().tolist(), key=str)
        dv = [v for v in (info.target_values if info and cfeat == info.target_feature else levels[:1]) if v in levels]
        cvals = c2.multiselect("Group value(s)", levels, default=dv)
        direction = c3.selectbox("Flip", ["0 → 1", "1 → 0"],
                                 index=0 if not info or info.flip_from == 0 else 1)
        rate = c4.slider("Corruption rate", 0.0, 0.6, 0.30, 0.05)
        if rate > 0 and cvals:
            ff = 0 if direction.startswith("0") else 1
            corr = CorruptionSpec(cfeat, tuple(cvals), ff, 1 - ff, rate)
        st.markdown("**Clean population shift** (biased audit sampling; P(Y | X) unchanged)")
        c1, c2 = st.columns(2)
        numeric = [f for f in feats_all if pd.api.types.is_numeric_dtype(raw[f])]
        sf_default = info.shift_feature if info and info.shift_feature in numeric else (numeric[0] if numeric else None)
        sfeat = c1.selectbox("Shift feature", numeric, index=numeric.index(sf_default) if sf_default else 0) if numeric else None
        strength = c2.slider("Shift strength (log-odds per SD)", 0.0, 2.0, 0.0, 0.25)
        if sfeat and strength > 0:
            shift = ShiftSpec(sfeat, strength)
    run_v5 = st.checkbox("Compare with V5", True)

    if st.button("▶ Run audit", type="primary"):
        with st.spinner("Splitting and auditing…"):
            if existing_source:
                data = raw.rename(columns={label: "y"}).copy()
                data["source"] = data["source"].astype(int)
                test = None
            else:
                data, test = make_semisynthetic(raw, label, test_frac=test_frac, trusted_frac=trusted_frac,
                                                corruption=corr, shift=shift, seed=int(seed))
            feats = feature_columns(data)
            cfg = AuditConfig(seed=int(seed))
            r6 = audit_v6(data, feats, config=cfg)
            r5 = audit_v5(data, feats, config=cfg) if run_v5 else None
        st.session_state["real"] = dict(data=data, test=test, feats=feats, r6=r6, r5=r5, corr=corr, shift=shift,
                                        info=info)
    if "real" not in st.session_state:
        st.stop()
    R = st.session_state["real"]
    data, test, feats, r6, r5 = R["data"], R["test"], R["feats"], R["r6"], R["r5"]
    st.write(f"Trusted rows: **{int((data.source == 0).sum())}** · audit rows: **{int((data.source == 1).sum())}**"
             + (f" · clean test rows: **{len(test)}**" if test is not None else ""))
    if "y_clean" in data:
        flipped = int((data["y"] != data["y_clean"]).sum())
        st.write(f"Injected label flips in audit: **{flipped}**")
    cols = st.columns(2 if r5 is not None else 1)
    with cols[0]:
        render_result(r6, feats, "real6")
    if r5 is not None:
        with cols[1]:
            render_result(r5, feats, "real5")
    if R["corr"] is not None:
        hit = r6.selects(R["corr"].feature)
        st.markdown(f"**Ground truth:** corruption injected on `{R['corr'].feature}` ∈ {list(R['corr'].values)} → "
                    f"V6 {'✅ recovered it' if hit else '❌ did not recover it'}"
                    + (f"; V5 {'✅ recovered it' if r5.selects(R['corr'].feature) else '❌ did not'}" if r5 else ""))
    if test is not None:
        spec = R["corr"] or (CorruptionSpec(R["info"].target_feature, R["info"].target_values, 0, 1, 0)
                             if R["info"] else None)
        if spec is not None:
            render_mitigation(data, test, feats, r6, spec.mask(test),
                              f"{spec.feature} ∈ {list(spec.values)}", "real")
    elif r6.flagged:
        wr = audit_guided_weights(data, feats, r6)
        st.subheader("Proposed audit-guided weights")
        st.dataframe(wr.table.round(3), hide_index=True)
        st.caption(f"ESS {wr.ess_ratio:.0%} · {'accepted' if wr.accepted else 'rejected: ' + wr.reason}. "
                   "No clean test labels are available in this mode, so the effect on fairness cannot be scored here.")


def page_calibration():
    st.header("📊 Calibration: does the auditor stay quiet on clean data?")
    st.caption("Gate from the project protocol: in EVERY clean condition, the exact two-sided 95% upper bound of "
               "the alarm rate must be below 5%. Random-label abstention needs a lower bound ≥ 90%. "
               "Power (recall) is only meaningful after calibration passes.")
    tab_live, tab_saved = st.tabs(["Run live", "Saved studies"])
    with tab_live:
        c1, c2, c3 = st.columns(3)
        cond = c1.selectbox("Condition", list(CALIBRATION_CONDITIONS), index=list(CALIBRATION_CONDITIONS).index("shift_1.50_nonlinear"))
        reps = c2.slider("Repetitions", 20, 300, 60, 20)
        methods = c3.multiselect("Methods", ["V6", "V5"], default=["V6", "V5"])
        if st.button("▶ Run Monte Carlo", type="primary"):
            sc = CALIBRATION_CONDITIONS[cond]
            feats = feature_names(sc)
            alarms = {m: 0 for m in methods}
            bar = st.progress(0.0)
            for i in range(reps):
                seed = 50_000_000 + i
                df = generate(sc, seed=seed)
                cfg = AuditConfig(seed=seed)
                for m in methods:
                    res = audit_v6(df, feats, config=cfg) if m == "V6" else audit_v5(df, feats, config=cfg, B=999)
                    alarms[m] += int(res.flagged)
                bar.progress((i + 1) / reps, text=f"{i + 1}/{reps} datasets")
            rows = []
            for m in methods:
                if cond == "random_labels":
                    g = abstention_gate(reps - alarms[m], reps)
                    rows.append({"Method": m, "Abstentions": f"{g['abstentions']}/{reps}", "Rate": f"{g['rate']:.1%}",
                                 "95% CI": f"[{g['ci_low']:.1%}, {g['ci_high']:.1%}]",
                                 "Gate (lower ≥ 90%)": "PASS" if g["passed"] else "FAIL"})
                else:
                    g = calibration_gate(alarms[m], reps)
                    rows.append({"Method": m, "Alarms": f"{g['alarms']}/{reps}", "Rate": f"{g['rate']:.1%}",
                                 "95% CI": f"[{g['ci_low']:.1%}, {g['ci_high']:.1%}]",
                                 "Gate (upper < 5%)": "PASS" if g["passed"] else "FAIL"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("With few repetitions even zero alarms cannot pass: 0/60 has an upper bound of 6.0%. "
                       "This is why the saved studies use 500 repetitions per condition.")
    with tab_saved:
        path = os.path.join(RESULTS, "calibration_summary.csv")
        if not os.path.exists(path):
            st.info("No saved study yet. Run `python experiments/run_study.py --study calibration`.")
        else:
            cal = pd.read_csv(path)
            meta = json.load(open(os.path.join(RESULTS, "calibration_meta.json")))
            st.markdown(f"**Calibration study**: {meta['datasets']:,} datasets, {meta['reps']} repetitions per "
                        f"condition, methods {', '.join(m.upper() for m in meta['methods'])} on the same data.")
            clean = cal[cal["endpoint"] == "clean_alarm"]
            for m, g in clean.groupby("method"):
                st.write(f"**{m}**: {int(g['passed'].sum())}/{len(g)} clean conditions pass the gate")
            base = alt.Chart(clean).encode(y=alt.Y("condition:N", sort=None, title=None),
                                           color=alt.Color("method:N", legend=alt.Legend(orient="bottom")),
                                           yOffset="method:N")
            pts = base.mark_point(filled=True).encode(x=alt.X("rate:Q", title="Clean alarm rate (exact 95% CI)",
                                                              axis=alt.Axis(format="%")),
                                                      tooltip=["condition", "method", "count", "runs",
                                                               alt.Tooltip("ci_high:Q", format=".2%")])
            err = base.mark_errorbar().encode(x=alt.X("ci_low:Q", title="Clean alarm rate (exact 95% CI)", axis=alt.Axis(format="%")), x2="ci_high:Q")
            rule = alt.Chart(pd.DataFrame({"x": [0.05]})).mark_rule(color="#e4572e", strokeDash=[4, 4]).encode(x="x:Q")
            st.altair_chart((err + pts + rule).properties(height=34 * len(clean["condition"].unique()) + 100),
                            use_container_width=True)
            st.dataframe(cal, hide_index=True, width="stretch")
        ppath = os.path.join(RESULTS, "power_summary.csv")
        if os.path.exists(ppath):
            pw = pd.read_csv(ppath)
            st.markdown("**Power study** (exploratory: recall of the corrupted block)")
            ch = alt.Chart(pw).mark_bar().encode(
                x=alt.X("rate:Q", title="Target recall", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("condition:N", sort=None, title=None), color="method:N", yOffset="method:N",
                tooltip=["condition", "method", "count", "runs", alt.Tooltip("off_target_rate:Q", format=".1%")])
            rule = alt.Chart(pd.DataFrame({"x": [0.8]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
            st.altair_chart((ch + rule).properties(height=40 * len(pw["condition"].unique()) + 100),
                            use_container_width=True)
            st.dataframe(pw, hide_index=True, width="stretch")


def page_method():
    st.header("📖 Method, evolution and limits")
    st.subheader("From V1 to V6")
    st.dataframe(pd.DataFrame([
        ["V1", "Local neighbour comparison + bounded reweighting", "Top-3 always selected, even on pure noise"],
        ["V2", "Empirical null, adjusted p-values, gated mitigation", "Leakage and weak precision; success claims withdrawn"],
        ["V3", "Independent trusted sources, feature blocks, coherent score", "9/9 clean gates passed; sensitivity target missed"],
        ["V4", "Trusted-sample allocation and correction (BY / Holm)", "Recall up; primary failed calibration"],
        ["V5", "Frozen validation with broader controls", "18 matched passed; 2 shift controls failed"],
        ["V6", "Cross-fitted source adjustment", "Shift alarms cut sharply; 24/25 conditions passed"],
    ], columns=["Version", "What changed", "What the archived study found"]), hide_index=True, width="stretch")
    st.subheader("V6 statistic")
    st.latex(r"r_i = \big(S_i - \hat e^{-k(i)}(X_i)\big)\big(Y_i - \hat m^{-k(i)}(X_i)\big)")
    st.latex(r"\Delta_{gh} = \bar r_g - \bar r_h,\qquad z = \Delta_{gh} \big/ \sqrt{s_g^2/n_g + s_h^2/n_h}")
    st.markdown(
        "Per-group effect size shown in the charts is the partialling-out estimate "
        r"$\hat\theta_g = \sum_{i\in g} r_i / \sum_{i\in g}(S_i-\hat e_i)^2$: the audit-minus-trusted "
        "difference in label probability inside group *g* after both models have used the observed features.")
    st.subheader("Support and selection rules")
    st.markdown(
        "- Candidate group: ≥ 20 records from each source, ≥ 10 per source in each fold; ≥ 100 analysis records per source.\n"
        "- Unsupported contrasts keep p = 1 and stay in the correction family.\n"
        "- Block p = Bonferroni over its contrasts; Benjamini–Yekutieli across blocks; select if q < 0.05.")
    st.subheader("What this tool does NOT establish")
    st.markdown(
        "- A flag is not proof of corruption, discrimination or causation, and not a reason to delete a feature.\n"
        "- Abstention does not certify the data as unbiased.\n"
        "- The auditor needs a trusted reference sample and observed informative features (the group itself or a proxy).\n"
        "- The normal approximation is not a finite-sample guarantee; calibration must be checked by simulation.\n"
        "- This is an independent re-implementation from the project's written method; numbers will not "
        "match the archived V6 study exactly.")


PAGES = {
    "🏠 Overview": page_overview,
    "🧪 Synthetic Lab": page_synthetic,
    "🌍 Real-Data Audit": page_real,
    "📊 Calibration": page_calibration,
    "📖 Method & Limits": page_method,
}

with st.sidebar:
    st.markdown("### Candidate Feature Auditor")
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("DSE3170 PBL-3 · Manipal University Jaipur  \nSoumya Shradha · Tanishk Gangwar · "
               "Yagya Salwan · Vivansh Garg  \nGuide: Dr. Chirag Joshi")

PAGES[page]()
