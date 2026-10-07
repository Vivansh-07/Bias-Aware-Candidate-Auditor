"""Audit-guided mitigation, evaluated SEPARATELY from detection.

Only runs after the auditor has flagged a block. For the block's lead feature,
the trusted reference supplies a clean label model m_T(X). Within each candidate
group g of the audit data:
    E_g = mean of m_T(X) over audit records in g   (expected clean positive rate)
    O_g = observed audit positive rate in g
    w(g, 1) = E_g / O_g,   w(g, 0) = (1 - E_g) / (1 - O_g)
Weights are optionally shrunk toward 1, clipped to [0.5, 2.0], normalised to
mean 1 and rejected if the effective sample size falls below 70%.

Downstream comparison (clean held-out test labels):
  baseline        train on audit labels as observed
  reweighted      audit-guided weights above
  drop block      remove the flagged block's features
  trusted only    train on the small trusted reference alone
  oracle          train on the clean audit labels (semi-synthetic only)

Proposed acceptance criteria from the project report: EO gap reduced,
AUC loss <= 0.02, ESS >= 70%, consistent across LR and RF.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from .geometry import DesignGeometry
from .v6 import AuditResult


@dataclass
class WeightResult:
    weights: np.ndarray | None
    ess_ratio: float
    accepted: bool
    feature: str
    table: pd.DataFrame
    reason: str = ""


def audit_guided_weights(data: pd.DataFrame, features: list[str], result: AuditResult,
                         label_col: str = "y", source_col: str = "source",
                         shrink: float = 0.0, clip: tuple[float, float] = (0.5, 2.0),
                         min_ess: float = 0.70, seed: int = 0) -> WeightResult:
    if not result.flagged:
        return WeightResult(None, 1.0, False, "", pd.DataFrame(), "auditor abstained: no intervention")
    geom = result.geometry
    feature = result.lead_feature(result.selected_blocks[0])
    trusted = data[data[source_col] == 0]
    audit = data[data[source_col] == 1]

    ref = LogisticRegression(max_iter=2000, random_state=seed)
    ref.fit(geom.encode(trusted), trusted[label_col].astype(int))
    expected = ref.predict_proba(geom.encode(audit))[:, 1]
    y = audit[label_col].to_numpy().astype(int)
    codes = geom.group_codes(audit, feature)
    labels = geom.group_labels(feature)

    w = np.ones(len(audit))
    rows = []
    for g in np.unique(codes):
        in_g = codes == g
        E, O = float(expected[in_g].mean()), float(y[in_g].mean())
        O = min(max(O, 1e-3), 1 - 1e-3)
        w1, w0 = E / O, (1 - E) / (1 - O)
        w1, w0 = 1 + (1 - shrink) * (w1 - 1), 1 + (1 - shrink) * (w0 - 1)
        w1, w0 = float(np.clip(w1, *clip)), float(np.clip(w0, *clip))
        w[in_g & (y == 1)] = w1
        w[in_g & (y == 0)] = w0
        rows.append({"group": labels[g], "n_audit": int(in_g.sum()), "observed_rate": O,
                     "expected_clean_rate": E, "weight_pos": w1, "weight_neg": w0})
    w = w / w.mean()
    ess = float(w.sum() ** 2 / (len(w) * np.sum(w ** 2)))
    accepted = ess >= min_ess
    reason = "" if accepted else f"effective sample size {ess:.0%} below {min_ess:.0%}"
    return WeightResult(w if accepted else None, ess, accepted, feature, pd.DataFrame(rows), reason)


def _fairness(y_true, y_prob, group, threshold=0.5) -> dict:
    y_hat = (y_prob >= threshold).astype(int)
    g1, g0 = group == 1, group == 0

    def rate(mask):
        return float(y_hat[mask].mean()) if mask.any() else float("nan")

    tpr = [rate(gm & (y_true == 1)) for gm in (g1, g0)]
    fpr = [rate(gm & (y_true == 0)) for gm in (g1, g0)]
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    return {
        "auc": auc,
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "dp_gap": abs(rate(g1) - rate(g0)),
        "eo_gap": max(abs(tpr[0] - tpr[1]), abs(fpr[0] - fpr[1])),
        "group_pred_bias": float(y_prob[g1].mean() - y_true[g1].mean()) if g1.any() else float("nan"),
    }


def _models(seed):
    return {
        "Logistic regression": lambda: LogisticRegression(max_iter=3000, random_state=seed),
        "Random forest": lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                                         random_state=seed, n_jobs=1),
    }


def evaluate_strategies(data: pd.DataFrame, test: pd.DataFrame, features: list[str],
                        result: AuditResult, protected: np.ndarray,
                        label_col: str = "y", source_col: str = "source",
                        weights: WeightResult | None = None, seed: int = 0) -> pd.DataFrame:
    """Train each strategy x model and score on the clean test set.

    `protected` is a boolean array over `test` rows marking the group whose
    fairness is assessed (the corrupted group in semi-synthetic studies).
    """
    audit = data[data[source_col] == 1]
    trusted = data[data[source_col] == 0]
    enc = DesignGeometry().fit(pd.concat([audit, trusted])[features], features)
    y_test = test["y_clean"].to_numpy().astype(int) if "y_clean" in test else test[label_col].to_numpy()
    group = np.asarray(protected).astype(int)

    dropped = set(result.selected_blocks[0]) if result.flagged else set()
    strategies = [("Baseline (audit labels)", audit, label_col, None, set())]
    if weights is not None and weights.accepted:
        strategies.append(("Audit-guided reweighting", audit, label_col, weights.weights, set()))
    if dropped:
        strategies.append(("Drop flagged block", audit, label_col, None, dropped))
    strategies.append(("Trusted reference only", trusted, label_col, None, set()))
    if "y_clean" in audit:
        strategies.append(("Oracle (clean audit labels)", audit, "y_clean", None, set()))

    rows = []
    for name, train, ycol, w, excl in strategies:
        X_tr, X_te = enc.encode(train, excl), enc.encode(test, excl)
        y_tr = train[ycol].to_numpy().astype(int)
        for mname, make in _models(seed).items():
            model = make()
            model.fit(X_tr, y_tr, sample_weight=w)
            prob = model.predict_proba(X_te)[:, 1]
            rows.append({"strategy": name, "model": mname, "n_train": len(train),
                         **_fairness(y_test, prob, group)})
    out = pd.DataFrame(rows)
    base = out[out["strategy"] == "Baseline (audit labels)"].set_index("model")
    out["auc_change"] = out.apply(lambda r: r["auc"] - base.loc[r["model"], "auc"], axis=1)
    out["eo_gap_change"] = out.apply(lambda r: r["eo_gap"] - base.loc[r["model"], "eo_gap"], axis=1)
    return out
