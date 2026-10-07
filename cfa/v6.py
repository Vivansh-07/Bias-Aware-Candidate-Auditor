"""Version 6: cross-fitted source-adjusted candidate feature auditor.

Procedure (paper Sec. V, report App. A):
1. Reserve 25% of trusted records as the design subset -> blocks (assoc >= 0.60)
   and candidate group boundaries.
2. Remaining trusted + audit records form the analysis set; split each source
   into two folds.
3. For each held-out fold, standardised logistic models trained on the opposite
   fold predict e(X) = P(S=1|X) and m(X) = P(Y=1|X) from all observed features.
4. Residual product r = (S - e(X)) (Y - m(X)).
5. Pairwise contrasts of mean r between candidate groups (normal approximation),
   Bonferroni within block, Benjamini-Yekutieli across blocks.
6. Select supported blocks with adjusted p < 0.05, otherwise abstain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .geometry import DesignGeometry
from .stats import by_adjust

ABSTAIN_MESSAGE = "No statistically supported candidate detected."


@dataclass
class AuditConfig:
    design_frac: float = 0.25
    assoc_threshold: float = 0.60
    alpha: float = 0.05
    n_folds: int = 2
    min_group_per_source: int = 20
    min_group_per_source_fold: int = 10
    min_source: int = 100
    n_bins: int = 4
    C: float = 1.0
    seed: int = 0


@dataclass
class AuditResult:
    method: str
    selected_blocks: list[list[str]]
    blocks: pd.DataFrame          # one row per block
    contrasts: pd.DataFrame       # one row per pairwise contrast
    groups: pd.DataFrame          # one row per (feature, group)
    diagnostics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    geometry: DesignGeometry | None = field(default=None, repr=False)

    @property
    def flagged(self) -> bool:
        return len(self.selected_blocks) > 0

    @property
    def decision(self) -> str:
        if not self.flagged:
            return ABSTAIN_MESSAGE
        names = ["{" + ", ".join(b) + "}" for b in self.selected_blocks]
        return "Flag for human review: " + "; ".join(names)

    def selects(self, feature: str) -> bool:
        return any(feature in b for b in self.selected_blocks)

    def lead_feature(self, block: list[str]) -> str:
        """Member of a block carrying its strongest contrast (or score)."""
        if "p_raw" in self.contrasts.columns:
            sub = self.contrasts[self.contrasts["feature"].isin(block)]
            return str(sub.loc[sub["p_raw"].idxmin(), "feature"]) if len(sub) else block[0]
        sub = self.contrasts[self.contrasts["feature"].isin(block) & self.contrasts["supported"]]
        return str(sub.loc[sub["T_obs"].idxmax(), "feature"]) if len(sub) else block[0]

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "decision": self.decision,
            "selected_blocks": self.selected_blocks,
            "blocks": self.blocks.to_dict(orient="records"),
            "contrasts": self.contrasts.to_dict(orient="records"),
            "groups": self.groups.to_dict(orient="records"),
            "diagnostics": self.diagnostics,
            "config": self.config,
        }


def split_design(df: pd.DataFrame, source_col: str, design_frac: float,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (design, analysis). Design is drawn from trusted rows only."""
    trusted_idx = np.flatnonzero(df[source_col].to_numpy() == 0)
    n_design = int(round(design_frac * len(trusted_idx)))
    design_idx = rng.choice(trusted_idx, size=n_design, replace=False)
    design = np.zeros(len(df), dtype=bool)
    design[design_idx] = True
    return design, ~design


def assign_folds(source: np.ndarray, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """Balanced fold assignment carried out separately inside each source."""
    folds = np.empty(len(source), dtype=int)
    for s in np.unique(source):
        idx = np.flatnonzero(source == s)
        perm = rng.permutation(idx)
        folds[perm] = np.arange(len(perm)) % n_folds
    return folds


def _fit_predict(X_tr, y_tr, X_te, C, seed) -> np.ndarray:
    if len(np.unique(y_tr)) < 2:
        return np.full(len(X_te), float(np.mean(y_tr)))
    model = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def cross_fit(X: np.ndarray, y: np.ndarray, s: np.ndarray, folds: np.ndarray,
              C: float = 1.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predictions m(X)=P(Y=1|X) and e(X)=P(S=1|X)."""
    m = np.empty(len(y))
    e = np.empty(len(y))
    for k in np.unique(folds):
        te = folds == k
        tr = ~te
        m[te] = _fit_predict(X[tr], y[tr], X[te], C, seed)
        e[te] = _fit_predict(X[tr], s[tr], X[te], C, seed)
    return m, e


def audit_v6(df: pd.DataFrame, features: list[str], label_col: str = "y",
             source_col: str = "source", config: AuditConfig | None = None) -> AuditResult:
    cfg = config or AuditConfig()
    rng = np.random.default_rng(cfg.seed)

    design_mask, analysis_mask = split_design(df, source_col, cfg.design_frac, rng)
    design = df.loc[design_mask]
    analysis = df.loc[analysis_mask].reset_index(drop=True)

    geom = DesignGeometry(n_bins=cfg.n_bins, assoc_threshold=cfg.assoc_threshold).fit(design, features)

    y = analysis[label_col].to_numpy().astype(int)
    s = analysis[source_col].to_numpy().astype(int)
    folds = assign_folds(s, cfg.n_folds, rng)
    X = geom.encode(analysis)
    m, e = cross_fit(X, y, s, folds, C=cfg.C, seed=cfg.seed)
    r = (s - e) * (y - m)

    n_source = {int(v): int((s == v).sum()) for v in (0, 1)}
    sources_ok = min(n_source.values()) >= cfg.min_source

    group_rows, contrast_rows = [], []
    feature_contrasts: dict[str, list[float]] = {}
    for f in features:
        codes = geom.group_codes(analysis, f)
        labels = geom.group_labels(f)
        present = [g for g in range(len(labels)) if np.any(codes == g)]
        supported = {}
        for g in present:
            in_g = codes == g
            per_source = [int(np.sum(in_g & (s == v))) for v in (0, 1)]
            per_fold = [int(np.sum(in_g & (s == v) & (folds == k)))
                        for v in (0, 1) for k in range(cfg.n_folds)]
            ok = (sources_ok and min(per_source) >= cfg.min_group_per_source
                  and min(per_fold) >= cfg.min_group_per_source_fold)
            supported[g] = ok
            rg = r[in_g]
            res_s = (s - e)[in_g]
            # partialling-out estimate of the audit-minus-trusted label gap in group g
            denom = float(np.sum(res_s ** 2))
            theta = float(np.sum(rg) / denom) if denom > 0 else float("nan")
            se_r = float(rg.std(ddof=1) / np.sqrt(len(rg))) if len(rg) > 1 else float("nan")
            theta_se = se_r * len(rg) / denom if denom > 0 else float("nan")
            group_rows.append({
                "feature": f, "group": labels[g], "code": g,
                "n_trusted": per_source[0], "n_audit": per_source[1],
                "mean_r": float(rg.mean()), "se_r": se_r,
                "label_gap": theta,
                "label_gap_lo": theta - 1.96 * theta_se,
                "label_gap_hi": theta + 1.96 * theta_se,
                "trusted_rate": float(y[in_g & (s == 0)].mean()) if per_source[0] else float("nan"),
                "audit_rate": float(y[in_g & (s == 1)].mean()) if per_source[1] else float("nan"),
                "supported": ok,
            })
        pvals = []
        for g, h in combinations(present, 2):
            if supported[g] and supported[h]:
                rg, rh = r[codes == g], r[codes == h]
                diff = rg.mean() - rh.mean()
                se = np.sqrt(rg.var(ddof=1) / len(rg) + rh.var(ddof=1) / len(rh))
                z = diff / se if se > 0 else 0.0
                p = float(2 * stats.norm.sf(abs(z)))
                sup = True
            else:
                diff, z, p, sup = float("nan"), float("nan"), 1.0, False
            pvals.append(p)
            contrast_rows.append({"feature": f, "group_a": labels[g], "group_b": labels[h],
                                  "diff_mean_r": float(diff), "z": float(z), "p_raw": p,
                                  "supported": sup})
        feature_contrasts[f] = pvals

    contrasts = pd.DataFrame(contrast_rows, columns=["feature", "group_a", "group_b",
                                                     "diff_mean_r", "z", "p_raw", "supported"])
    groups = pd.DataFrame(group_rows)
    blocks = _block_table(geom.blocks, feature_contrasts, contrasts, cfg.alpha)
    selected = [list(b) for b, sel in zip(blocks["members"], blocks["selected"]) if sel]

    diagnostics = {
        "n_design": int(design_mask.sum()),
        "n_analysis_trusted": n_source[0],
        "n_analysis_audit": n_source[1],
        "source_support_ok": sources_ok,
        "source_model_auc": _safe_auc(s, e),
        "label_model_auc": _safe_auc(y, m),
        "n_blocks": len(geom.blocks),
        "association": geom.association.round(3).to_dict(),
    }
    return AuditResult("V6", selected, blocks, contrasts, groups, diagnostics, cfg.__dict__.copy(), geom)


def _block_table(blocks: list[list[str]], feature_contrasts: dict[str, list[float]],
                 contrasts: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    for members in blocks:
        ps = [p for f in members for p in feature_contrasts.get(f, [])]
        n_tests = len(ps)
        supported = bool(contrasts.loc[contrasts["feature"].isin(members), "supported"].any()) if n_tests else False
        p_min = min(ps) if ps else 1.0
        p_block = min(1.0, p_min * n_tests) if supported else 1.0
        best = ""
        if supported:
            sub = contrasts[contrasts["feature"].isin(members) & contrasts["supported"]]
            top = sub.loc[sub["p_raw"].idxmin()]
            best = f"{top['group_a']} vs {top['group_b']}"
        rows.append({"members": members, "block": "{" + ", ".join(members) + "}",
                     "n_contrasts": n_tests, "supported": supported, "p_min_raw": p_min,
                     "p_block": p_block, "best_contrast": best})
    table = pd.DataFrame(rows)
    table["q_BY"] = by_adjust(table["p_block"].to_numpy()) if len(table) else []
    table["selected"] = table["supported"] & (table["q_BY"] < alpha)
    table["exploratory_rank"] = table["p_block"].rank(method="first").astype(int)
    return table.sort_values("exploratory_rank").reset_index(drop=True)


def _safe_auc(y, p) -> float:
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")
