"""Version 5 comparator: coherent score with conditional source permutations.

This is the V3-milestone-4 auditor frozen in V5 (25% trusted fit / 75%
calibration, BY across blocks). Within cells defined jointly by the block's
group pattern and a risk stratum, audit and trusted-calibration records are
assumed exchangeable. Null audit-positive counts are drawn from the
hypergeometric distribution with cell totals fixed, and every draw recomputes
the complete block maximum.

Coherent score for a member feature:
    T = max_g sum_s w(s) d(s,g) - min_g sum_s w(s) d(s,g)
with d(s,g) = audit minus calibration positive rate in stratum s, group g.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .geometry import DesignGeometry
from .stats import by_adjust
from .v6 import AuditConfig, AuditResult, split_design


def _risk_strata(geom: DesignGeometry, design: pd.DataFrame, analysis: pd.DataFrame,
                 label_col: str, exclude: set[str], n_strata: int, C: float, seed: int) -> np.ndarray:
    y_d = design[label_col].to_numpy().astype(int)
    if len(np.unique(y_d)) < 2 or len(exclude) == len(geom.features):
        return np.zeros(len(analysis), dtype=int)
    model = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    model.fit(geom.encode(design, exclude), y_d)
    risk_d = model.predict_proba(geom.encode(design, exclude))[:, 1]
    edges = np.unique(np.quantile(risk_d, np.linspace(0, 1, n_strata + 1)[1:-1]))
    risk_a = model.predict_proba(geom.encode(analysis, exclude))[:, 1]
    return np.searchsorted(edges, risk_a, side="right")


def audit_v5(df: pd.DataFrame, features: list[str], label_col: str = "y",
             source_col: str = "source", config: AuditConfig | None = None,
             n_strata: int = 4, B: int = 1999, min_cell: int = 10,
             min_eligible: int = 100, min_coverage: float = 0.5) -> AuditResult:
    cfg = config or AuditConfig()
    rng = np.random.default_rng(cfg.seed)

    design_mask, analysis_mask = split_design(df, source_col, cfg.design_frac, rng)
    design = df.loc[design_mask]
    analysis = df.loc[analysis_mask].reset_index(drop=True)
    geom = DesignGeometry(n_bins=cfg.n_bins, assoc_threshold=cfg.assoc_threshold).fit(design, features)

    y = analysis[label_col].to_numpy().astype(int)
    s = analysis[source_col].to_numpy().astype(int)
    n_src = np.array([(s == 0).sum(), (s == 1).sum()])

    block_rows, member_rows, group_rows = [], [], []
    for members in geom.blocks:
        strata = _risk_strata(geom, design, analysis, label_col, set(members),
                              n_strata, cfg.C, cfg.seed)
        member_codes = np.column_stack([geom.group_codes(analysis, f) for f in members])
        # joint cell = (joint block pattern, stratum)
        joint_keys = np.column_stack([member_codes, strata])
        uniq, cell = np.unique(joint_keys, axis=0, return_inverse=True)
        cell = cell.ravel()
        n_cells = len(uniq)
        N_a = np.bincount(cell, weights=(s == 1), minlength=n_cells)
        N_c = np.bincount(cell, weights=(s == 0), minlength=n_cells)
        K = np.bincount(cell, weights=y, minlength=n_cells)
        A_obs = np.bincount(cell, weights=y * (s == 1), minlength=n_cells)

        A_null = rng.hypergeometric(K.astype(np.int64)[None, :].repeat(B, 0),
                                    (N_a + N_c - K).astype(np.int64)[None, :].repeat(B, 0),
                                    N_a.astype(np.int64)[None, :].repeat(B, 0))
        block_obs, block_null, block_supported = -np.inf, np.full(B, -np.inf), False

        for j, f in enumerate(members):
            stat = _member_statistic(uniq[:, j], uniq[:, -1], N_a, N_c, K,
                                     geom.n_groups(f), n_strata, n_src,
                                     min_cell, min_eligible, min_coverage)
            if stat is None:
                member_rows.append({"feature": f, "supported": False, "T_obs": float("nan"), "p_member": 1.0})
                continue
            T_obs, groups_diff = stat["T"](A_obs[None, :])[0], stat["pooled"](A_obs[None, :])[0]
            T_null = stat["T"](A_null)
            block_supported = True
            block_obs = max(block_obs, T_obs)
            block_null = np.maximum(block_null, T_null)
            p_member = (1 + np.sum(T_null >= T_obs - 1e-12)) / (B + 1)
            member_rows.append({"feature": f, "supported": True, "T_obs": float(T_obs),
                                "p_member": float(p_member), "eligible_strata": stat["eligible"]})
            labels = geom.group_labels(f)
            for gi, g in enumerate(stat["groups"]):
                in_g = member_codes[:, j] == g
                group_rows.append({
                    "feature": f, "group": labels[g], "code": int(g),
                    "n_trusted": int(np.sum(in_g & (s == 0))), "n_audit": int(np.sum(in_g & (s == 1))),
                    "coherent_diff": float(groups_diff[gi]),
                    "trusted_rate": float(y[in_g & (s == 0)].mean()) if np.any(in_g & (s == 0)) else float("nan"),
                    "audit_rate": float(y[in_g & (s == 1)].mean()) if np.any(in_g & (s == 1)) else float("nan"),
                    "supported": True,
                })

        if block_supported:
            p_block = (1 + np.sum(block_null >= block_obs - 1e-12)) / (B + 1)
        else:
            p_block, block_obs = 1.0, float("nan")
        block_rows.append({"members": members, "block": "{" + ", ".join(members) + "}",
                           "supported": block_supported, "T_block": float(block_obs),
                           "p_block": float(p_block)})

    blocks = pd.DataFrame(block_rows)
    blocks["q_BY"] = by_adjust(blocks["p_block"].to_numpy())
    blocks["selected"] = blocks["supported"] & (blocks["q_BY"] < cfg.alpha)
    blocks["exploratory_rank"] = blocks["T_block"].fillna(-1).rank(method="first", ascending=False).astype(int)
    blocks = blocks.sort_values("exploratory_rank").reset_index(drop=True)
    selected = [list(b) for b, sel in zip(blocks["members"], blocks["selected"]) if sel]

    diagnostics = {
        "n_design": int(design_mask.sum()),
        "n_analysis_trusted": int(n_src[0]),
        "n_analysis_audit": int(n_src[1]),
        "permutations": B,
        "min_attainable_p": 1 / (B + 1),
        "n_blocks": len(geom.blocks),
    }
    conf = cfg.__dict__.copy()
    conf.update({"B": B, "n_strata": n_strata})
    return AuditResult("V5", selected, blocks, pd.DataFrame(member_rows), pd.DataFrame(group_rows),
                       diagnostics, conf, geom)


def _member_statistic(g_of_cell, s_of_cell, N_a, N_c, K, n_groups, n_strata, n_src,
                      min_cell, min_eligible, min_coverage):
    """Build vectorised score functions for one member, or None if unsupported."""
    present = np.unique(g_of_cell)
    G, S = n_groups, n_strata
    idx = g_of_cell * S + s_of_cell                     # (group, stratum) index per joint cell
    M = np.zeros((len(g_of_cell), G * S))
    M[np.arange(len(g_of_cell)), idx] = 1.0
    na = (N_a @ M).reshape(G, S)
    nc = (N_c @ M).reshape(G, S)
    kk = (K @ M).reshape(G, S)

    eligible = [st for st in range(S)
                if np.all(na[present, st] >= min_cell) and np.all(nc[present, st] >= min_cell)]
    if not eligible:
        return None
    n_elig = na[:, eligible].sum() + nc[:, eligible].sum()
    cov_a = na[:, eligible].sum() / n_src[1]
    cov_c = nc[:, eligible].sum() / n_src[0]
    if n_elig < min_eligible or cov_a < min_coverage or cov_c < min_coverage:
        return None

    na_s = na[:, eligible].sum(0)
    nc_s = nc[:, eligible].sum(0)
    h = na_s * nc_s / (na_s + nc_s)
    w = h / h.sum()
    na_e = na[np.ix_(present, eligible)]
    nc_e = nc[np.ix_(present, eligible)]
    kk_e = kk[np.ix_(present, eligible)]

    def pooled(A):  # A: (draws, cells) audit positives per joint cell
        pa = (A @ M).reshape(len(A), G, S)[:, present][:, :, eligible]
        pc = kk_e[None] - pa
        d = pa / na_e[None] - pc / nc_e[None]
        return (d * w[None, None, :]).sum(-1)       # (draws, groups)

    def T(A):
        sg = pooled(A)
        return sg.max(1) - sg.min(1)

    return {"T": T, "pooled": pooled, "groups": present, "eligible": [int(e) for e in eligible]}
