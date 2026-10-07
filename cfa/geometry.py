"""Design geometry learned ONLY from the trusted design subset.

The design subset fixes three things before any analysis outcome is seen:
1. candidate groups for every feature (quantile bins or category levels),
2. association blocks (features linked when association >= threshold),
3. the encoding used by nuisance / risk models (imputation, scaling, one-hot).

Keeping this fit separate from the analysis labels is the sample-role rule
that the project introduced in V3 and retained in V6.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

MISSING = "__missing__"
OTHER = "__other__"


@dataclass
class FeatureSpec:
    name: str
    kind: str  # "numeric" or "categorical"
    median: float | None = None
    mean: float | None = None
    std: float | None = None
    levels: list = field(default_factory=list)       # categorical levels kept for encoding
    group_levels: list = field(default_factory=list)  # categorical levels with their own group
    bin_edges: np.ndarray | None = None              # numeric grouping by quantile bins
    discrete_values: list = field(default_factory=list)  # low-cardinality numeric grouping
    group_labels: list = field(default_factory=list)


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    """Cramér's V between two integer-coded variables (equals |phi| for 2x2)."""
    table = pd.crosstab(a, b).to_numpy()
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n and k else 0.0


class DesignGeometry:
    """Groups, blocks and encodings fitted on trusted design records."""

    def __init__(self, n_bins: int = 4, max_discrete: int = 6,
                 min_group_frac: float = 0.03, assoc_threshold: float = 0.60):
        self.n_bins = n_bins
        self.max_discrete = max_discrete
        self.min_group_frac = min_group_frac
        self.assoc_threshold = assoc_threshold
        self.specs: dict[str, FeatureSpec] = {}
        self.features: list[str] = []
        self.blocks: list[list[str]] = []
        self.association: pd.DataFrame | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, design: pd.DataFrame, features: list[str]) -> "DesignGeometry":
        self.features = list(features)
        n = len(design)
        min_count = max(5, int(np.ceil(self.min_group_frac * n)))
        for f in self.features:
            col = design[f]
            if _is_numeric(col):
                med = float(np.nanmedian(col)) if col.notna().any() else 0.0
                filled = col.fillna(med).astype(float)
                spec = FeatureSpec(f, "numeric", median=med, mean=float(filled.mean()),
                                   std=float(filled.std(ddof=0)) or 1.0)
                uniq = np.sort(filled.unique())
                if len(uniq) <= self.max_discrete:
                    counts = filled.value_counts()
                    spec.discrete_values = [v for v in uniq if counts.get(v, 0) >= min_count]
                    spec.group_labels = [f"{f}={_fmt(v)}" for v in spec.discrete_values] + [f"{f}=other"]
                else:
                    qs = np.quantile(filled, np.linspace(0, 1, self.n_bins + 1)[1:-1])
                    edges = np.unique(qs)
                    spec.bin_edges = edges
                    spec.group_labels = _bin_labels(f, edges)
            else:
                filled = col.astype(object).where(col.notna(), MISSING).astype(str)
                counts = filled.value_counts()
                spec = FeatureSpec(f, "categorical")
                spec.levels = list(counts.index)
                spec.group_levels = [lv for lv in counts.index if counts[lv] >= min_count]
                spec.group_labels = [f"{f}={lv}" for lv in spec.group_levels] + [f"{f}={OTHER}"]
            self.specs[f] = spec

        self.association = self._association_matrix(design)
        self.blocks = self._blocks_from_association(self.association)
        return self

    # --------------------------------------------------------------- groups
    def group_codes(self, df: pd.DataFrame, feature: str) -> np.ndarray:
        """Integer group code per record for a candidate feature."""
        spec = self.specs[feature]
        col = df[feature]
        if spec.kind == "numeric":
            x = col.fillna(spec.median).astype(float).to_numpy()
            if spec.bin_edges is not None:
                return np.searchsorted(spec.bin_edges, x, side="right")
            codes = np.full(len(x), len(spec.discrete_values), dtype=int)
            for i, v in enumerate(spec.discrete_values):
                codes[np.isclose(x, v)] = i
            return codes
        x = col.astype(object).where(col.notna(), MISSING).astype(str).to_numpy()
        lookup = {lv: i for i, lv in enumerate(spec.group_levels)}
        other = len(spec.group_levels)
        return np.fromiter((lookup.get(v, other) for v in x), dtype=int, count=len(x))

    def group_labels(self, feature: str) -> list[str]:
        return self.specs[feature].group_labels

    def n_groups(self, feature: str) -> int:
        return len(self.specs[feature].group_labels)

    # ------------------------------------------------------------- encoding
    def encode(self, df: pd.DataFrame, exclude: set[str] | None = None) -> np.ndarray:
        """Standardised numeric columns plus one-hot categoricals (design levels)."""
        exclude = exclude or set()
        parts = []
        for f in self.features:
            if f in exclude:
                continue
            spec = self.specs[f]
            col = df[f]
            if spec.kind == "numeric":
                x = col.fillna(spec.median).astype(float).to_numpy()
                parts.append(((x - spec.mean) / spec.std)[:, None])
            else:
                x = col.astype(object).where(col.notna(), MISSING).astype(str).to_numpy()
                # drop the most frequent level as reference
                for lv in spec.levels[1:]:
                    parts.append((x == lv).astype(float)[:, None])
        if not parts:
            return np.zeros((len(df), 1))
        return np.hstack(parts)

    # ---------------------------------------------------------- association
    def _association_matrix(self, design: pd.DataFrame) -> pd.DataFrame:
        k = len(self.features)
        mat = np.eye(k)
        codes = {f: self.group_codes(design, f) for f in self.features}
        numeric = {}
        for f in self.features:
            if self.specs[f].kind == "numeric":
                numeric[f] = design[f].fillna(self.specs[f].median).astype(float).to_numpy()
        for i in range(k):
            for j in range(i + 1, k):
                fi, fj = self.features[i], self.features[j]
                if fi in numeric and fj in numeric:
                    xi, xj = numeric[fi], numeric[fj]
                    if np.std(xi) == 0 or np.std(xj) == 0:
                        v = 0.0
                    else:
                        v = abs(float(stats.spearmanr(xi, xj)[0]))
                else:
                    v = cramers_v(codes[fi], codes[fj])
                mat[i, j] = mat[j, i] = 0.0 if np.isnan(v) else v
        return pd.DataFrame(mat, index=self.features, columns=self.features)

    def _blocks_from_association(self, assoc: pd.DataFrame) -> list[list[str]]:
        """Connected components of the graph with edges where association >= threshold."""
        parent = {f: f for f in self.features}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, fi in enumerate(self.features):
            for fj in self.features[i + 1:]:
                if assoc.loc[fi, fj] >= self.assoc_threshold:
                    parent[find(fi)] = find(fj)
        comps: dict[str, list[str]] = {}
        for f in self.features:
            comps.setdefault(find(f), []).append(f)
        return list(comps.values())


def _fmt(v) -> str:
    v = float(v)
    return str(int(v)) if v.is_integer() else f"{v:.3g}"


def _bin_labels(f: str, edges: np.ndarray) -> list[str]:
    if len(edges) == 0:
        return [f"{f}:all"]
    labels = [f"{f}<{_fmt(edges[0])}"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels.append(f"{f}[{_fmt(lo)},{_fmt(hi)})")
    labels.append(f"{f}>={_fmt(edges[-1])}")
    return labels
