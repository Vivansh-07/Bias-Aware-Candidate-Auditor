"""Behavioural checks of the V6 and V5 auditors on the synthetic family."""
from dataclasses import replace

import numpy as np
import pytest

from cfa.geometry import DesignGeometry
from cfa.simulate import DEFAULT, feature_names, generate
from cfa.v5 import audit_v5
from cfa.v6 import ABSTAIN_MESSAGE, AuditConfig, assign_folds, audit_v6, split_design


def test_proxy_association_matches_report():
    # Report: prevalence 25%, 10% proxy flips -> population association ~0.756 (< 0.80, > 0.60)
    df = generate(replace(DEFAULT, n_trusted=40000, n_audit=10), seed=1)
    geom = DesignGeometry().fit(df[df.source == 0], ["a", "p"])
    assoc = geom.association.loc["a", "p"]
    assert assoc == pytest.approx(0.756, abs=0.01)
    assert ["a", "p"] in [sorted(b) for b in geom.blocks]


def test_design_rows_are_trusted_only_and_folds_balanced():
    df = generate(DEFAULT, seed=2)
    rng = np.random.default_rng(0)
    design, analysis = split_design(df, "source", 0.25, rng)
    assert (df.loc[design, "source"] == 0).all()
    assert design.sum() == round(0.25 * (df.source == 0).sum())
    assert not np.any(design & analysis)
    s = df.loc[analysis, "source"].to_numpy()
    folds = assign_folds(s, 2, rng)
    for v in (0, 1):
        counts = np.bincount(folds[s == v])
        assert abs(counts[0] - counts[1]) <= 1


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_v6_recovers_strong_corruption(seed):
    sc = replace(DEFAULT, corruption=0.40, n_audit=10000, n_trusted=3333)
    res = audit_v6(generate(sc, seed=seed), feature_names(sc), config=AuditConfig(seed=seed))
    assert res.selects("a")


def test_v6_effect_size_sign_and_magnitude():
    sc = replace(DEFAULT, corruption=0.40, n_audit=10000, n_trusted=3333)
    res = audit_v6(generate(sc, seed=8), feature_names(sc), config=AuditConfig(seed=8))
    g = res.groups[res.groups.feature == "a"].set_index("group")
    assert g.loc["a=1", "label_gap"] < -0.1          # audit positives were removed in a = 1
    assert abs(g.loc["a=0", "label_gap"]) < 0.05


def test_v6_abstains_on_clean_control():
    res = audit_v6(generate(DEFAULT, seed=11), feature_names(DEFAULT), config=AuditConfig(seed=11))
    assert not res.flagged
    assert res.decision == ABSTAIN_MESSAGE


def test_unsupported_sources_force_abstention():
    sc = replace(DEFAULT, n_trusted=120, n_audit=2000, corruption=0.5)
    res = audit_v6(generate(sc, seed=1), feature_names(sc), config=AuditConfig(seed=1))
    assert not res.diagnostics["source_support_ok"]
    assert not res.flagged
    assert (res.blocks["p_block"] == 1).all()


def test_v5_recovers_strong_corruption_and_reports_permutation_floor():
    sc = replace(DEFAULT, corruption=0.40, n_audit=10000, n_trusted=3333)
    res = audit_v5(generate(sc, seed=6), feature_names(sc), config=AuditConfig(seed=6), B=999)
    assert res.selects("a")
    assert res.blocks["p_block"].min() >= 1 / 1000
