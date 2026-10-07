"""Semi-synthetic splitting and the mitigation stage."""
import numpy as np
import pandas as pd

from cfa.mitigation import audit_guided_weights, evaluate_strategies
from cfa.realdata import DATASETS, CorruptionSpec, ShiftSpec, feature_columns, load_dataset, make_semisynthetic
from cfa.v6 import AuditConfig, audit_v6


def _compas(corr_rate=0.3, shift=None, seed=3):
    info = DATASETS["COMPAS"]
    raw = load_dataset("COMPAS")
    corr = CorruptionSpec(info.target_feature, info.target_values, info.flip_from, info.flip_to, corr_rate)
    data, test = make_semisynthetic(raw, info.label, corruption=corr, shift=shift, seed=seed)
    return info, raw, corr, data, test


def test_splits_are_record_disjoint_and_corruption_only_in_audit():
    info, raw, corr, data, test = _compas()
    assert len(data) + len(test) <= len(raw)
    flipped = data["y"] != data["y_clean"]
    assert flipped.sum() > 0
    assert (data.loc[flipped, "source"] == 1).all()
    assert data.loc[flipped, "race"].isin(corr.values).all()
    assert (data.loc[flipped, "y_clean"] == corr.flip_from).all()
    assert (test["y"] == test["y_clean"]).all()


def test_shift_changes_audit_covariates_not_labels():
    info, raw, corr, data, test = _compas(corr_rate=0.0, shift=ShiftSpec("age", 1.5))
    trusted_age = data.loc[data.source == 0, "age"].mean()
    audit_age = data.loc[data.source == 1, "age"].mean()
    assert audit_age > trusted_age + 3
    assert (data["y"] == data["y_clean"]).all()


def test_german_credit_codes_decoded():
    df = load_dataset("German Credit")
    assert "female: div/sep/married" in set(df["personal_status"])


def test_mitigation_pipeline_on_compas():
    info, raw, corr, data, test = _compas()
    feats = feature_columns(data)
    res = audit_v6(data, feats, config=AuditConfig(seed=3))
    assert res.selects("race")
    wr = audit_guided_weights(data, feats, res)
    assert wr.accepted and 0.7 <= wr.ess_ratio <= 1.0
    assert wr.weights.min() >= 0.5 / wr.weights.mean() - 1e-9
    ev = evaluate_strategies(data, test, feats, res, corr.mask(test), weights=wr)
    assert set(ev["strategy"]) >= {"Baseline (audit labels)", "Audit-guided reweighting", "Oracle (clean audit labels)"}
    lr = ev[ev.model == "Logistic regression"].set_index("strategy")
    # reweighting should move the corrupted group's prediction bias toward zero
    assert abs(lr.loc["Audit-guided reweighting", "group_pred_bias"]) < abs(lr.loc["Baseline (audit labels)", "group_pred_bias"])


def test_abstention_closes_mitigation_gate():
    info, raw, corr, data, test = _compas(corr_rate=0.0)
    feats = feature_columns(data)
    res = audit_v6(data, feats, config=AuditConfig(seed=3))
    wr = audit_guided_weights(data, feats, res)
    assert wr.weights is None and not wr.accepted
