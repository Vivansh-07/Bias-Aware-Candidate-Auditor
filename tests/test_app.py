"""Headless smoke test: every page of the web app renders without an exception."""
import os

import pytest
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
PAGES = ["🏠 Overview", "🧪 Synthetic Lab", "🌍 Real-Data Audit", "📊 Calibration", "📖 Method & Limits"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.message for e in at.exception]


def test_synthetic_audit_runs_and_flags_corruption():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("🧪 Synthetic Lab").run()
    at.selectbox[0].set_value("Strong corruption (40%, n=10k)").run()
    next(b for b in at.button if "Run audit" in b.label).click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("flag for human review" in w.value for w in at.warning)


def _audited_synthetic(preset="Label corruption (30%)"):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("🧪 Synthetic Lab").run()
    at.selectbox[0].set_value(preset).run()
    next(b for b in at.button if "Run audit" in b.label).click().run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def _flagged(at):
    return any("flag for human review" in w.value for w in at.warning)


def test_results_hidden_once_settings_change():
    at = _audited_synthetic()
    assert _flagged(at)
    at.selectbox[0].set_value("Matched clean control").run()
    assert not _flagged(at), "results of the corruption run shown under the clean-control preset"
    assert any("Run audit" in i.value for i in at.info)


def test_selection_and_results_survive_page_switch():
    at = _audited_synthetic()
    at.sidebar.radio[0].set_value("📊 Calibration").run()
    at.sidebar.radio[0].set_value("🧪 Synthetic Lab").run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.selectbox[0].value == "Label corruption (30%)"
    assert _flagged(at)
    assert not any("Session State API" in w.value for w in at.warning)


@pytest.mark.parametrize("page,key", [("🧪 Synthetic Lab", "syn"), ("🌍 Real-Data Audit", "real")])
def test_results_saved_by_an_older_app_version_do_not_crash(page, key):
    # A browser session that stays open while a new version is deployed keeps results saved without "settings".
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state[key] = {"r6": None}
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("Run audit" in i.value for i in at.info)


def test_real_data_results_hidden_once_settings_change():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("🌍 Real-Data Audit").run()
    next(b for b in at.button if "Run audit" in b.label).click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("V6: flag for human review" in w.value and "race" in w.value for w in at.warning)
    at.number_input[0].set_value(4).run()
    assert not _flagged(at)
    assert any("Run audit" in i.value for i in at.info)


def test_new_audit_clears_old_mitigation_results():
    at = _audited_synthetic()
    next(b for b in at.button if "Train & evaluate" in b.label).click().run()
    assert any(m.label == "Weights accepted" for m in at.metric)
    at.selectbox[0].set_value("Matched clean control").run()
    next(b for b in at.button if "Run audit" in b.label).click().run()
    assert not _flagged(at)
    assert not any(m.label == "Weights accepted" for m in at.metric), "mitigation of the previous audit still shown"
