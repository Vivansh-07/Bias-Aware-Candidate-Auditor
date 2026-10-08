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
