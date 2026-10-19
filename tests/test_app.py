"""Headless tests of the Streamlit app using streamlit.testing.AppTest.

These run app.py exactly as Streamlit would, without opening a browser.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _run():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _page_text(at) -> str:
    return " ".join(m.value for m in at.markdown)


def test_app_renders_all_tabs():
    at = _run()
    assert len(at.tabs) == 4
    assert "Predicted species" in _page_text(at)


def test_preset_button_changes_prediction():
    at = _run()
    at.button(key="preset_0").click().run()  # typical setosa
    assert not at.exception
    assert "Iris setosa" in _page_text(at)
    at.button(key="preset_100").click().run()  # typical virginica
    assert "Iris virginica" in _page_text(at)


def test_out_of_range_input_shows_warning():
    at = _run()
    at.slider(key="in_petal_length").set_value(9.5).run()
    assert not at.exception
    assert any("Outside the training data" in w.value for w in at.warning)
