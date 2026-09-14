"""Smoke tests for the Streamlit explorer app, run headlessly via
``streamlit.testing.v1.AppTest`` (no browser or server needed)."""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def _run() -> "st_testing.AppTest":
    at = st_testing.AppTest.from_file(APP_PATH)
    at.run(timeout=120)
    return at


def test_app_loads_with_no_exceptions_in_default_state():
    at = _run()
    assert not at.exception


@pytest.mark.parametrize("demo", ["Billiards", "Continuous systems", "Bifurcation diagrams", "Basins of attraction"])
def test_each_demo_runs_with_no_exceptions(demo):
    at = _run()
    at.selectbox[0].set_value(demo).run(timeout=120)
    assert not at.exception


def test_every_billiard_shape_runs_with_no_exceptions():
    at = _run()
    for option in list(at.selectbox[1].options):
        at.selectbox[1].set_value(option).run(timeout=120)
        assert not at.exception, f"{option} raised: {list(at.exception)}"


def test_every_map_and_sweep_parameter_runs_with_no_exceptions():
    at = _run()
    at.selectbox[0].set_value("Bifurcation diagrams").run(timeout=120)
    for map_name in list(at.selectbox[1].options):
        at.selectbox[1].set_value(map_name).run(timeout=120)
        for param_name in list(at.selectbox[2].options):
            at.selectbox[2].set_value(param_name).run(timeout=120)
            assert not at.exception, f"{map_name}/{param_name} raised: {list(at.exception)}"
