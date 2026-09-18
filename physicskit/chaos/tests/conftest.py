"""Test-session setup: close every Matplotlib figure after each test.

This module's tests create many figures via the plotting/animation helpers
without ever closing them (a deliberate smoke-test simplicity -- see
CLAUDE.md), which otherwise trips Matplotlib's "more than 20 figures open"
warning partway through the suite.
"""

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")
