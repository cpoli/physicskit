"""Test-session setup: force Matplotlib's non-interactive Agg backend
before anything imports pyplot, so the suite (and CI, which has no
display) can exercise the plotting/animation code headlessly."""

import matplotlib

matplotlib.use("Agg")
