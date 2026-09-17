import numpy as np

from physicskit.statphys.chapters.kpz_growth import KPZInterface


def test_initial_heights_are_zero():
    interface = KPZInterface(L=64, seed=0)
    assert np.all(interface.heights == 0)
    assert interface.width() == 0.0


def test_rsos_constraint_is_maintained_after_growth():
    interface = KPZInterface(L=64, seed=1)
    interface.grow(n_sweeps=500)
    h = interface.heights
    diffs = np.abs(h - np.roll(h, -1))
    assert diffs.max() <= 1


def test_width_grows_from_zero():
    interface = KPZInterface(L=64, seed=2)
    interface.grow(n_sweeps=200)
    assert interface.width() > 0.0


def test_time_advances_with_grow():
    interface = KPZInterface(L=32, seed=3)
    interface.grow(n_sweeps=10)
    assert interface.time == 10


def test_run_growth_curve_shapes_and_monotonic_time():
    interface = KPZInterface(L=64, seed=4)
    times, widths = interface.run_growth_curve(t_max=500, n_points=20)
    assert times.shape == widths.shape
    assert np.all(np.diff(times) > 0)
    assert np.all(widths >= 0.0)


def test_run_growth_curve_linearly_spaced_checkpoints():
    interface = KPZInterface(L=64, seed=5)
    times, widths = interface.run_growth_curve(t_max=200, n_points=10, log_spaced=False)
    assert times.shape == widths.shape
    assert np.all(np.diff(times) > 0)
    # linearly (not logarithmically) spaced: consecutive gaps stay within
    # +/-1 of each other after integer rounding, unlike the geometrically
    # growing gaps of the log-spaced default.
    assert np.ptp(np.diff(times)) <= 1


def test_mean_height_grows_with_deposition():
    interface = KPZInterface(L=64, seed=6)
    assert interface.mean_height() == 0.0
    interface.grow(n_sweeps=200)
    assert interface.mean_height() > 0.0
