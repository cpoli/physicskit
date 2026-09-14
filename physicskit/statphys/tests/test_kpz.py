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
