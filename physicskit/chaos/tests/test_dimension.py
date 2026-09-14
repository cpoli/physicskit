import numpy as np
import pytest

from physicskit.chaos.systems.maps import HenonMap
from physicskit.chaos.utils.dimension import box_counting_dimension, correlation_dimension


def _cantor_set_points(n_iter=10):
    """Midpoints of the intervals surviving `n_iter` steps of the middle-thirds
    Cantor set construction; its exact box-counting/correlation dimension is
    log(2)/log(3), giving a ground-truth target for both estimators."""
    intervals = [(0.0, 1.0)]
    for _ in range(n_iter):
        new_intervals = []
        for a, b in intervals:
            third = (b - a) / 3.0
            new_intervals.append((a, a + third))
            new_intervals.append((b - third, b))
        intervals = new_intervals
    return np.array([[(a + b) / 2.0] for a, b in intervals])


CANTOR_DIMENSION = np.log(2) / np.log(3)


def test_box_counting_dimension_matches_cantor_set_exact_value():
    points = _cantor_set_points(n_iter=10)
    d0, epsilons, counts = box_counting_dimension(points, n_scales=15, eps_min=3.0**-8, eps_max=3.0**-2)
    assert d0 == pytest.approx(CANTOR_DIMENSION, abs=0.05)
    assert epsilons.shape == (15,)
    assert counts.shape == (15,)
    assert np.all(counts >= 1.0)


def test_correlation_dimension_matches_cantor_set_exact_value():
    points = _cantor_set_points(n_iter=10)
    d2, _, correlation_sums = correlation_dimension(points, n_scales=15, eps_min=3.0**-8, eps_max=3.0**-2)
    assert d2 == pytest.approx(CANTOR_DIMENSION, abs=0.05)
    assert np.all((correlation_sums >= 0.0) & (correlation_sums <= 1.0))


def test_box_counting_dimension_of_a_line_segment_is_about_one():
    rng = np.random.default_rng(0)
    points = np.column_stack([rng.uniform(size=5000), np.zeros(5000)])
    d0, _, _ = box_counting_dimension(points)
    assert d0 == pytest.approx(1.0, abs=0.15)


def test_box_counting_dimension_of_a_filled_square_approaches_two():
    """With enough points to avoid box-counting's finite-sample saturation,
    a uniformly-filled 2D region should have dimension close to 2."""
    rng = np.random.default_rng(0)
    points = rng.uniform(size=(100000, 2))
    d0, _, _ = box_counting_dimension(points)
    assert d0 == pytest.approx(2.0, abs=0.15)


def test_henon_attractor_dimension_matches_known_literature_value():
    """The classic Henon attractor (a=1.4, b=0.3) has a well-documented
    fractal dimension of approximately 1.2-1.3."""
    system = HenonMap(a=1.4, b=0.3)
    traj = system.trajectory(np.array([0.0, 0.0]), n_iter=8000)
    traj = traj[500:]  # discard transient

    d0, _, _ = box_counting_dimension(traj)
    assert 1.1 < d0 < 1.4

    d2, _, _ = correlation_dimension(traj)
    assert 1.0 < d2 < 1.4
