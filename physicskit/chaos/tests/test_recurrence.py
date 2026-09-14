import numpy as np
import pytest

from physicskit.chaos.systems.maps import HenonMap
from physicskit.chaos.utils.recurrence import kac_lemma_estimate, recurrence_matrix, recurrence_times


def test_recurrence_matrix_is_symmetric_with_true_diagonal():
    rng = np.random.default_rng(0)
    states = rng.uniform(size=(50, 2))
    matrix = recurrence_matrix(states, epsilon=0.3)
    assert matrix.shape == (50, 50)
    assert matrix.dtype == np.bool_
    np.testing.assert_array_equal(matrix, matrix.T)
    assert np.all(np.diag(matrix))  # every point recurs to itself at distance 0


def test_recurrence_matrix_epsilon_zero_is_only_the_diagonal():
    rng = np.random.default_rng(0)
    states = rng.uniform(size=(20, 2))
    matrix = recurrence_matrix(states, epsilon=1e-12)
    np.testing.assert_array_equal(matrix, np.eye(20, dtype=bool))


def test_recurrence_times_detects_a_simple_periodic_orbit():
    """A trajectory that revisits the same three points in a cycle should
    show a recurrence time of exactly the period (3 steps) every time."""
    states = np.tile(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), (10, 1))
    times = recurrence_times(states, epsilon=0.01, reference_idx=0, dt=1.0)
    assert times.size > 0
    np.testing.assert_allclose(times, 3.0)


def test_recurrence_times_scales_with_dt():
    states = np.tile(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), (10, 1))
    times_dt1 = recurrence_times(states, epsilon=0.01, dt=1.0)
    times_dt2 = recurrence_times(states, epsilon=0.01, dt=0.5)
    np.testing.assert_allclose(times_dt2, times_dt1 * 0.5)


def test_kac_lemma_holds_for_a_long_henon_trajectory():
    """Kac's lemma predicts mean_recurrence_time * measure(epsilon-ball) ~= 1
    for an ergodic, measure-preserving trajectory sampled long enough."""
    system = HenonMap(a=1.4, b=0.3)
    traj = system.trajectory(np.array([0.1, 0.1]), n_iter=100000)
    traj = traj[500:]  # discard transient

    mean_time, measure, kac_product = kac_lemma_estimate(traj, epsilon=0.05, reference_idx=0)
    assert mean_time > 0.0
    assert 0.0 < measure < 1.0
    assert kac_product == pytest.approx(1.0, abs=0.1)
