"""Direct tests for the standalone helpers in
physicskit.classical.utils.conservation that test_conservation.py's
end-to-end energy-drift checks never call: energy_drift() (only its
relative sibling is used there), angular_momentum_2d()/
angular_momentum_drift() (no test currently checks angular momentum at
all), and relative_energy_drift()'s zero-initial-energy edge case."""

from __future__ import annotations

import numpy as np

from physicskit.classical.utils.conservation import (
    angular_momentum_2d,
    angular_momentum_drift,
    energy_drift,
    relative_energy_drift,
)


def test_energy_drift_is_absolute_deviation_from_initial_value():
    energy = np.array([10.0, 10.5, 9.5, 10.0])
    drift = energy_drift(energy)
    np.testing.assert_allclose(drift, [0.0, 0.5, 0.5, 0.0])


def test_relative_energy_drift_is_infinite_when_initial_energy_is_zero():
    energy = np.array([0.0, 1.0, -1.0])
    drift = relative_energy_drift(energy)
    assert np.all(np.isinf(drift))


def test_angular_momentum_2d_matches_cross_product_formula():
    q = np.array([[1.0, 0.0], [0.0, 2.0]])
    p = np.array([[0.0, 1.0], [3.0, 0.0]])
    L = angular_momentum_2d(q, p)
    np.testing.assert_allclose(L, [1.0 * 1.0 - 0.0 * 0.0, 0.0 * 0.0 - 2.0 * 3.0])


def test_angular_momentum_drift_is_zero_for_a_conserved_circular_orbit():
    """A particle moving on a circle at constant angular speed has
    exactly conserved angular momentum, so drift should be exactly
    zero at every sample."""
    theta = np.linspace(0.0, 2.0 * np.pi, 50, endpoint=False)
    q = np.column_stack([np.cos(theta), np.sin(theta)])
    p = np.column_stack([-np.sin(theta), np.cos(theta)])
    drift = angular_momentum_drift(q, p)
    np.testing.assert_allclose(drift, 0.0, atol=1e-12)
