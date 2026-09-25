import numpy as np
import pytest

from physicskit.fluids.exceptions import InvalidParameterError
from physicskit.fluids.systems.vortex_dynamics import (
    PointVortexSystem,
    point_vortex_velocities,
    von_karman_vortex_street,
)


def test_two_equal_vortices_orbit_at_the_analytic_angular_velocity():
    """Two point vortices of equal strength Gamma separated by distance d
    rotate rigidly about their midpoint at angular velocity Omega = Gamma / (pi d^2)."""
    Gamma, d = 1.0, 1.0
    positions0 = np.array([[d / 2, 0.0], [-d / 2, 0.0]])
    system = PointVortexSystem(positions=positions0, circulations=[Gamma, Gamma])
    omega_expected = Gamma / (np.pi * d**2)
    t_final = 0.2
    dt, n_steps = 1e-4, round(t_final / 1e-4)
    _, trajectory = system.trajectory(dt=dt, n_steps=n_steps)
    final = trajectory[-1]
    d_final = np.hypot(*(final[0] - final[1]))
    assert d_final == pytest.approx(d, rel=1e-3)  # separation is conserved
    angle_final = np.arctan2(final[0, 1], final[0, 0])
    assert angle_final == pytest.approx(omega_expected * t_final, abs=1e-3)


def test_opposite_sign_vortex_pair_translates_without_rotating():
    """A vortex dipole (equal and opposite circulation) advects itself in a
    straight line at constant velocity Gamma / (2 pi d), rather than orbiting."""
    Gamma, d = 1.0, 1.0
    positions0 = np.array([[0.0, d / 2], [0.0, -d / 2]])
    system = PointVortexSystem(positions=positions0, circulations=[Gamma, -Gamma])
    dt, n_steps = 1e-3, 200
    _, trajectory = system.trajectory(dt=dt, n_steps=n_steps)
    displacement = trajectory[-1] - trajectory[0]
    # both vortices move by (nearly) the same vector: rigid translation
    np.testing.assert_allclose(displacement[0], displacement[1], atol=1e-3)
    speed_expected = Gamma / (2 * np.pi * d)
    speed_numeric = np.hypot(*displacement[0]) / (dt * n_steps)
    assert speed_numeric == pytest.approx(speed_expected, rel=1e-2)


def test_point_vortex_velocities_rejects_mismatched_lengths():
    with pytest.raises(InvalidParameterError):
        point_vortex_velocities(positions=[[0.0, 0.0], [1.0, 0.0]], circulations=[1.0])


def test_point_vortex_system_rejects_mismatched_lengths():
    with pytest.raises(InvalidParameterError):
        PointVortexSystem(positions=[[0.0, 0.0], [1.0, 0.0]], circulations=[1.0])


def test_point_vortex_system_velocities_matches_module_function():
    positions = [[0.0, 0.0], [1.0, 0.0]]
    circulations = [1.0, -1.0]
    system = PointVortexSystem(positions=positions, circulations=circulations)
    np.testing.assert_allclose(system.velocities(), point_vortex_velocities(positions, circulations))


def test_point_vortex_system_repr():
    system = PointVortexSystem(positions=[[0.0, 0.0]], circulations=[2.0])
    assert repr(system) == "PointVortexSystem(n_vortices=1, circulations=array([2.]))"


def test_von_karman_street_rejects_nonpositive_spacing():
    with pytest.raises(InvalidParameterError):
        von_karman_vortex_street(n_pairs=3, spacing_l=0.0)


def test_von_karman_street_rows_have_uniform_opposite_signs():
    positions, circulations = von_karman_vortex_street(n_pairs=4, spacing_l=1.0)
    top_row = circulations[0::2]
    bottom_row = circulations[1::2]
    np.testing.assert_allclose(top_row, -1.0)
    np.testing.assert_allclose(bottom_row, 1.0)


def test_von_karman_street_translates_at_the_classical_speed():
    # Deep inside a long street, every vortex moves along -x at U = (Gamma / 2l) tanh(pi h / l).
    n_pairs, spacing_l = 200, 1.0
    positions, circulations = von_karman_vortex_street(n_pairs=n_pairs, spacing_l=spacing_l)
    velocities = point_vortex_velocities(positions, circulations)
    middle = velocities[[n_pairs, n_pairs + 1]]
    h = positions[0, 1] - positions[1, 1]
    U = np.tanh(np.pi * h / spacing_l) / (2.0 * spacing_l)
    np.testing.assert_allclose(middle[:, 0], -U, rtol=5e-3)
    np.testing.assert_allclose(middle[:, 1], 0.0, atol=5e-3)


def test_von_karman_street_uses_the_stable_spacing_ratio_by_default():
    spacing_l = 2.0
    positions, _ = von_karman_vortex_street(n_pairs=3, spacing_l=spacing_l)
    h = positions[0, 1] - positions[1, 1]
    assert h / spacing_l == pytest.approx(0.2805, abs=1e-4)


def test_von_karman_street_rejects_too_few_pairs():
    with pytest.raises(InvalidParameterError):
        von_karman_vortex_street(n_pairs=1, spacing_l=1.0)
