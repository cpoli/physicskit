"""Tests for physicskit.classical.systems.newtonian helpers that
test_conservation.py's energy-conservation checks never call directly:
ProjectileMotion.range_and_max_height() (a standalone closed-form
helper, not used by any trajectory check), and FoucaultPendulum's
position/velocity convenience properties and precession_period()."""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.classical.systems.newtonian import FoucaultPendulum, ProjectileMotion


def test_range_and_max_height_matches_analytic_trajectory_peak_and_landing():
    """Cross-check the closed-form range/max-height formula against the
    same speed/angle's full analytic_trajectory(), by finding where that
    trajectory peaks and lands."""
    speed, angle_deg, g = 20.0, 30.0, 9.81
    rng, max_height = ProjectileMotion.range_and_max_height(speed, angle_deg, g)

    t = np.linspace(0.0, 5.0, 200_000)
    x, y = ProjectileMotion.analytic_trajectory(speed, angle_deg, g, t)
    assert max_height == pytest.approx(y.max(), abs=1e-3)
    landing_idx = np.argmax(y[1:] < 0.0) + 1
    assert rng == pytest.approx(x[landing_idx], rel=1e-3)


def test_range_and_max_height_with_initial_height_raises_both():
    rng_ground, height_ground = ProjectileMotion.range_and_max_height(15.0, 45.0, 9.81, height=0.0)
    rng_elevated, height_elevated = ProjectileMotion.range_and_max_height(15.0, 45.0, 9.81, height=10.0)
    assert rng_elevated > rng_ground
    assert height_elevated == pytest.approx(height_ground + 10.0)


def test_foucault_pendulum_position_and_velocity_properties_match_state():
    system = FoucaultPendulum(q0=[1.0, 2.0], v0=[0.3, -0.4])
    np.testing.assert_allclose(system.position, [1.0, 2.0])
    np.testing.assert_allclose(system.velocity, [0.3, -0.4])


def test_foucault_pendulum_precession_period_matches_sidereal_day_over_sin_latitude():
    system = FoucaultPendulum(q0=[1.0, 0.0], v0=[0.0, 0.0], latitude_deg=48.85)
    expected = (2.0 * np.pi / system.omega_earth) / np.sin(np.deg2rad(48.85))
    assert system.precession_period() == pytest.approx(expected)
