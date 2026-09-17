"""Tests for physicskit.classical.systems.lagrangian's concrete systems and,
via them, the LagrangianSystem base class's own defaults that
test_conservation.py's energy-conservation checks never reach: direct
calls to ``acceleration()``/``derivatives()`` (every conservation test
goes through ``integrate()``, which calls the underlying njit
callbacks directly rather than through those convenience methods),
``ElasticPendulum.positions()``, and the ``rk4``/unknown-method
branches of ``LagrangianSystem.integrate()`` (every existing test uses
the default ``implicit_midpoint``).
"""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.classical.systems.lagrangian import BeadOnRotatingHoop, ElasticPendulum


def test_bead_on_rotating_hoop_acceleration_matches_euler_lagrange_by_hand():
    """theta'' = Omega^2 sin(theta) cos(theta) - (g/R) sin(theta), derived
    by hand from the Lagrangian in the class docstring."""
    system = BeadOnRotatingHoop(0.3, 0.1, R=1.5, omega=2.0, g=9.81)
    theta, thetadot = system.q, system.qdot
    accel = system.acceleration(theta, thetadot, 0.0)
    expected = system.omega**2 * np.sin(theta) * np.cos(theta) - (system.g / system.R) * np.sin(theta)
    np.testing.assert_allclose(accel, expected, rtol=1e-6)


def test_bead_on_rotating_hoop_derivatives_matches_qdot_and_acceleration():
    system = BeadOnRotatingHoop(0.3, 0.1, R=1.5, omega=2.0, g=9.81)
    d = system.derivatives(0.0, system.state)
    np.testing.assert_allclose(d[:1], system.qdot)
    np.testing.assert_allclose(d[1:], system.acceleration(system.q, system.qdot, 0.0))


def test_lagrangian_system_integrate_rk4_matches_implicit_midpoint_over_short_time():
    """rk4 is not symplectic, but over a short-enough window both
    integrators should agree closely -- the standard cross-check that
    the rk4 branch (which uses the direct (q, qdot) acceleration form,
    not the canonical Hamiltonian form implicit_midpoint uses) is wired
    correctly."""
    dt = 1e-4
    result_rk4 = BeadOnRotatingHoop(0.3, 0.0, omega=3.0).integrate((0.0, 0.05), dt=dt, method="rk4")
    result_im = BeadOnRotatingHoop(0.3, 0.0, omega=3.0).integrate((0.0, 0.05), dt=dt, method="implicit_midpoint")
    assert np.max(np.abs(result_rk4.q - result_im.q)) < 1e-5


def test_lagrangian_system_integrate_rejects_unknown_method():
    system = BeadOnRotatingHoop(0.3, 0.0, omega=3.0)
    with pytest.raises(ValueError):
        system.integrate((0.0, 1.0), dt=0.1, method="bogus")


def test_elastic_pendulum_positions_matches_polar_to_cartesian_by_hand():
    system = ElasticPendulum([0.2, 0.3], [0.0, 0.0], L0=1.0)
    x, y = system.positions()
    s, theta = system.q
    r = system.L0 + s
    assert x == pytest.approx(r * np.sin(theta))
    assert y == pytest.approx(-r * np.cos(theta))


def test_elastic_pendulum_positions_defaults_to_current_state_but_accepts_override():
    system = ElasticPendulum([0.2, 0.3], [0.0, 0.0], L0=1.0)
    default = system.positions()
    overridden = system.positions(np.array([0.0, 0.0]))
    np.testing.assert_allclose(overridden, [0.0, -1.0])
    assert not np.allclose(default, overridden)
