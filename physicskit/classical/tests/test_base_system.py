"""Tests for the DynamicalSystem/ODESystem/HamiltonianSystem base-class
machinery in physicskit.classical.core.base_system that every concrete system
uses but that no single concrete system exercises end to end:
``reset()`` (never called by any real system or example), the
``force()``/``derivatives()`` defensive defaults for a non-separable
Hamiltonian system with no ``_deriv_njit`` of its own (every real
HamiltonianSystem subclass in this package is separable and never hits
them), and both base classes' "unknown integration method" guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.classical.core.base_system import HamiltonianSystem
from physicskit.classical.systems.newtonian import FoucaultPendulum, ProjectileMotion


class _MinimalNonSeparableHamiltonian(HamiltonianSystem):
    """Smallest possible non-separable HamiltonianSystem subclass. Every
    real system in this package is separable and provides its own
    ``_force_njit``/``_deriv_njit``, so the base class's defensive
    defaults for the non-separable, nothing-provided case are otherwise
    dead code."""

    separable = False

    def __init__(self):
        super().__init__([0.0], [0.0])

    def kinetic_energy(self, p):
        return 0.5 * p[0] ** 2

    def potential_energy(self, q):
        return 0.0


def test_hamiltonian_system_reset_updates_state_and_time():
    system = ProjectileMotion(q0=[0.0, 0.0], p0=[1.0, 1.0])
    new_state = system.reset(state0=[5.0, 5.0, 2.0, 2.0], t0=3.0)
    np.testing.assert_allclose(new_state, [5.0, 5.0, 2.0, 2.0])
    assert system.t == 3.0
    assert system.state is new_state


def test_reset_with_no_state0_keeps_current_state_but_updates_time():
    system = ProjectileMotion(q0=[1.0, 2.0], p0=[3.0, 4.0])
    original_state = system.state
    returned = system.reset(t0=7.0)
    np.testing.assert_allclose(returned, original_state)
    assert system.t == 7.0


def test_hamiltonian_system_force_raises_for_non_separable_system_without_force():
    system = _MinimalNonSeparableHamiltonian()
    with pytest.raises(NotImplementedError):
        system.force(system.q, 0.0)


def test_hamiltonian_system_derivatives_raises_for_non_separable_system_without_deriv():
    system = _MinimalNonSeparableHamiltonian()
    with pytest.raises(NotImplementedError):
        system.derivatives(0.0, system.state)


def test_hamiltonian_system_integrate_rejects_symplectic_method_for_non_separable_system():
    system = _MinimalNonSeparableHamiltonian()
    with pytest.raises(ValueError):
        system.integrate((0.0, 1.0), dt=0.1, method="verlet")


def test_hamiltonian_system_integrate_rejects_unknown_method():
    system = ProjectileMotion(q0=[0.0, 0.0], p0=[1.0, 1.0])
    with pytest.raises(ValueError):
        system.integrate((0.0, 1.0), dt=0.1, method="bogus")


def test_hamiltonian_system_force_matches_gravity_for_separable_system():
    system = ProjectileMotion(q0=[0.0, 10.0], p0=[1.0, 0.0])
    fx, fy = system.force(system.q, 0.0)
    assert fx == pytest.approx(0.0)
    assert fy == pytest.approx(-system.m * system.g)


def test_hamiltonian_system_derivatives_matches_auto_built_separable_deriv():
    """ProjectileMotion without drag never sets self._deriv_njit itself,
    so .derivatives() must fall back to auto-building it from force()
    (HamiltonianSystem._ensure_deriv_njit's separable branch)."""
    system = ProjectileMotion(q0=[0.0, 10.0], p0=[1.0, 0.0])
    d = system.derivatives(0.0, system.state)
    assert d.shape == (4,)
    vx, vy, fx, fy = d
    assert vx == pytest.approx(system.p[0] * system.mass_inv)
    assert vy == pytest.approx(system.p[1] * system.mass_inv)
    assert fx == pytest.approx(0.0)
    assert fy == pytest.approx(-system.m * system.g)


def test_ode_system_derivatives_matches_foucault_equations_by_hand():
    system = FoucaultPendulum(q0=[1.0, 0.0], v0=[0.5, -0.2])
    d = system.derivatives(0.0, system.state)
    x, y, vx, vy = system.state
    assert d[0] == pytest.approx(vx)
    assert d[1] == pytest.approx(vy)
    assert d[2] == pytest.approx(-(system.omega0**2) * x + 2 * system.omega_z * vy)
    assert d[3] == pytest.approx(-(system.omega0**2) * y - 2 * system.omega_z * vx)


def test_ode_system_integrate_rejects_unknown_method():
    system = FoucaultPendulum(q0=[1.0, 0.0], v0=[0.0, 0.0])
    with pytest.raises(ValueError):
        system.integrate((0.0, 1.0), dt=0.1, method="bogus")
