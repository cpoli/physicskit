"""Direct unit tests for physicskit.classical.core.integrators, independent of any
physicskit.classical.systems class. These exercise the four integrators (and the
make_separable_derivatives helper) against a plain harmonic oscillator
and a simple exponential-decay ODE with known analytic solutions, so a
regression here is caught at the numerical-core level rather than only
showing up as a mysterious failure in some system built on top of it.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from physicskit.classical.core.integrators import (
    implicit_midpoint_integrate,
    implicit_midpoint_step,
    make_separable_derivatives,
    rk4_integrate,
    velocity_verlet_integrate,
    yoshida4_integrate,
)


@njit(cache=False)
def _sho_force(q, t):
    return -q  # harmonic oscillator, k = m = 1


@njit(cache=False)
def _exp_decay_deriv(t, y):
    return -y


def _sho_energy(q, p):
    return 0.5 * q**2 + 0.5 * p**2


def test_rk4_matches_analytic_exponential_decay():
    y0 = np.array([1.0])
    dt = 1e-3
    n_steps = 1000
    ts, ys = rk4_integrate(_exp_decay_deriv, y0, 0.0, n_steps, dt)
    analytic = np.exp(-ts)
    assert np.max(np.abs(ys[:, 0] - analytic)) < 1e-9


def test_verlet_and_yoshida4_conserve_harmonic_oscillator_energy():
    q0, p0 = np.array([1.0]), np.array([0.0])
    dt, n_steps = 0.01, 50_000
    mass_inv = 1.0

    ts, qs, ps = velocity_verlet_integrate(_sho_force, mass_inv, q0, p0, 0.0, n_steps, dt)
    E = _sho_energy(qs[:, 0], ps[:, 0])
    assert np.max(np.abs(E - E[0])) / E[0] < 1e-3  # 2nd order, bounded oscillation

    ts, qs, ps = yoshida4_integrate(_sho_force, mass_inv, q0, p0, 0.0, n_steps, dt)
    E4 = _sho_energy(qs[:, 0], ps[:, 0])
    assert np.max(np.abs(E4 - E4[0])) / E4[0] < 1e-8  # 4th order: much tighter at the same dt


def test_yoshida4_more_accurate_than_verlet_at_fixed_dt():
    """4th order should beat 2nd order by several orders of magnitude
    at a dt small enough for both to be in their asymptotic regime."""
    q0, p0 = np.array([1.0]), np.array([0.0])
    dt, n_steps = 0.02, 5_000
    mass_inv = 1.0

    _, qs_v, ps_v = velocity_verlet_integrate(_sho_force, mass_inv, q0, p0, 0.0, n_steps, dt)
    _, qs_y, ps_y = yoshida4_integrate(_sho_force, mass_inv, q0, p0, 0.0, n_steps, dt)

    drift_v = np.max(np.abs(_sho_energy(qs_v[:, 0], ps_v[:, 0]) - 0.5))
    drift_y = np.max(np.abs(_sho_energy(qs_y[:, 0], ps_y[:, 0]) - 0.5))
    assert drift_y < drift_v * 1e-3


def test_implicit_midpoint_conserves_harmonic_oscillator_energy():
    y0 = np.array([1.0, 0.0])

    @njit(cache=False)
    def deriv(t, y):
        out = np.empty(2)
        out[0] = y[1]
        out[1] = -y[0]
        return out

    dt, n_steps = 0.01, 50_000
    ts, ys = implicit_midpoint_integrate(deriv, y0, 0.0, n_steps, dt)
    E = _sho_energy(ys[:, 0], ys[:, 1])
    assert np.max(np.abs(E - E[0])) / E[0] < 1e-10


def test_implicit_midpoint_step_exactly_preserves_norm_for_a_rotation_generator():
    """For dy/dt = A y with A skew-symmetric (a pure rotation vector
    field), |y|^2 is a quadratic invariant; a single implicit-midpoint
    step -- solved via the Cayley transform of A, not the matrix
    exponential -- should still preserve |y| exactly (to machine
    precision) even though it does not reproduce the exact rotation
    angle. This is a direct check of the step formula itself, distinct
    from the aggregate energy-conservation checks elsewhere."""

    @njit(cache=False)
    def deriv(t, y):
        out = np.empty(2)
        out[0] = -y[1]
        out[1] = y[0]
        return out

    y0 = np.array([1.0, 0.0])
    dt = 0.1
    y1 = implicit_midpoint_step(deriv, 0.0, y0, dt)

    assert abs(np.linalg.norm(y1) - np.linalg.norm(y0)) < 1e-12
    assert y1[1] > 0  # rotated counter-clockwise, matching the sign of the generator


def test_make_separable_derivatives_matches_verlet_force():
    mass_inv = 1.0
    ndof = 1
    deriv = make_separable_derivatives(_sho_force, mass_inv, ndof)
    y = np.array([0.5, -0.3])
    dy = deriv(0.0, y)
    # dq/dt = p/m, dp/dt = force(q) = -q
    assert np.allclose(dy, [-0.3, -0.5])
