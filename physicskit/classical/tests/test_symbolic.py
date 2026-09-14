"""Direct unit tests for physicskit.classical.utils.symbolic.LagrangianEngine, in
isolation from any physicskit.classical.systems class.

These exist specifically because of a real bug found during
development: a spurious sign flip in the Legendre-transform code
produced a self-consistent but unphysical (negative-definite kinetic
term) Hamiltonian for CoupledOscillators. It was NOT caught by that
system's own energy-conservation test, because implicit_midpoint
faithfully conserves whatever Hamiltonian it is handed, even a wrong
one -- energy conservation alone cannot distinguish a correctly-derived
Hamiltonian from a self-consistent but wrong one. The tests below check
LagrangianEngine's output against independently-known-correct physics
(closed-form accelerations, finite-difference Hamiltonian gradients),
which does catch that class of bug.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from physicskit.classical.utils.symbolic import LagrangianEngine


def test_simple_harmonic_oscillator_acceleration_and_hamiltonian_signs():
    """L = (m/2) qdot^2 - (k/2) q^2 must give qddot = -(k/m) q, p = m*qdot,
    and H = p^2/(2m) + (k/2) q^2 with a POSITIVE kinetic term -- exactly
    the sign the historical bug got backwards."""
    m, k = 2.0, 3.0
    q, qdot = sp.symbols("q qdot")
    L = sp.Rational(1, 2) * m * qdot**2 - sp.Rational(1, 2) * k * q**2

    engine = LagrangianEngine([q], [qdot], L)

    q0, qdot0 = np.array([1.5]), np.array([0.7])
    accel = engine.acceleration_njit(q0, qdot0, 0.0)
    assert np.allclose(accel, [-(k / m) * q0[0]])

    p0 = engine.momentum_njit(q0, qdot0)
    assert np.allclose(p0, [m * qdot0[0]])

    H = engine.hamiltonian_njit(q0, p0, 0.0)
    expected_H = p0[0] ** 2 / (2 * m) + 0.5 * k * q0[0] ** 2
    assert np.isclose(H, expected_H)

    # The regression this guards against: a flipped Legendre transform
    # sign gives H = -3*T + V instead of T + V, which for qdot != 0
    # differs from the correct value by far more than float noise.
    wrong_H = -3 * (0.5 * m * qdot0[0] ** 2) + 0.5 * k * q0[0] ** 2
    assert not np.isclose(H, wrong_H)


def test_canonical_gradient_matches_finite_difference_of_hamiltonian():
    """For a non-trivial single pendulum, dH/dq and dH/dp from
    canonical_deriv_njit must match a central finite difference of
    hamiltonian_njit itself -- a general, formula-agnostic consistency
    check that would have caught the historical sign bug regardless of
    which specific system triggered it."""
    m, l, g = 1.3, 0.8, 9.81
    q, qdot = sp.symbols("q qdot")
    L = sp.Rational(1, 2) * m * l**2 * qdot**2 - m * g * l * (1 - sp.cos(q))

    engine = LagrangianEngine([q], [qdot], L)

    q0, p0 = np.array([0.6]), np.array([0.9])
    y0 = np.concatenate([q0, p0])
    dy = engine.canonical_deriv_njit(0.0, y0)
    dq_analytic, dp_analytic = dy[0], dy[1]

    eps = 1e-6

    def H(qa, pa):
        return float(engine.hamiltonian_njit(np.array([qa]), np.array([pa]), 0.0))

    dHdq_fd = (H(q0[0] + eps, p0[0]) - H(q0[0] - eps, p0[0])) / (2 * eps)
    dHdp_fd = (H(q0[0], p0[0] + eps) - H(q0[0], p0[0] - eps)) / (2 * eps)

    assert np.isclose(dq_analytic, dHdp_fd, atol=1e-6)
    assert np.isclose(dp_analytic, -dHdq_fd, atol=1e-6)


def test_two_dof_coupled_system_matches_finite_difference():
    """Same finite-difference consistency check, but for a genuinely
    coupled 2-DOF Lagrangian (unequal masses, linear coupling) -- the
    structurally closest case to the CoupledOscillators bug."""
    m1, m2, k = 1.7, 0.9, 2.5
    q0s, q1s, w0s, w1s = sp.symbols("q0 q1 w0 w1")
    L = sp.Rational(1, 2) * m1 * w0s**2 + sp.Rational(1, 2) * m2 * w1s**2 - sp.Rational(1, 2) * k * q0s**2 - sp.Rational(1, 2) * k * (q1s - q0s) ** 2
    engine = LagrangianEngine([q0s, q1s], [w0s, w1s], L)

    q0, p0 = np.array([0.3, -0.2]), np.array([0.5, 0.1])
    y0 = np.concatenate([q0, p0])
    dy = engine.canonical_deriv_njit(0.0, y0)
    dq_analytic, dp_analytic = dy[:2], dy[2:]

    eps = 1e-6

    def H(qa, pa):
        return float(engine.hamiltonian_njit(qa, pa, 0.0))

    dHdq_fd = np.array([(H(q0 + eps * e, p0) - H(q0 - eps * e, p0)) / (2 * eps) for e in np.eye(2)])
    dHdp_fd = np.array([(H(q0, p0 + eps * e) - H(q0, p0 - eps * e)) / (2 * eps) for e in np.eye(2)])

    assert np.allclose(dq_analytic, dHdp_fd, atol=1e-6)
    assert np.allclose(dp_analytic, -dHdq_fd, atol=1e-6)

    # Momentum should be the classical p_i = m_i * qdot_i for this
    # diagonal-mass-matrix system (a sign flip here would break it).
    qdot0 = np.array([0.4, -0.1])
    p_from_qdot = engine.momentum_njit(q0, qdot0)
    assert np.allclose(p_from_qdot, [m1 * qdot0[0], m2 * qdot0[1]])


def test_explicit_time_dependence():
    """A driven oscillator L = (1/2)qdot^2 - (1/2)q^2 + q*sin(t) must
    give the textbook forced equation qddot = -q + sin(t)."""
    t = sp.symbols("t")
    q, qdot = sp.symbols("q qdot")
    L = sp.Rational(1, 2) * qdot**2 - sp.Rational(1, 2) * q**2 + q * sp.sin(t)

    engine = LagrangianEngine([q], [qdot], L, t=t)

    q0, qdot0, t0 = np.array([0.4]), np.array([-0.2]), 1.1
    accel = engine.acceleration_njit(q0, qdot0, t0)
    assert np.allclose(accel, [-q0[0] + np.sin(t0)])
