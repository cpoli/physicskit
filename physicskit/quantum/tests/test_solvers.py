"""Tests for physicskit.quantum.core.solvers: the time-dependent-potential
branch of SplitOperatorSolver2D (test_unitarity.py only exercises the
static-potential path) and _accepts_time's non-introspectable-callable
fallback.
"""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.quantum.core.solvers import SplitOperatorSolver2D, _accepts_time


def _gaussian_2d(X, Y, x0=0.0, y0=0.0, sigma=1.0):
    norm = (2 * np.pi * sigma**2) ** (-0.5)
    return norm * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))


def test_accepts_time_detects_three_argument_signature():
    def V_static(x, y):
        return x + y

    def V_time_dependent(x, y, t):
        return x + y + t

    assert _accepts_time(V_static, n_spatial_args=2) is False
    assert _accepts_time(V_time_dependent, n_spatial_args=2) is True


def test_accepts_time_falls_back_to_false_for_non_introspectable_callable():
    # inspect.signature(min) raises ValueError (no signature metadata for
    # this builtin), and _accepts_time should fall back to False rather
    # than propagating the exception.
    assert _accepts_time(min, n_spatial_args=2) is False


def test_split_operator_2d_with_time_dependent_potential_conserves_norm():
    x = np.linspace(-8, 8, 64)
    y = np.linspace(-8, 8, 64)

    def V(X, Y, t):
        return 0.5 * (X**2 + Y**2) * (1.0 + 0.1 * np.sin(t))

    solver = SplitOperatorSolver2D(x, y, V, dt=1e-3)
    assert solver._time_dependent is True

    X, Y = np.meshgrid(x, y, indexing="ij")
    psi = _gaussian_2d(X, Y).astype(complex)
    psi /= np.sqrt(solver.norm(psi))

    t = 0.0
    for _ in range(50):
        psi = solver.step(psi, t)
        t += solver.dt

    assert solver.norm(psi) == pytest.approx(1.0, abs=1e-6)


def test_split_operator_2d_time_dependent_potential_actually_varies_with_time():
    x = np.linspace(-8, 8, 32)
    y = np.linspace(-8, 8, 32)

    def V(X, Y, t):
        return t * (X**2 + Y**2)

    solver = SplitOperatorSolver2D(x, y, V, dt=1e-3)
    phase_t0 = solver._half_potential_phase(0.0)
    phase_t1 = solver._half_potential_phase(1.0)
    assert not np.allclose(phase_t0, phase_t1)
