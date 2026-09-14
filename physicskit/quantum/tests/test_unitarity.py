"""Verify probability conservation integral |psi(x,t)|^2 dx = 1 +/- 1e-6 over
10^4 timesteps of the split-operator FFT propagator, across a variety of
potential shapes (harmonic, double well, finite well, asymmetric well,
gravitational bouncer, soft infinite box, and a 2D rectangular box).
"""

import numpy as np
import pytest

from physicskit.quantum.core.eigensolvers import (
    asymmetric_step_well,
    double_well,
    finite_well,
    harmonic_well,
    linear_gravitational_well,
)
from physicskit.quantum.core.solvers import SplitOperatorSolver1D, SplitOperatorSolver2D

N_STEPS = 10_000
TOL = 1e-6


def _gaussian(x, x0=0.0, sigma=1.0, k0=0.0):
    norm = (2 * np.pi * sigma**2) ** (-0.25)
    return norm * np.exp(-((x - x0) ** 2) / (4 * sigma**2)) * np.exp(1j * k0 * x)


def _run_1d(V, x, psi0, dt):
    solver = SplitOperatorSolver1D(x, V, dt=dt)
    psi = psi0.astype(complex)
    psi /= np.sqrt(solver.norm(psi))
    for _ in range(N_STEPS):
        psi = solver.step(psi)
    return solver.norm(psi)


@pytest.mark.parametrize(
    "name,V,x_range,x0,sigma,dt",
    [
        ("harmonic", harmonic_well(), (-15, 15), 2.0, 1.0, 1e-3),
        ("double_well", double_well(lam=0.5, a=2.5), (-8, 8), 2.5, 0.5, 5e-4),
        ("finite_well", finite_well(V0=30.0, width=3.0), (-10, 10), 0.0, 0.5, 5e-4),
        ("asymmetric_well", asymmetric_step_well(width=3.0, V_left=40.0, V_right=15.0), (-8, 8), 0.0, 0.5, 5e-4),
        ("gravitational_bouncer", linear_gravitational_well(alpha=2.0), (-12, 12), 3.0, 0.5, 2e-4),
    ],
)
def test_1d_unitarity(name, V, x_range, x0, sigma, dt):
    x = np.linspace(*x_range, 2048)
    psi0 = _gaussian(x, x0=x0, sigma=sigma)
    final_norm = _run_1d(V, x, psi0, dt)
    assert abs(final_norm - 1.0) < TOL, f"{name}: norm drifted to {final_norm}"


def test_1d_unitarity_soft_infinite_box():
    L, wall = 1.0, 4000.0

    def V(x):
        return np.where((x >= 0) & (x <= L), 0.0, wall)

    x = np.linspace(-0.3, 1.3, 2048)
    psi0 = _gaussian(x, x0=0.5, sigma=0.08)
    final_norm = _run_1d(V, x, psi0, dt=2e-5)
    assert abs(final_norm - 1.0) < TOL


def test_2d_unitarity_rectangular_box():
    Lx, Ly, wall = 1.0, 1.0, 3000.0

    def V(x, y):
        return np.where((x >= 0) & (x <= Lx) & (y >= 0) & (y <= Ly), 0.0, wall)

    x = np.linspace(-0.3, 1.3, 160)
    y = np.linspace(-0.3, 1.3, 160)
    solver = SplitOperatorSolver2D(x, y, V, dt=5e-5)

    X, Y = solver.X, solver.Y
    psi0 = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (4 * 0.05**2)).astype(complex)
    psi0 /= np.sqrt(solver.norm(psi0))

    psi = psi0
    for _ in range(2000):
        psi = solver.step(psi)

    assert abs(solver.norm(psi) - 1.0) < TOL
