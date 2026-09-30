"""Benchmarks for three representative solvers, one per numerical method.

Each checks a closed-form property of its output, so a benchmark that got
faster by becoming wrong fails.
"""

import numpy as np
import pytest

from physicskit.fluids.systems.navier_stokes import NavierStokes2D
from physicskit.quantum.core.eigensolvers import NumerovSolver
from physicskit.statphys.chapters.ising_lattice import Ising2D


def test_ising_metropolis(benchmark):
    """20 Metropolis sweeps of a 64x64 Ising lattice, deep in the ordered phase."""
    model = Ising2D(L=64, seed=0)

    def run():
        model.reset(ordered=True)
        model.sweep(beta=1.0, n_sweeps=20)
        return model

    run()
    model = benchmark(run)
    # At T = 1 << T_c = 2.269 an ordered start stays ordered: |m| close to Onsager's m(T=1) = 0.99927.
    assert abs(model.magnetization()) / model.n_sites > 0.99


def test_navier_stokes_spectral(benchmark):
    """50 steps of the decaying Taylor-Green vortex on a 64x64 periodic grid."""
    nu, dt, steps = 0.05, 0.01, 50
    solver = NavierStokes2D(n=64, length=2 * np.pi, nu=nu)
    omega0 = 2 * np.sin(solver.X) * np.sin(solver.Y)

    def run():
        return solver.simulate(omega0, dt=dt, steps=steps)

    run()
    result = benchmark(run)
    # Taylor-Green vorticity decays exactly as exp(-2 nu t) (Taylor & Green 1937).
    omega_end = result["omega"]
    assert np.max(np.abs(omega_end)) == pytest.approx(2 * np.exp(-2 * nu * dt * steps), rel=1e-3)


def test_numerov_eigensolver(benchmark):
    """Lowest 10 levels of the harmonic oscillator on a 2001-point grid."""
    solver = NumerovSolver(np.linspace(-10.0, 10.0, 2001), lambda x: 0.5 * x**2)

    solver.solve(n_states=10)
    result = benchmark(solver.solve, n_states=10)
    # E_n = n + 1/2 for hbar = m = omega = 1
    np.testing.assert_allclose(result.energies, np.arange(10) + 0.5, rtol=1e-4)
