"""Benchmarks for the shared integrators in physicskit.integrators.

Each integrates ten periods of a circular Kepler orbit (G = M = 1, r = 1,
period 2 pi), then checks the final radius, so a benchmark that got
faster by becoming wrong fails.
"""

import numpy as np
import pytest

from physicskit.integrators import dopri5_integrate, leapfrog_integrate, rk4_integrate, yoshida4_integrate

T_END = 10 * 2 * np.pi
DT = 1e-3
N_STEPS = int(round(T_END / DT))


def _run(benchmark, fn):
    fn()  # compile outside the timed region
    return benchmark(fn)


def test_rk4(benchmark, kepler):
    state0 = np.concatenate([kepler["pos0"], kepler["vel0"]])
    t, y = _run(benchmark, lambda: rk4_integrate(kepler["rhs"], state0, 0.0, DT, N_STEPS, kepler["params"]))
    assert np.hypot(*y[-1, :2]) == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("integrator", [leapfrog_integrate, yoshida4_integrate], ids=["leapfrog", "yoshida4"])
def test_symplectic(benchmark, kepler, integrator):
    t, pos, vel = _run(benchmark, lambda: integrator(kepler["accel"], kepler["pos0"], kepler["vel0"], 0.0, DT, N_STEPS, kepler["params"]))
    assert np.hypot(*pos[-1]) == pytest.approx(1.0, rel=1e-5)


def test_dopri5(benchmark, kepler):
    state0 = np.concatenate([kepler["pos0"], kepler["vel0"]])
    t, y = _run(benchmark, lambda: dopri5_integrate(kepler["rhs"], state0, 0.0, T_END, 1e-2, kepler["params"], rtol=1e-9, atol=1e-12))
    assert t[-1] == pytest.approx(T_END)
    assert np.hypot(*y[-1, :2]) == pytest.approx(1.0, rel=1e-6)
