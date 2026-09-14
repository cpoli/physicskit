"""Tests for physicskit.classical.utils.stepsize.estimate_dt.

The critical property this checks is not just "it finds a dt," but
that the returned dt genuinely meets the tolerance at the *requested*
step count, including for a chaotic system where a naive short probe
can find a dt that looks fine briefly and then fails badly later.
"""

from __future__ import annotations

import numpy as np

from physicskit.classical.systems.lagrangian import DoublePendulum
from physicskit.classical.systems.newtonian import KeplerSystem
from physicskit.classical.utils.conservation import relative_energy_drift
from physicskit.classical.utils.stepsize import estimate_dt


def test_estimate_dt_quasi_periodic_kepler():
    tol = 1e-6
    est = estimate_dt(
        lambda: KeplerSystem.from_orbital_elements(a=1.0, e=0.5),
        method="yoshida4",
        tol=tol,
        target_steps=20_000,
        probe_steps=1000,
        dt_initial=1e-2,
    )
    assert est.achieved_drift < tol

    system = KeplerSystem.from_orbital_elements(a=1.0, e=0.5)
    result = system.integrate((0.0, 20_000 * est.dt), dt=est.dt, method="yoshida4")
    assert np.max(relative_energy_drift(result.energy)) < tol


def test_estimate_dt_verification_catches_chaotic_underestimate():
    """A short probe alone can find a dt for the (chaotic) double
    pendulum that passes over a few hundred steps but fails once run
    for the full target_steps; estimate_dt must not return such a dt --
    its own verification stage has to catch and shrink it."""
    tol = 1e-6
    target_steps = 20_000
    est = estimate_dt(
        lambda: DoublePendulum([2.0, 1.0], [0.5, -0.3]),
        method="implicit_midpoint",
        tol=tol,
        target_steps=target_steps,
        probe_steps=500,
        dt_initial=1e-3,
    )
    assert est.achieved_drift < tol

    system = DoublePendulum([2.0, 1.0], [0.5, -0.3])
    result = system.integrate((0.0, target_steps * est.dt), dt=est.dt, method="implicit_midpoint")
    assert np.max(relative_energy_drift(result.energy)) < tol
