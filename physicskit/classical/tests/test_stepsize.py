"""Tests for physicskit.classical.utils.stepsize.estimate_dt.

The critical property this checks is not just "it finds a dt," but
that the returned dt genuinely meets the tolerance at the *requested*
step count, including for a chaotic system where a naive short probe
can find a dt that looks fine briefly and then fails badly later.
"""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.classical.systems.lagrangian import DoublePendulum
from physicskit.classical.systems.newtonian import KeplerSystem
from physicskit.classical.utils.conservation import relative_energy_drift
from physicskit.classical.utils.stepsize import estimate_dt


class _FakeSystemNeverConvergesAtProbe:
    """Deterministically fails the fast probe stage at every dt, however
    small -- used to exercise estimate_dt's shrink-loop-exhausted error
    path without relying on a real system happening to never converge."""

    def energy(self):
        return 1.0

    def integrate(self, t_span, dt, method):
        del t_span, dt, method
        result = type("Result", (), {})()
        result.energy = np.array([1.0, 11.0])
        return result


class _FakeSystemNeverConvergesAtTarget:
    """Passes the fast probe at any dt, but fails verification at the
    full target_steps at every dt -- used to exercise estimate_dt's
    verification-stage error path (the case the module docstring
    describes: a dt that looks fine briefly but fails over a longer
    run) without depending on a real chaotic system's exact behavior."""

    def __init__(self, probe_steps):
        self.probe_steps = probe_steps

    def energy(self):
        return 1.0

    def integrate(self, t_span, dt, method):
        del method
        n_steps = round((t_span[1] - t_span[0]) / dt)
        result = type("Result", (), {})()
        result.energy = np.array([1.0, 1.0]) if n_steps <= self.probe_steps else np.array([1.0, 11.0])
        return result


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


def test_estimate_dt_bracket_search_grows_dt_across_several_doublings():
    """A deliberately tiny dt_initial forces the fast-probe bracket
    search to succeed across several successive doublings before it
    finally fails, rather than failing on the very first growth step."""
    est = estimate_dt(
        lambda: KeplerSystem.from_orbital_elements(a=1.0, e=0.5),
        method="yoshida4",
        tol=1e-6,
        target_steps=2000,
        probe_steps=1000,
        dt_initial=1e-4,
    )
    assert est.achieved_drift < 1e-6
    assert est.n_evaluations > 5  # several bracket/verification evaluations, not just one


def test_estimate_dt_bracket_search_shrinks_dt_across_several_halvings():
    """A deliberately oversized dt_initial forces the fast-probe bracket
    search to fail across several successive halvings before it
    finally succeeds, rather than succeeding on the very first shrink."""
    est = estimate_dt(
        lambda: KeplerSystem.from_orbital_elements(a=1.0, e=0.5),
        method="yoshida4",
        tol=1e-6,
        target_steps=2000,
        probe_steps=1000,
        dt_initial=0.1024,
    )
    assert est.achieved_drift < 1e-6


def test_estimate_dt_raises_when_probe_never_converges_within_max_iter():
    with pytest.raises(RuntimeError, match="shrink steps"):
        estimate_dt(
            lambda: _FakeSystemNeverConvergesAtProbe(),
            method="unused",
            tol=1e-6,
            target_steps=100,
            probe_steps=10,
            dt_initial=1.0,
            max_iter=3,
        )


def test_estimate_dt_raises_when_target_verification_never_converges():
    with pytest.raises(RuntimeError, match="verification shrink steps"):
        estimate_dt(
            lambda: _FakeSystemNeverConvergesAtTarget(probe_steps=10),
            method="unused",
            tol=1e-6,
            target_steps=100,
            probe_steps=10,
            dt_initial=1.0,
            max_iter=3,
        )


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
