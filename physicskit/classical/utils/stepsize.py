"""Automated timestep selection.

None of physicskit.classical's integrators are adaptive -- Verlet, Yoshida4, and
implicit midpoint all take a single fixed ``dt`` for the whole run.
Choosing that ``dt`` by hand (integrate a trial run, check the energy
drift, halve or double, repeat) is exactly the kind of "black art" a
numerical library should not leave to its users. :func:`estimate_dt`
automates that search.

For a quasi-periodic system (e.g. an unperturbed Kepler orbit), the
characteristic energy-drift amplitude at a given ``dt`` is essentially
independent of how many periods you run for, so a short probe
integration is enough. **That is not true for chaotic systems**: a
short probe can find a ``dt`` that looks fine over a few hundred steps
but fails badly once the trajectory (e.g. a double pendulum) later
swings through a more demanding configuration than the probe window
happened to sample. To guard against exactly that,
:func:`estimate_dt` always re-verifies its candidate at the full
intended step count (``target_steps``) and shrinks ``dt`` further if
that longer run doesn't also meet the tolerance -- it does not just
trust the fast probe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = ["StepSizeEstimate", "estimate_dt"]


@dataclass
class StepSizeEstimate:
    """Result of :func:`estimate_dt`.

    Attributes
    ----------
    dt : float
        The recommended, verified step size.
    achieved_drift : float
        Relative energy drift measured at ``dt`` over ``verified_at_steps``.
    verified_at_steps : int
        Step count the verification stage actually confirmed.
    n_evaluations : int
        Total number of trial integrations run during the search.
    """

    dt: float
    achieved_drift: float
    verified_at_steps: int
    n_evaluations: int


def _drift_for(system_factory, method, dt, n_steps) -> float:
    system = system_factory()
    e0 = system.energy()
    result = system.integrate((0.0, n_steps * dt), dt=dt, method=method)
    denom = abs(e0) if e0 != 0.0 else 1.0
    return float(np.max(np.abs(result.energy - e0)) / denom)


def estimate_dt(
    system_factory: Callable[[], object],
    method: str,
    tol: float = 1e-6,
    target_steps: int = 100_000,
    probe_steps: int = 2000,
    dt_initial: float = 1e-2,
    growth: float = 2.0,
    rel_precision: float = 0.05,
    max_iter: int = 40,
) -> StepSizeEstimate:
    """Find the largest ``dt`` for which ``target_steps`` of ``method``
    keeps the relative energy drift below ``tol``.

    A fast bracket-and-bisect search over a short ``probe_steps``
    window finds an initial candidate; that candidate is then always
    re-checked at the full ``target_steps`` and shrunk further (by
    ``growth``) if the longer run doesn't also clear ``tol`` -- see the
    module docstring for why this verification stage matters for
    chaotic systems.

    Parameters
    ----------
    system_factory : callable
        Zero-argument callable returning a *fresh* system instance in
        its initial state (a new instance each call, since
        ``integrate`` mutates state) -- e.g.
        ``lambda: KeplerSystem.from_orbital_elements(a=1.0, e=0.5)``.
    method : str
        Integration method to pass to ``system.integrate`` (e.g.
        ``"yoshida4"`` or ``"implicit_midpoint"``).
    tol : float
        Target relative energy-drift tolerance,
        ``max|H(t) - H(0)| / |H(0)|``.
    target_steps : int
        The step count you actually intend to integrate for; the
        returned ``dt`` is verified to meet ``tol`` over this many
        steps, not just over the fast probe.
    probe_steps : int
        Length of each *fast* bracketing/bisection trial, in steps.
        Should cover at least a few characteristic oscillation periods.
    dt_initial : float
        Starting guess for the fast probe stage.
    growth : float
        Factor used both to bracket a pass/fail boundary during the
        fast probe stage, and to shrink ``dt`` during verification.
    rel_precision : float
        Stop bisecting once the probe-stage bracket
        ``[dt_good, dt_bad]`` has ``(dt_bad - dt_good) / dt_good``
        below this.
    max_iter : int
        Safety cap on iterations, applied separately to the
        bracketing, bisection, and verification stages.

    Returns
    -------
    StepSizeEstimate
        ``.dt`` is the recommended, *verified* step size;
        ``.achieved_drift`` is the relative drift measured at that
        ``dt`` over ``target_steps`` (i.e. what verification actually
        confirmed, not the fast-probe estimate).
    """
    n_evals = 0

    def probe_drift(dt: float) -> float:
        nonlocal n_evals
        n_evals += 1
        return _drift_for(system_factory, method, dt, probe_steps)

    def target_drift(dt: float) -> float:
        nonlocal n_evals
        n_evals += 1
        return _drift_for(system_factory, method, dt, target_steps)

    # Stage 1: fast bracket + bisect over the short probe window.
    dt = dt_initial
    drift = probe_drift(dt)

    if drift < tol:
        dt_good, dt_bad = dt, None
        for _ in range(max_iter):
            dt *= growth
            drift = probe_drift(dt)
            if drift >= tol:
                dt_bad = dt
                break
            dt_good = dt
        else:
            dt_bad = dt  # never failed within max_iter growth steps; use the last value as the bracket edge
    else:
        dt_bad = dt
        dt_good = None
        for _ in range(max_iter):
            dt /= growth
            drift = probe_drift(dt)
            if drift < tol:
                dt_good = dt
                break
            dt_bad = dt
        else:
            raise RuntimeError(f"Could not find a dt meeting tol={tol:.1e} within {max_iter} shrink steps")

    lo, hi = dt_good, dt_bad
    for _ in range(max_iter):
        if (hi - lo) / lo < rel_precision:
            break
        mid = np.sqrt(lo * hi)
        if probe_drift(mid) < tol:
            lo = mid
        else:
            hi = mid

    # Stage 2: verify (and if necessary shrink further) at the full target_steps.
    candidate = lo
    for _ in range(max_iter):
        drift = target_drift(candidate)
        if drift < tol:
            return StepSizeEstimate(dt=candidate, achieved_drift=drift, verified_at_steps=target_steps, n_evaluations=n_evals)
        candidate /= growth
    raise RuntimeError(
        f"Could not find a dt meeting tol={tol:.1e} over target_steps={target_steps} within {max_iter} "
        "verification shrink steps -- this system's worst-case error may need a much smaller dt than "
        "usual (common for chaotic systems); try a smaller dt_initial."
    )
