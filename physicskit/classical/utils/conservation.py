"""Conservation-law diagnostics: energy drift, angular momentum, and the
Laplace-Runge-Lenz (LRL) vector.

These are thin, system-agnostic helpers that operate on the arrays
already produced by ``system.integrate(...)`` (a
:class:`physicskit.classical.core.base_system.SimulationResult`), used both by the
test suite (``tests/test_conservation.py``) and by the visualizers to
annotate plots with drift metrics.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "energy_drift",
    "relative_energy_drift",
    "angular_momentum_2d",
    "angular_momentum_drift",
    "lrl_drift",
]


def energy_drift(energy: np.ndarray) -> np.ndarray:
    """Absolute energy deviation from the initial value.

    Parameters
    ----------
    energy : ndarray, shape (n_steps + 1,)
        Energy trajectory, e.g. ``result.energy``.

    Returns
    -------
    ndarray
        ``|H(t) - H(0)|`` at each sample.
    """
    energy = np.asarray(energy, dtype=np.float64)
    return np.abs(energy - energy[0])


def relative_energy_drift(energy: np.ndarray) -> np.ndarray:
    """Relative energy deviation from the initial value.

    This is the quantity conventionally checked against a tolerance
    (e.g. < 1e-6) to certify a symplectic integrator is behaving
    correctly over long integrations.

    Parameters
    ----------
    energy : ndarray, shape (n_steps + 1,)
        Energy trajectory, e.g. ``result.energy``.

    Returns
    -------
    ndarray
        ``|H(t) - H(0)| / |H(0)|`` at each sample; ``inf`` everywhere
        if ``H(0)`` is exactly zero.
    """
    energy = np.asarray(energy, dtype=np.float64)
    e0 = energy[0]
    if e0 == 0.0:
        return np.full_like(energy, np.inf)
    return np.abs(energy - e0) / np.abs(e0)


def angular_momentum_2d(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Out-of-plane angular momentum for a planar trajectory.

    Parameters
    ----------
    q, p : ndarray, shape (n_steps, 2)
        Position and momentum trajectories.

    Returns
    -------
    ndarray, shape (n_steps,)
        ``L = x*py - y*px`` at each sample.
    """
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return q[:, 0] * p[:, 1] - q[:, 1] * p[:, 0]


def angular_momentum_drift(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Absolute drift of the planar angular momentum from its initial value.

    Parameters
    ----------
    q, p : ndarray, shape (n_steps, 2)
        Position and momentum trajectories.

    Returns
    -------
    ndarray, shape (n_steps,)
        ``|L(t) - L(0)|`` at each sample.
    """
    L = angular_momentum_2d(q, p)
    return np.abs(L - L[0])


def lrl_drift(system, q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Drift of the Laplace-Runge-Lenz vector's magnitude across a trajectory.

    For the unperturbed 1/r potential the LRL vector is exactly
    conserved (both magnitude and direction); its direction sweeping
    out an angle over time is the visual signature of apsidal
    precession once a perturbation is present, so magnitude drift
    remaining small while direction changes is the expected, physically
    correct behavior for a perturbed orbit.

    Parameters
    ----------
    system : object
        A :class:`physicskit.classical.systems.newtonian.KeplerSystem`-like object
        exposing ``lrl_vector(q, p) -> array``.
    q, p : ndarray, shape (n_steps, 2)
        Position and momentum trajectories.

    Returns
    -------
    ndarray, shape (n_steps,)
        ``||A(t)| - |A(0)||`` at each sample.
    """
    vecs = np.array([system.lrl_vector(qi, pi) for qi, pi in zip(q, p)])
    mags = np.linalg.norm(vecs, axis=1)
    return np.abs(mags - mags[0])
