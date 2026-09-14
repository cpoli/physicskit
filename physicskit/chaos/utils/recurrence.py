"""Poincare recurrence analysis: return-time statistics and Kac's lemma.

The Poincare recurrence theorem guarantees that, for a measure-preserving
dynamical system, almost every trajectory returns arbitrarily close to its
starting point infinitely often. Kac's lemma sharpens this: for an ergodic
system, the *mean* time between successive returns to a region ``A`` equals
the reciprocal of that region's invariant measure, ``mean_recurrence_time ~=
1 / mu(A)``. The functions here compute both sides of that relationship
numerically from a sampled trajectory.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def recurrence_matrix(states: ArrayLike, epsilon: float) -> NDArray[np.bool_]:
    """The pairwise epsilon-neighborhood recurrence matrix of a trajectory.

    Parameters
    ----------
    states : array_like of float, shape (n, dim)
        Trajectory samples.
    epsilon : float
        Neighborhood radius.

    Returns
    -------
    ndarray of bool, shape (n, n)
        ``R[i, j]`` is ``True`` iff ``states[i]`` and ``states[j]`` are within
        `epsilon` of each other (Euclidean distance). The classic "recurrence
        plot" of nonlinear time series analysis is this matrix rendered as an
        image; see :func:`physicskit.chaos.visualizers.recurrence.plot_recurrence_matrix`.

    Notes
    -----
    This computes all :math:`O(n^2)` pairwise distances; keep `states` to at
    most a few thousand rows.
    """
    states = np.asarray(states, dtype=np.float64)
    diffs = states[:, np.newaxis, :] - states[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    return dists < epsilon


def recurrence_times(states: ArrayLike, epsilon: float, reference_idx: int = 0, dt: float = 1.0) -> NDArray[np.float64]:
    """Gap times between successive returns of a trajectory to a reference point.

    Parameters
    ----------
    states : array_like of float, shape (n, dim)
        Trajectory samples.
    epsilon : float
        Neighborhood radius defining a "return".
    reference_idx : int, default 0
        Index into `states` of the reference point to measure returns to.
    dt : float, default 1.0
        Time between consecutive rows of `states`, used to convert the
        (integer) step counts between returns into physical time units.

    Returns
    -------
    ndarray of float, shape (m,)
        The `m` gap times between successive entries into the
        epsilon-neighborhood of ``states[reference_idx]`` (``m`` is the
        number of *returns* observed minus one; empty if there were fewer
        than two returns).
    """
    states = np.asarray(states, dtype=np.float64)
    reference = states[reference_idx]
    dists = np.sqrt(np.sum((states - reference) ** 2, axis=-1))
    inside = dists < epsilon
    # "Entries" are steps where the trajectory just crossed from outside the
    # ball to inside it -- each one is a single recurrence event, even if the
    # trajectory then lingers nearby for several consecutive samples.
    entries = np.flatnonzero(inside[1:] & ~inside[:-1]) + 1
    return np.diff(entries).astype(np.float64) * dt


def kac_lemma_estimate(states: ArrayLike, epsilon: float, reference_idx: int = 0, dt: float = 1.0) -> tuple[float, float, float]:
    """Numerically validate Kac's lemma on a sampled trajectory.

    Parameters
    ----------
    states : array_like of float, shape (n, dim)
        Trajectory samples, assumed to come from an ergodic,
        measure-preserving (or already attractor-restricted, invariant-measure
        sampling) system.
    epsilon : float
        Neighborhood radius defining "returns" and the region whose measure
        is estimated.
    reference_idx : int, default 0
        Index into `states` of the reference point.
    dt : float, default 1.0
        Time between consecutive rows of `states`.

    Returns
    -------
    mean_recurrence_time : float
        The mean of :func:`recurrence_times` for this trajectory.
    measure_estimate : float
        The empirical invariant measure of the epsilon-neighborhood: the
        fraction of all sampled points in `states` lying within `epsilon` of
        ``states[reference_idx]``.
    kac_product : float
        ``mean_recurrence_time * measure_estimate / dt``, which Kac's lemma
        predicts should be close to 1 for a long-enough, ergodic trajectory
        (the ``/dt`` converts the measure, a dimensionless fraction, onto the
        same per-step footing as a recurrence time measured in physical
        time units).
    """
    states = np.asarray(states, dtype=np.float64)
    times = recurrence_times(states, epsilon, reference_idx=reference_idx, dt=dt)
    mean_time = float(np.mean(times)) if times.size else float("nan")

    reference = states[reference_idx]
    dists = np.sqrt(np.sum((states - reference) ** 2, axis=-1))
    measure = float(np.mean(dists < epsilon))

    kac_product = mean_time * measure / dt
    return mean_time, measure, kac_product
