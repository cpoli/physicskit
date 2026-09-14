"""Fractal dimension estimators: box-counting (capacity) and correlation dimension.

Both estimators quantify how the "size" of a point cloud (typically a sampled
strange attractor) scales as the measuring resolution ``epsilon`` shrinks,
and both reduce to the ordinary embedding dimension for a non-fractal set
(e.g. a filled disk in 2D gives dimension ~2, a line segment gives dimension
~1) but return a *non-integer* value for a genuine fractal attractor.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import pdist


def box_counting_dimension(
    points: ArrayLike,
    n_scales: int = 20,
    eps_min: float | None = None,
    eps_max: float | None = None,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Estimate the box-counting (capacity) dimension ``D0`` of a point cloud.

    Covers the bounding box of `points` with a grid of boxes of side
    `epsilon`, counts the number of occupied boxes ``N(epsilon)`` over a
    range of `epsilon`, and estimates ``D0`` as the slope of
    ``log(N(epsilon))`` vs. ``log(1/epsilon)`` (via ordinary least squares).

    Parameters
    ----------
    points : array_like of float, shape (n, dim)
        Point cloud to measure (e.g. a sampled attractor trajectory).
    n_scales : int, default 20
        Number of `epsilon` values to evaluate, log-spaced between `eps_min`
        and `eps_max`.
    eps_min : float, optional
        Smallest box size; defaults to ``1/200`` of the point cloud's extent.
    eps_max : float, optional
        Largest box size; defaults to ``1/2`` of the point cloud's extent.

    Returns
    -------
    D0 : float
        Estimated box-counting dimension.
    epsilons : ndarray of float, shape (n_scales,)
        The box sizes evaluated.
    counts : ndarray of float, shape (n_scales,)
        The number of occupied boxes ``N(epsilon)`` at each box size.

    Notes
    -----
    Once `epsilon` is small enough that most occupied boxes contain only a
    single point, ``N(epsilon)`` stops scaling like the true fractal measure
    and instead approaches the point count itself, biasing ``D0`` low. If the
    estimate looks suspiciously low (e.g. well under 2 for a point cloud that
    should densely fill a 2D region), either supply a larger `eps_min` or use
    more sample points so the default (finer) `eps_min` stays in the regime
    where boxes still contain multiple points on average.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    mins = points.min(axis=0)
    extent = float(np.max(points.max(axis=0) - mins))
    if eps_max is None:
        eps_max = extent / 2.0
    if eps_min is None:
        eps_min = extent / 200.0

    epsilons = np.geomspace(eps_max, eps_min, n_scales)
    counts = np.empty(n_scales)
    for i, eps in enumerate(epsilons):
        box_idx = np.floor((points - mins) / eps).astype(np.int64)
        counts[i] = np.unique(box_idx, axis=0).shape[0]

    slope, _ = np.polyfit(np.log(1.0 / epsilons), np.log(counts), 1)
    return float(slope), epsilons, counts


def correlation_dimension(
    points: ArrayLike,
    n_scales: int = 20,
    eps_min: float | None = None,
    eps_max: float | None = None,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Estimate the correlation dimension ``D2`` via the Grassberger-Procaccia algorithm.

    Computes the correlation sum ``C(epsilon)`` -- the fraction of all point
    pairs closer together than `epsilon` -- over a range of `epsilon`, and
    estimates ``D2`` as the slope of ``log(C(epsilon))`` vs. ``log(epsilon)``
    (via ordinary least squares). ``D2`` is generally a slightly tighter
    (never larger) estimate of an attractor's fractal dimension than the
    box-counting ``D0``, and converges faster with the number of sample
    points since it uses pairwise distances directly rather than a fixed grid.

    Parameters
    ----------
    points : array_like of float, shape (n, dim)
        Point cloud to measure (e.g. a sampled attractor trajectory). Keep
        `points` to at most a few thousand rows: this computes all
        :math:`O(n^2)` pairwise distances.
    n_scales : int, default 20
        Number of `epsilon` values to evaluate, log-spaced between `eps_min`
        and `eps_max`.
    eps_min : float, optional
        Smallest neighborhood radius; defaults to the 5th percentile of the
        pairwise distances. The very smallest pairwise distances are
        dominated by finite-sample discreteness rather than the attractor's
        true fractal scaling, so a percentile well above the minimum gives a
        more reliable default than the minimum itself.
    eps_max : float, optional
        Largest neighborhood radius; defaults to the 50th percentile (median)
        of the pairwise distances. The largest distances are dominated by
        ``C(epsilon)`` saturating toward 1 (once `epsilon` exceeds the
        attractor's overall diameter), which likewise biases a naive default.

    Returns
    -------
    D2 : float
        Estimated correlation dimension.
    epsilons : ndarray of float, shape (n_scales,)
        The neighborhood radii evaluated.
    correlation_sums : ndarray of float, shape (n_scales,)
        The correlation sum ``C(epsilon)`` at each radius.
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    pair_dists = pdist(points)

    if eps_max is None:
        eps_max = float(np.percentile(pair_dists, 50))
    if eps_min is None:
        eps_min = float(np.percentile(pair_dists[pair_dists > 0], 5))

    epsilons = np.geomspace(eps_max, eps_min, n_scales)
    n_pairs = pair_dists.shape[0]
    correlation_sums = np.array([np.count_nonzero(pair_dists < eps) / n_pairs for eps in epsilons])

    mask = correlation_sums > 0
    slope, _ = np.polyfit(np.log(epsilons[mask]), np.log(correlation_sums[mask]), 1)
    return float(slope), epsilons, correlation_sums
