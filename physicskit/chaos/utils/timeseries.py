"""Analysis of raw time series with unknown governing equations.

Every other Lyapunov/dimension tool in physicskit.chaos works on a
:class:`~physicskit.chaos.core.base_system.DynamicalSystem` (or a point cloud
generated from one). This module instead works on a bare, scalar (or
already-embedded) time series with no known equations of motion -- the
situation for real experimental or observational data -- via delay-coordinate
(Takens) embedding, the Rosenstein algorithm for the largest Lyapunov
exponent, and IAAFT surrogate-data significance testing.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree
from tqdm.auto import tqdm


def delay_embed(series: ArrayLike, dim: int, tau: int = 1) -> NDArray[np.float64]:
    """Reconstruct a delay-coordinate (Takens) embedding of a scalar time series.

    Parameters
    ----------
    series : array_like of float, shape (n,)
        Scalar time series.
    dim : int
        Embedding dimension.
    tau : int, default 1
        Delay, in samples, between successive coordinates.

    Returns
    -------
    ndarray of float, shape (n - (dim - 1) * tau, dim)
        Row ``i`` is ``(series[i], series[i + tau], ..., series[i + (dim-1)*tau])``.

    Notes
    -----
    Choosing `dim` and `tau` well matters a great deal and is its own
    sub-field; this function does not attempt to choose them for you. Common
    heuristics are the first minimum of the time-delayed mutual information
    for `tau`, and the false-nearest-neighbors method for `dim`.
    """
    series = np.asarray(series, dtype=np.float64)
    n = series.shape[0] - (dim - 1) * tau
    if n <= 0:
        raise ValueError("series is too short for the requested dim and tau")
    return np.column_stack([series[i * tau : i * tau + n] for i in range(dim)])


def _nearest_neighbors_excluding_theiler_window(embedded: NDArray[np.float64], min_tsep: int) -> NDArray[np.int64]:
    """For each embedded point, find its nearest neighbor at least `min_tsep`
    samples away in time, via a KD-tree (``O(n log n)``) rather than a full
    ``O(n^2)`` pairwise distance matrix -- the latter is both slow and, for
    anything beyond a few thousand points, memory-prohibitive (e.g. ~2.9 GB
    of ``float64`` distances for a 19,000-point series).
    """
    n = embedded.shape[0]
    tree = cKDTree(embedded)
    k = min(n, max(4, 2 * min_tsep + 4))
    _, candidate_idx = tree.query(embedded, k=k)
    candidate_idx = np.atleast_2d(candidate_idx)

    row_idx = np.arange(n)[:, None]
    valid = np.abs(candidate_idx - row_idx) > min_tsep
    has_valid = valid.any(axis=1)
    first_valid_col = np.argmax(valid, axis=1)
    nearest = candidate_idx[np.arange(n), first_valid_col]

    # Rare fallback for points whose k nearest neighbors are all within the
    # Theiler window: search directly, excluding just that window.
    for i in np.flatnonzero(~has_valid):
        dists_i = np.linalg.norm(embedded - embedded[i], axis=1)
        dists_i[max(0, i - min_tsep) : i + min_tsep + 1] = np.inf
        nearest[i] = np.argmin(dists_i)

    return np.asarray(nearest, dtype=np.int64)


def average_log_divergence(
    series: ArrayLike,
    dim: int,
    tau: int = 1,
    min_tsep: int | None = None,
    max_iter: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rosenstein's average log nearest-neighbor divergence curve.

    Embeds `series` (via :func:`delay_embed`), finds each embedded point's
    nearest neighbor (excluding temporally close points, within `min_tsep`,
    to avoid spuriously "finding" a neighbor that is really just the same
    part of the trajectory shifted slightly in time), and tracks how the
    distance to that neighbor grows as both points evolve forward -- the
    time-series analog of :func:`physicskit.chaos.visualizers.divergence.trajectory_divergence`,
    reconstructed from a single observed series with unknown dynamics.

    Parameters
    ----------
    series : array_like of float, shape (n,)
        Scalar time series.
    dim : int
        Embedding dimension.
    tau : int, default 1
        Embedding delay, in samples.
    min_tsep : int, optional
        Minimum temporal separation (in samples) for a candidate nearest
        neighbor; defaults to `tau`.
    max_iter : int, optional
        Number of forward-evolution steps to track; defaults to
        ``n_embedded // 10``.

    Returns
    -------
    steps : ndarray of int, shape (m,)
        Forward-evolution step counts (in samples) with at least one valid
        pair to average over.
    log_divergence : ndarray of float, shape (m,)
        The mean, over all valid pairs, of ``log(||embedded[i+k] -
        embedded[nearest[i]+k]||)`` at each step `k`. Its slope (in the
        linear-looking early region) estimates the largest Lyapunov
        exponent, in units of "per sample"; see :func:`rosenstein_lyapunov`.
    """
    embedded = delay_embed(series, dim, tau)
    n = embedded.shape[0]
    if min_tsep is None:
        min_tsep = tau
    if max_iter is None:
        max_iter = max(1, n // 10)

    nearest = _nearest_neighbors_excluding_theiler_window(embedded, min_tsep)
    idx = np.arange(n)

    steps = []
    log_divergence = []
    for k in range(max_iter):
        valid = (idx + k < n) & (nearest + k < n)
        if not np.any(valid):
            break
        d = np.linalg.norm(embedded[idx[valid] + k] - embedded[nearest[valid] + k], axis=1)
        d = d[d > 0]
        if d.size == 0:
            continue
        steps.append(k)
        log_divergence.append(np.mean(np.log(d)))

    return np.array(steps), np.array(log_divergence)


def rosenstein_lyapunov(
    series: ArrayLike,
    dim: int,
    tau: int = 1,
    dt: float = 1.0,
    min_tsep: int | None = None,
    max_iter: int | None = None,
    fit_fraction: float = 0.5,
) -> float:
    """Estimate the largest Lyapunov exponent of a scalar time series (Rosenstein's algorithm).

    Unlike every other Lyapunov estimator in physicskit.chaos, this one requires no
    known equations of motion -- only a sampled time series -- making it the
    tool to reach for on real experimental or observational data. Internally
    calls :func:`average_log_divergence` and fits its slope over the first
    `fit_fraction` of the curve, mirroring
    :func:`physicskit.chaos.utils.metrics.lyapunov_exponent_from_divergence`.

    Parameters
    ----------
    series : array_like of float, shape (n,)
        Scalar time series.
    dim : int
        Embedding dimension.
    tau : int, default 1
        Embedding delay, in samples.
    dt : float, default 1.0
        Time between consecutive samples of `series`, used to convert the
        per-sample slope into a per-time-unit exponent.
    min_tsep : int, optional
        Minimum temporal separation for a candidate nearest neighbor;
        defaults to `tau`.
    max_iter : int, optional
        Number of forward-evolution steps to track; defaults to
        ``n_embedded // 10``.
    fit_fraction : float, default 0.5
        Fraction of the (start of the) divergence curve used for the linear
        fit, before it saturates at the attractor's scale.

    Returns
    -------
    float
        Estimated largest Lyapunov exponent, in units of 1/`dt`.
    """
    steps, log_divergence = average_log_divergence(series, dim, tau=tau, min_tsep=min_tsep, max_iter=max_iter)
    if steps.size < 2:
        raise ValueError("not enough divergence-curve points to fit a slope")
    n_fit = max(2, int(fit_fraction * steps.size))
    slope, _ = np.polyfit(steps[:n_fit] * dt, log_divergence[:n_fit], 1)
    return float(slope)


def iaaft_surrogate(series: ArrayLike, n_iter: int = 100, seed: int | None = None) -> NDArray[np.float64]:
    """Generate one IAAFT surrogate of a time series.

    The Iterative Amplitude-Adjusted Fourier Transform (Schreiber & Schmitz
    1996) produces a surrogate series with (very close to) the same value
    distribution and the same power spectrum as `series`, but with its phase
    relationships randomized -- i.e. a realization of a linear, Gaussian
    stochastic process with the same second-order statistics. Any nonlinear
    structure (including determinism/chaos) present in `series` but absent
    from its surrogates is evidence the series is not just linearly
    correlated noise; see :func:`surrogate_test`.

    Parameters
    ----------
    series : array_like of float, shape (n,)
        Time series to generate a surrogate of.
    n_iter : int, default 100
        Number of alternating spectrum/distribution adjustment iterations.
    seed : int, optional
        Seed for the initial random shuffle, for reproducibility.

    Returns
    -------
    ndarray of float, shape (n,)
        One IAAFT surrogate realization.
    """
    series = np.asarray(series, dtype=np.float64)
    n = series.shape[0]
    rng = np.random.default_rng(seed)
    sorted_original = np.sort(series)
    target_amplitude = np.abs(np.fft.rfft(series))

    surrogate = rng.permutation(series)
    for _ in range(n_iter):
        phases = np.angle(np.fft.rfft(surrogate))
        surrogate = np.fft.irfft(target_amplitude * np.exp(1j * phases), n=n)
        ranks = np.argsort(np.argsort(surrogate))
        surrogate = sorted_original[ranks]
    return surrogate


def surrogate_test(
    series: ArrayLike,
    statistic_fn: Callable[[NDArray[np.float64]], float],
    n_surrogates: int = 39,
    n_iter: int = 100,
    seed: int | None = None,
    show_progress: bool = False,
) -> tuple[float, NDArray[np.float64], float]:
    """Test whether a time series shows structure beyond a linear stochastic process.

    Computes `statistic_fn` (e.g. :func:`rosenstein_lyapunov`, or
    :func:`physicskit.chaos.utils.dimension.correlation_dimension` wrapped to return
    just its dimension estimate) on `series` and on `n_surrogates` IAAFT
    surrogates (:func:`iaaft_surrogate`), then reports how many standard
    deviations the observed value is from the surrogate distribution. The
    default ``n_surrogates=39`` follows the standard convention (Theiler et
    al. 1992) of giving a one-sided 2.5% significance threshold when
    `statistic_fn` is significantly larger (or smaller) than every surrogate.

    Parameters
    ----------
    series : array_like of float, shape (n,)
        Time series to test.
    statistic_fn : callable
        Function ``statistic_fn(series) -> float`` computing the statistic of
        interest (e.g. a Lyapunov exponent or fractal dimension estimate).
    n_surrogates : int, default 39
        Number of IAAFT surrogates to generate.
    n_iter : int, default 100
        Number of IAAFT iterations per surrogate.
    seed : int, optional
        Seed for surrogate generation, for reproducibility.
    show_progress : bool, default False
        Display a `tqdm` progress bar over surrogates.

    Returns
    -------
    observed : float
        ``statistic_fn(series)``.
    surrogate_values : ndarray of float, shape (n_surrogates,)
        ``statistic_fn`` evaluated on each surrogate.
    significance : float
        ``|observed - mean(surrogate_values)| / std(surrogate_values)``: the
        number of surrogate standard deviations the observed value is from
        the surrogate mean. Values well above ~2-3 are typically taken as
        evidence of nonlinear structure not explained by a linear process
        with the same spectrum and distribution.
    """
    series = np.asarray(series, dtype=np.float64)
    rng = np.random.default_rng(seed)
    observed = statistic_fn(series)
    surrogate_values = np.array(
        [
            statistic_fn(iaaft_surrogate(series, n_iter=n_iter, seed=int(rng.integers(0, 2**32))))
            for _ in tqdm(range(n_surrogates), disable=not show_progress, desc="surrogates")
        ]
    )
    sigma = float(np.std(surrogate_values))
    significance = abs(observed - float(np.mean(surrogate_values))) / sigma if sigma > 0 else np.inf
    return observed, surrogate_values, significance
