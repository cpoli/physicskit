"""Spectral diagnostics: power spectrum and autocorrelation.

The cheapest, most classical way to distinguish periodic, quasiperiodic, and
chaotic behavior at a glance: a periodic signal's power spectrum is a forest
of sharp lines, a quasiperiodic one a handful of incommensurate lines (plus
their sums/differences), and a chaotic one a broadband continuum. These
functions work on any scalar time series -- a physicskit.chaos-simulated trajectory
component or independently-supplied data alike.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def power_spectrum(series: ArrayLike, dt: float = 1.0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the power spectrum of a (real-valued) scalar time series.

    Parameters
    ----------
    series : array_like of float, shape (n,)
        Time series (its mean is removed before transforming, so the
        spectrum reflects only its fluctuations).
    dt : float, default 1.0
        Time between consecutive samples.

    Returns
    -------
    frequencies : ndarray of float, shape (n // 2 + 1,)
        Non-negative frequencies, in cycles per unit time (i.e. matching
        1/`dt` units).
    power : ndarray of float, shape (n // 2 + 1,)
        Power at each frequency (``|FFT|^2 / n``).
    """
    series = np.asarray(series, dtype=np.float64)
    series = series - series.mean()
    n = series.shape[0]
    spectrum = np.fft.rfft(series)
    power = (np.abs(spectrum) ** 2) / n
    frequencies = np.fft.rfftfreq(n, d=dt)
    return frequencies, power


def autocorrelation(series: ArrayLike, max_lag: int | None = None) -> NDArray[np.float64]:
    """Compute the (normalized) autocorrelation function of a scalar time series.

    Parameters
    ----------
    series : array_like of float, shape (n,)
        Time series (its mean is removed before correlating).
    max_lag : int, optional
        Largest lag to compute, in samples; defaults to ``n - 1``.

    Returns
    -------
    ndarray of float, shape (max_lag + 1,)
        The autocorrelation at lags ``0, 1, ..., max_lag``, normalized so
        that ``result[0] == 1.0``.
    """
    series = np.asarray(series, dtype=np.float64)
    series = series - series.mean()
    n = series.shape[0]
    if max_lag is None:
        max_lag = n - 1

    full = np.correlate(series, series, mode="full")
    zero_lag = full[full.shape[0] // 2 :]
    return np.asarray(zero_lag[: max_lag + 1] / zero_lag[0], dtype=np.float64)
