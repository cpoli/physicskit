"""Dynamical diagnostics for Monte Carlo time series: autocorrelation and critical slowing down.

Near a continuous phase transition, successive Monte Carlo configurations
become strongly correlated in (simulation) time -- "critical slowing down"
-- so that naively spaced samples are not statistically independent. The
integrated autocorrelation time quantifies exactly how many sweeps separate
effectively independent samples, and comparing it between update algorithms
(e.g. single-spin-flip Metropolis versus the Wolff cluster algorithm) is the
standard way to demonstrate why cluster algorithms were invented.
"""

from __future__ import annotations

import numpy as np

__all__ = ["autocorrelation_function", "integrated_autocorrelation_time"]


def autocorrelation_function(series, max_lag=None):
    """Normalized autocorrelation function of a scalar Monte Carlo time series.

    .. math::

        \\rho(t) = \\frac{\\langle (x_i - \\bar x)(x_{i+t} - \\bar x) \\rangle}{\\mathrm{Var}(x)}

    computed efficiently via the FFT (Wiener-Khinchin theorem).

    Parameters
    ----------
    series : array_like
        A scalar observable sampled at successive Monte Carlo sweeps (e.g.
        magnetization or energy).
    max_lag : int, optional
        Largest lag to return. Defaults to ``len(series) // 2``.

    Returns
    -------
    ndarray
        :math:`\\rho(t)` for :math:`t = 0, 1, \\ldots,` ``max_lag``, with
        :math:`\\rho(0) = 1`.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if max_lag is None:
        max_lag = n // 2

    padded = np.zeros(2 * n)
    padded[:n] = x
    fft = np.fft.fft(padded)
    autocorr = np.fft.ifft(fft * np.conj(fft)).real[:n]
    autocorr /= np.arange(n, 0, -1)
    return autocorr[: max_lag + 1] / autocorr[0]


def integrated_autocorrelation_time(series, c=5.0, max_lag=None):
    """Estimate the integrated autocorrelation time via Sokal's automatic windowing.

    .. math::

        \\tau_{\\text{int}}(M) = 1 + 2 \\sum_{t=1}^{M} \\rho(t)

    The summation window :math:`M` is chosen automatically as the smallest
    value satisfying :math:`M \\ge c \\, \\tau_{\\text{int}}(M)` (Sokal
    1997), balancing statistical noise against truncation bias. With this
    normalization (:math:`\\tau_{\\text{int}} = 1` for uncorrelated data),
    :math:`\\tau_{\\text{int}}` itself is (approximately) the number of Monte
    Carlo sweeps per effectively independent sample, since
    :math:`\\mathrm{Var}(\\bar x) \\approx \\tau_{\\text{int}}\\,
    \\mathrm{Var}(x)/N` -- the standard quantitative measure of critical
    slowing down.

    Parameters
    ----------
    series : array_like
        A scalar Monte Carlo observable time series.
    c : float, default=5.0
        Windowing constant (Sokal recommends 4-10; smaller is noisier,
        larger is more biased).
    max_lag : int, optional
        Largest lag considered; defaults to ``len(series) // 2``.

    Returns
    -------
    float
        Estimated integrated autocorrelation time, in units of sweeps.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=5000)  # independent samples: tau ~ 1
    >>> tau = integrated_autocorrelation_time(x)
    >>> tau < 3.0
    True
    """
    rho = autocorrelation_function(series, max_lag=max_lag)
    cumulative = 0.0
    for M in range(1, len(rho)):
        cumulative += rho[M]
        tau = 1.0 + 2.0 * cumulative
        if c * tau <= M:
            return float(tau)
    return float(1.0 + 2.0 * cumulative)
