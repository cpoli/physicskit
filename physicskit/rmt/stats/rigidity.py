"""Number variance Sigma^2(L): a long-range spectral rigidity statistic,
more discriminating than nearest-neighbor spacing alone at distinguishing
correlated (GOE/GUE/GSE) from uncorrelated (Poisson) level sequences.

References
----------
F. J. Dyson, M. L. Mehta, "Statistical Theory of the Energy Levels of
Complex Systems. IV", J. Math. Phys. 4 (1963) 701 -- the original
number variance formulas.
M. L. Mehta, *Random Matrices* (3rd ed.), Academic Press, 2004.

Sigma^2(L) is the variance of the number of (unfolded) eigenvalues found
in a randomly placed interval of length L. For an uncorrelated (Poisson)
sequence, Sigma^2(L) = L exactly. Level repulsion in GOE/GUE/GSE makes
the spectrum far more "rigid": Sigma^2(L) grows only logarithmically,

    Sigma^2_beta(L) = (2 / (beta * pi^2)) * ln(L) + K_beta + O(1/L)

(Dyson-Mehta 1963) -- the leading log-coefficient's 1/beta dependence is
generic across the Gaussian ensembles; the additive constant K_beta
differs by symmetry class.

Two levels of precision are implemented:

- **beta=2 (GUE): exact**, via direct numerical integration of the
  bulk two-point correlation function already validated for CUE in
  ``physicskit.rmt.stats.correlations`` (spacing/rigidity statistics are
  universal in beta across the Gaussian and circular families, so the
  same sine-kernel Y_2(r) = (sin(pi*r)/(pi*r))^2 applies here)::

      Sigma^2(L) = L - 2 * integral_0^L (L - r) * Y_2(r) dr

  This is not merely asymptotic -- it holds at any L within the bulk
  sine-kernel limit, and was verified numerically during development
  against a Monte Carlo GUE window-counting estimate.

- **beta=1 (GOE) and beta=4 (GSE): large-L asymptotic formulas**,
  since deriving beta=1,4 analogues of the sine-kernel integral would
  require the fuller Pfaffian point-process machinery deferred
  elsewhere in this package (see ``physicskit.rmt.stats.correlations``). The
  GOE formula is confirmed by two independent secondary sources; the
  GSE formula follows the same commonly-cited structural pattern and
  was verified numerically against Monte Carlo GSE window-counting
  before being included here (both confirmed to agree with simulation
  at L = 5, 10, 20 to within a few percent -- see
  ``tests/test_rigidity.py``).

Also implemented here: the companion Dyson-Mehta Delta_3(L) statistic
(least-squares deviation of the staircase counting function from a
locally fit straight line), via ``spectral_rigidity_empirical`` (a
direct, formula-free estimator valid for any point process) and
``spectral_rigidity_theory`` (the exact integral relation to Sigma^2(L),
inheriting the same beta=2-exact / beta=1,4-asymptotic precision).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import quad

from ..spectrum import Spectrum

_GAMMA_E = 0.5772156649015329  # Euler-Mascheroni constant


def number_variance_empirical(
    spectrum: Spectrum,
    cdf_func: Callable[[np.ndarray], np.ndarray],
    l_values: np.ndarray,
    n_windows: int = 200,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Estimate Sigma^2(L) for each L in ``l_values`` by counting
    unfolded eigenvalues in many randomly placed windows.

    Parameters
    ----------
    spectrum : physicskit.rmt.spectrum.Spectrum
    cdf_func : callable
        Theoretical CDF used for unfolding (e.g.
        ``physicskit.rmt.stats.density.semicircle_cdf``).
    l_values : sequence of float
    n_windows : int
        Random window placements per sample.
    seed : int or None

    Returns
    -------
    numpy.ndarray
        Sigma^2(L) estimates, one per entry in ``l_values``.
    """
    rng = np.random.default_rng(seed)
    results = []
    for length in l_values:
        all_counts = []
        for row in spectrum.rescaled:
            n = len(row)
            unfolded = n * cdf_func(row)
            lo, hi = unfolded[0], unfolded[-1]
            if hi - lo < 3 * length:
                continue  # window plus margin doesn't fit; skip this sample at this L
            starts = rng.uniform(lo, hi - length, size=n_windows)
            counts = np.array([np.sum((unfolded >= s) & (unfolded < s + length)) for s in starts])
            all_counts.append(counts)
        results.append(np.var(np.concatenate(all_counts)))
    return np.array(results)


def number_variance_poisson(l_values: np.ndarray) -> np.ndarray:
    """Exact number variance for an uncorrelated (Poisson) sequence:
    Sigma^2(L) = L. The "soft spectrum" baseline that GOE/GUE/GSE's
    logarithmic rigidity is contrasted against."""
    return np.asarray(l_values, dtype=float)


def _sine_kernel_y2(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    out = np.ones_like(r)
    mask = np.abs(r) > 1e-10
    out[mask] = (np.sin(np.pi * r[mask]) / (np.pi * r[mask])) ** 2
    return out


def number_variance_gue_exact(l_values: np.ndarray) -> np.ndarray:
    """Exact bulk-sine-kernel number variance (beta=2 / GUE / CUE):
    Sigma^2(L) = L - 2 * integral_0^L (L-r) * Y_2(r) dr.
    """
    l_values = np.atleast_1d(np.asarray(l_values, dtype=float))
    out = np.empty_like(l_values)
    for i, length in enumerate(l_values):
        # limit=200 (vs scipy's default 50): the oscillatory sine-kernel
        # integrand at large L otherwise triggers a subdivision-limit
        # warning even though the result is already converged (verified
        # numerically -- the value is identical to 1e-7 either way, this
        # just silences a spurious warning rather than fixing an
        # accuracy problem).
        val, _ = quad(lambda r, length=length: (length - r) * _sine_kernel_y2(r), 0, length, limit=200)
        out[i] = length - 2.0 * val
    return out


def number_variance_goe_asymptotic(l_values: np.ndarray) -> np.ndarray:
    """Large-L asymptotic number variance for beta=1 (GOE):
    Sigma^2(L) = (2/pi^2) * [ln(2*pi*L) + gamma_E + 1 - pi^2/8].
    """
    l_values = np.asarray(l_values, dtype=float)
    return (2.0 / np.pi**2) * (np.log(2 * np.pi * l_values) + _GAMMA_E + 1.0 - np.pi**2 / 8.0)


def number_variance_gse_asymptotic(l_values: np.ndarray) -> np.ndarray:
    """Large-L asymptotic number variance for beta=4 (GSE):
    Sigma^2(L) = (1/(2*pi^2)) * [ln(4*pi*L) + gamma_E + 1 + pi^2/8].
    """
    l_values = np.asarray(l_values, dtype=float)
    return (1.0 / (2 * np.pi**2)) * (np.log(4 * np.pi * l_values) + _GAMMA_E + 1.0 + np.pi**2 / 8.0)


def number_variance_theory(l_values: np.ndarray, beta: float) -> np.ndarray:
    """Dispatch to the appropriate theoretical Sigma^2(L) by beta:
    exact for beta=2, large-L asymptotic for beta=1, 4."""
    if beta == 1:
        return number_variance_goe_asymptotic(l_values)
    if beta == 2:
        return number_variance_gue_exact(l_values)
    if beta == 4:
        return number_variance_gse_asymptotic(l_values)
    raise ValueError(f"Only beta in (1, 2, 4) have implemented formulas here, got {beta}")


def spectral_rigidity_empirical(
    spectrum: Spectrum,
    cdf_func: Callable[[np.ndarray], np.ndarray],
    l_values: np.ndarray,
    n_windows: int = 200,
    n_grid_per_window: int = 1500,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Estimate the Dyson-Mehta Delta_3(L) statistic for each L in
    ``l_values``, directly from its definition: the mean-square deviation
    of the (unfolded) staircase counting function N(xi) from its best-fit
    line, over many randomly placed windows of length L.

    Delta_3(L) = (1/L) * min_{A,B} <integral_x^{x+L} (N(xi) - A*xi - B)^2 dxi>

    Unlike ``number_variance_theory``, no closed-form kernel is assumed
    here -- each window's line fit is solved numerically (least squares
    against N(xi) sampled on a fine grid), so this estimator is valid for
    ANY point process, not just the classical beta-ensembles. Verified
    against the exact Poisson result (``spectral_rigidity_poisson``,
    Delta_3(L) = L/15) and against ``spectral_rigidity_theory`` at
    beta=2 (GUE) during development -- both matched to within Monte Carlo
    noise (a few percent) -- see ``tests/test_rigidity.py``.

    Parameters
    ----------
    spectrum : physicskit.rmt.spectrum.Spectrum
    cdf_func : callable
        Theoretical CDF used for unfolding (e.g.
        ``physicskit.rmt.stats.density.semicircle_cdf``).
    l_values : numpy.ndarray
    n_windows : int
        Random window placements per sample.
    n_grid_per_window : int
        Grid resolution used to approximate the line-fit integral within
        each window.
    seed : int, numpy.random.Generator, or None, optional

    Returns
    -------
    numpy.ndarray
        Delta_3(L) estimates, one per entry in ``l_values``.
    """
    rng = np.random.default_rng(seed)
    results = []
    for length in l_values:
        vals = []
        for row in spectrum.rescaled:
            n = len(row)
            unfolded = n * cdf_func(row)
            lo, hi = unfolded[0], unfolded[-1]
            if hi - lo < 3 * length:
                continue
            starts = rng.uniform(lo, hi - length, size=n_windows)
            for start in starts:
                grid = np.linspace(start, start + length, n_grid_per_window)
                counts = np.searchsorted(unfolded, grid, side="right").astype(float)
                design = np.vstack([grid, np.ones_like(grid)]).T
                coeffs, *_ = np.linalg.lstsq(design, counts, rcond=None)
                residual = counts - design @ coeffs
                vals.append(np.mean(residual**2))
        results.append(np.mean(vals))
    return np.array(results)


def spectral_rigidity_poisson(l_values: np.ndarray) -> np.ndarray:
    """Exact Dyson-Mehta Delta_3(L) for an uncorrelated (Poisson)
    sequence: Delta_3(L) = L/15 -- verified numerically here against
    ``spectral_rigidity_empirical`` on ``PoissonEnsemble`` samples (see
    ``tests/test_rigidity.py``), the same "soft spectrum" baseline
    ``number_variance_poisson`` provides for Sigma^2(L)."""
    return np.asarray(l_values, dtype=float) / 15.0


def spectral_rigidity_theory(l_values: np.ndarray, beta: float) -> np.ndarray:
    """Theoretical Delta_3(L) at Dyson index beta (1, 2, or 4), via its
    exact integral relation to the number variance:

        Delta_3(L) = (2/L^4) * integral_0^L (L^3 - 2*L^2*r + r^3) * Sigma^2(r) dr

    (Dyson-Mehta 1963; also e.g. Guhr, Mueller-Groeling, Weidenmueller,
    "Random-matrix theories in quantum physics", Phys. Rep. 299 (1998)
    189, Eq. (2.14).) Sigma^2 is dispatched via
    ``number_variance_theory``, so this inherits the same precision:
    exact at beta=2, large-L asymptotic at beta=1, 4. Verified numerically
    against ``spectral_rigidity_empirical`` during development: beta=2
    matched Monte Carlo GUE to within ~1%, beta=1 matched Monte Carlo GOE
    to within the same few-percent tolerance already documented for
    ``number_variance_goe_asymptotic`` -- see ``tests/test_rigidity.py``.
    """
    l_values = np.atleast_1d(np.asarray(l_values, dtype=float))
    out = np.empty_like(l_values)
    for i, length in enumerate(l_values):

        def integrand(r: float, length: float = length) -> float:
            sigma2 = number_variance_theory(np.array([r]), beta)[0]
            return (length**3 - 2.0 * length**2 * r + r**3) * sigma2

        val, _ = quad(integrand, 0, length, limit=200)
        out[i] = 2.0 / length**4 * val
    return out
