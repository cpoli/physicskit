"""Consecutive-spacing ratio statistic (Oganesyan & Huse, Phys. Rev. B 75,
155111, 2007) and its closed-form surmise (Atas, Bogomolny, Giraud & Roux,
Phys. Rev. Lett. 110, 084101, 2013).

The ratio statistic::

    r_n = min(s_n, s_{n-1}) / max(s_n, s_{n-1})   in [0, 1]

where s_n = x_{n+1} - x_n are consecutive *raw* spacings, is invariant
under any smooth local rescaling of the eigenvalues -- unlike the spacing
distribution P(s), it requires no unfolding. That makes it a genuinely
independent validation of the same underlying level-repulsion physics: if
both the (unfolding-dependent) spacing distribution and the
(unfolding-free) ratio distribution agree with their respective surmises,
unfolding artifacts can be ruled out as the source of agreement.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from ..spectrum import Spectrum


def ratio_statistics(spectrum: Spectrum, edge_trim: float = 0.1) -> np.ndarray:
    """Pooled ratio statistics from every sample in a Spectrum, computed
    directly on raw eigenvalues (rescaling is irrelevant to a ratio, but
    ``spectrum.rescaled`` is used for consistency with the rest of the
    package).

    Parameters
    ----------
    spectrum : physicskit.rmt.spectrum.Spectrum
    edge_trim : float
        Fraction of eigenvalues discarded from each edge before taking
        spacings, for the same reason as in ``nearest_neighbor_spacings``.
        No unfolding is involved, so trim order doesn't matter here.

    Returns
    -------
    numpy.ndarray
        Pooled ratio statistics, each in [0, 1].
    """
    all_ratios = []
    for row in spectrum.rescaled:
        n = len(row)
        lo = int(edge_trim * n)
        hi = n - lo
        if hi - lo < 3:
            continue
        s = np.diff(row[lo:hi])
        r = np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])
        all_ratios.append(r)
    return np.concatenate(all_ratios)


def _ratio_surmise_unnormalized_pdf(r: np.ndarray, beta: float) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return (r + r**2) ** beta / (1.0 + r + r**2) ** (1.0 + 1.5 * beta)


class RatioSurmise:
    """The Atas et al. (2013) surmise for the ratio statistic, restricted
    to r in [0, 1] (the min/max-normalized convention used by
    ``ratio_statistics`` above).

    The normalizing constant and CDF have no simple closed form, so both
    are built once per beta via a fine-grid cumulative trapezoidal
    integration and cached as interpolators -- this is far cheaper than
    calling ``scipy.integrate.quad`` per evaluation point, which matters
    since a KS test evaluates the CDF at every data point.
    """

    def __init__(self, beta: float, grid_size: int = 4000) -> None:
        """
        Note on accuracy: this surmise is exact for a reduced (3-level)
        model, not for the true N-level ensemble -- analogous to how the
        classical Wigner surmise is exact for a 2x2 matrix but only an
        excellent approximation at general N. Its own exact mean at
        beta=1 is 4 - 2*sqrt(3) ~= 0.5359 (verified numerically against
        this implementation), which is measurably different from the
        numerically-exact full-GOE mean ratio (~0.5307) reported
        elsewhere in the literature -- both numbers are correct, they
        just answer different questions (surmise vs. true ensemble).
        """
        self.beta = beta
        grid = np.linspace(0.0, 1.0, grid_size)
        pdf_unnorm = _ratio_surmise_unnormalized_pdf(grid, beta)
        cdf_unnorm = cumulative_trapezoid(pdf_unnorm, grid, initial=0.0)
        self._normalization = cdf_unnorm[-1]
        cdf = cdf_unnorm / self._normalization
        self._cdf_interp = interp1d(grid, cdf, bounds_error=False, fill_value=(0.0, 1.0))
        # For sampling: invert the CDF via interpolation the other way.
        # cdf is monotonically increasing, so this is well-defined.
        self._inv_cdf_interp = interp1d(cdf, grid, bounds_error=False, fill_value=(0.0, 1.0))
        self._pdf_grid = grid
        self._pdf_unnorm = pdf_unnorm

    def pdf(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        return np.interp(np.clip(r, 0.0, 1.0), self._pdf_grid, self._pdf_unnorm) / self._normalization

    def cdf(self, r: np.ndarray) -> np.ndarray:
        return self._cdf_interp(r)

    def rvs(self, size: int | tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        u = rng.uniform(0.0, 1.0, size=size)
        return self._inv_cdf_interp(u)
