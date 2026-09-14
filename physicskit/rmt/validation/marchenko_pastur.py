"""Marchenko-Pastur law benchmark for the Wishart (Laguerre) ensembles.

Reference: V. A. Marchenko, L. A. Pastur, Mat. Sb. 72 (1967) 507.
"""

import numpy as np
from scipy.interpolate import interp1d

from ..stats.marchenko_pastur import mp_pdf, mp_support
from .base import Benchmark


class MarchenkoPastur(Benchmark):
    """Validates the empirical spectral density of a Wishart ensemble
    (LOE/LUE/LSE, or general beta-Laguerre) against the exact
    Marchenko-Pastur law for a given aspect ratio gamma = n/m.

    CDF and its inverse are built once per (gamma, sigma2) via a fine-grid
    cumulative trapezoidal integration and cached as interpolators, for
    the same reason as ``physicskit.rmt.stats.ratios.RatioSurmise``: a KS test
    evaluates the CDF at every data point, so per-point ``quad`` calls
    would be far too slow.
    """

    def __init__(self, gamma: float, sigma2: float = 1.0, grid_size: int = 4000) -> None:
        if not (0 < gamma <= 1):
            raise ValueError("gamma must be in (0, 1] for this ensemble convention (m >= n samples/variables, no point mass at zero)")
        self.gamma = gamma
        self.sigma2 = sigma2
        self._lo, self._hi = mp_support(gamma, sigma2)

        from scipy.integrate import cumulative_trapezoid

        grid = np.linspace(self._lo, self._hi, grid_size)
        pdf_vals = mp_pdf(grid, gamma, sigma2)
        cdf_vals = cumulative_trapezoid(pdf_vals, grid, initial=0.0)
        cdf_vals /= cdf_vals[-1]  # correct for edge-integrable-singularity discretization error
        self._grid = grid
        self._pdf_vals = pdf_vals
        self._cdf_interp = interp1d(grid, cdf_vals, bounds_error=False, fill_value=(0.0, 1.0))
        self._inv_cdf_interp = interp1d(cdf_vals, grid, bounds_error=False, fill_value=(self._lo, self._hi))

    def theoretical_pdf(self, x: np.ndarray) -> np.ndarray:
        return mp_pdf(x, self.gamma, self.sigma2)

    def theoretical_cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.where(x <= self._lo, 0.0, np.where(x >= self._hi, 1.0, self._cdf_interp(x)))
        return out

    def reference_samples(self, size: int, rng: np.random.Generator) -> np.ndarray:
        u = rng.uniform(0.0, 1.0, size=size)
        return self._inv_cdf_interp(u)
