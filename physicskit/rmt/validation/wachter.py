"""Wachter distribution benchmark for the Jacobi (MANOVA) ensembles.

Reference: K. W. Wachter, Ann. Statist. 8 (1980) 937.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from ..stats.wachter import wachter_pdf, wachter_support
from .base import Benchmark


class Wachter(Benchmark):
    """Validates the empirical spectral density of a Jacobi ensemble
    (JOE/JUE) against the exact Wachter law for given inverse aspect
    ratios a = m1/n, b = m2/n.

    CDF and its inverse are built once via a fine-grid cumulative
    trapezoidal integration and cached as interpolators, for the same
    reason as ``physicskit.rmt.validation.marchenko_pastur.MarchenkoPastur``: a
    KS test evaluates the CDF at every data point, so per-point
    ``quad`` calls (as in ``physicskit.rmt.stats.wachter.wachter_cdf``) would be
    far too slow.
    """

    def __init__(self, a: float, b: float, grid_size: int = 4000) -> None:
        if a < 1 or b < 1:
            raise ValueError(f"a and b are inverse aspect ratios (m/n) and must be >= 1; got a={a}, b={b}")
        self.a = a
        self.b = b
        self._lo, self._hi = wachter_support(a, b)

        grid = np.linspace(self._lo, self._hi, grid_size)
        pdf_vals = wachter_pdf(grid, a, b)
        cdf_vals = cumulative_trapezoid(pdf_vals, grid, initial=0.0)
        cdf_vals /= cdf_vals[-1]
        self._grid = grid
        self._pdf_vals = pdf_vals
        self._cdf_interp = interp1d(grid, cdf_vals, bounds_error=False, fill_value=(0.0, 1.0))
        self._inv_cdf_interp = interp1d(cdf_vals, grid, bounds_error=False, fill_value=(self._lo, self._hi))

    def theoretical_pdf(self, x: np.ndarray) -> np.ndarray:
        return wachter_pdf(x, self.a, self.b)

    def theoretical_cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.where(x <= self._lo, 0.0, np.where(x >= self._hi, 1.0, self._cdf_interp(x)))

    def reference_samples(self, size: int, rng: np.random.Generator) -> np.ndarray:
        u = rng.uniform(0.0, 1.0, size=size)
        return self._inv_cdf_interp(u)
