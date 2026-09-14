"""Spectral unfolding: rescale eigenvalues locally so the mean
nearest-neighbor spacing is 1 everywhere, which is what the spacing
distribution and rigidity statistics are defined relative to.

Where the ensemble's limiting spectral density is known exactly (e.g. the
semicircle law for the Gaussian ensembles), unfolding via that theoretical
CDF is preferable to a polynomial/spline fit to the empirical cumulative
distribution: it introduces no additional fitting noise, and it's exact in
the same sense the benchmark it will be validated against is exact.
"""

from collections.abc import Callable

import numpy as np


def unfold_by_cdf(eigenvalues: np.ndarray, cdf_func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Unfold a single realization's eigenvalues using a known theoretical
    integrated density (CDF).

    Parameters
    ----------
    eigenvalues : numpy.ndarray, shape (n,)
        A single realization's eigenvalues (already in the ensemble's
        natural/rescaled units -- e.g. semicircle support [-2, 2]),
        ascending order.
    cdf_func : callable
        The theoretical CDF matching the ensemble's limiting spectral
        density (e.g. ``physicskit.rmt.stats.density.semicircle_cdf``).

    Returns
    -------
    numpy.ndarray, shape (n,)
        Unfolded eigenvalues with unit mean nearest-neighbor spacing.
    """
    n = len(eigenvalues)
    return n * cdf_func(eigenvalues)
