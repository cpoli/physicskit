"""The Marchenko-Pastur law: limiting eigenvalue density of the Wishart
ensembles.

Reference: V. A. Marchenko, L. A. Pastur, "Distribution of eigenvalues
for some sets of random matrices", Mat. Sb. 72 (1967) 507.
"""

import numpy as np
from scipy.integrate import quad


def mp_support(gamma: float, sigma2: float = 1.0) -> tuple[float, float]:
    """Support edges [lambda_minus, lambda_plus] of the Marchenko-Pastur
    density, for aspect ratio gamma = n/m (n variables, m samples,
    m >= n so 0 < gamma <= 1) and entry variance sigma2."""
    lo = sigma2 * (1.0 - np.sqrt(gamma)) ** 2
    hi = sigma2 * (1.0 + np.sqrt(gamma)) ** 2
    return lo, hi


def mp_pdf(x: np.ndarray, gamma: float, sigma2: float = 1.0) -> np.ndarray:
    """Marchenko-Pastur probability density at x, for aspect ratio
    gamma = n/m (0 < gamma <= 1; the m >= n regime relevant to the
    ``LaguerreBetaEnsemble`` convention here, which has no point mass at
    zero)."""
    lo, hi = mp_support(gamma, sigma2)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    mask = (x > lo) & (x < hi)
    xm = x[mask]
    out[mask] = np.sqrt((hi - xm) * (xm - lo)) / (2.0 * np.pi * gamma * sigma2 * xm)
    return out


def mp_cdf(x: np.ndarray, gamma: float, sigma2: float = 1.0) -> np.ndarray:
    """Marchenko-Pastur CDF at x (0 < gamma <= 1 regime, no point mass).

    No elementary closed form for general x, so integrated numerically
    from the lower edge; ``scipy.integrate.quad`` is called per point,
    which is fine for the (typically small) grids this is evaluated on
    directly, but ``validation.marchenko_pastur.MarchenkoPastur`` caches
    a fast interpolated version for use inside a KS test.
    """
    lo, hi = mp_support(gamma, sigma2)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi <= lo:
            out[i] = 0.0
        elif xi >= hi:
            out[i] = 1.0
        else:
            val, _ = quad(mp_pdf, lo, xi, args=(gamma, sigma2))
            out[i] = val
    return out
