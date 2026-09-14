"""The Wachter distribution: limiting eigenvalue density of the Jacobi
(MANOVA) ensembles.

Reference: K. W. Wachter, "The limiting empirical measure of multiple
discriminant ratios", Ann. Statist. 8 (1980) 937. Formula and
parametrization as given in L. Erdos, B. Farrell, "Local Eigenvalue
Density for General MANOVA Matrices" (arXiv:1207.0031), verified
numerically during development against real (beta=1) and complex
(beta=2) Jacobi ensemble simulations (KS statistic < 0.002 in both
cases at moderate N -- see ``tests/test_jacobi_wachter.py``).
"""

import numpy as np
from scipy.integrate import quad


def wachter_support(a: float, b: float) -> tuple[float, float]:
    """Support edges [lambda_minus, lambda_plus] of the Wachter
    distribution, for inverse aspect ratios a = m1/n, b = m2/n
    (a, b >= 1)."""
    s = a + b
    term1 = np.sqrt((a / s) * (1.0 - 1.0 / s))
    term2 = np.sqrt((1.0 / s) * (1.0 - a / s))
    lam_minus = (term1 - term2) ** 2
    lam_plus = (term1 + term2) ** 2
    return lam_minus, lam_plus


def wachter_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Wachter probability density at x, for inverse aspect ratios
    a = m1/n, b = m2/n."""
    lam_minus, lam_plus = wachter_support(a, b)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    mask = (x > lam_minus) & (x < lam_plus)
    xm = x[mask]
    out[mask] = (a + b) * np.sqrt((xm - lam_minus) * (lam_plus - xm)) / (2.0 * np.pi * xm * (1.0 - xm))
    return out


def _wachter_pdf_scalar(x: float, a: float, b: float, lam_minus: float, lam_plus: float) -> float:
    # Non-vectorized version for use as a quad integrand (quad calls its
    # integrand with plain Python floats; an array-returning function
    # here caused a real bug during development -- see design notes).
    if x <= lam_minus or x >= lam_plus:
        return 0.0
    return (a + b) * np.sqrt((x - lam_minus) * (lam_plus - x)) / (2.0 * np.pi * x * (1.0 - x))


def wachter_cdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Wachter CDF at x. No elementary closed form, so integrated
    numerically from the lower edge (see
    ``validation.wachter.Wachter`` for a fast interpolated version used
    inside KS tests)."""
    lam_minus, lam_plus = wachter_support(a, b)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.zeros_like(x)
    for i, xi in enumerate(x):
        xi = float(xi)
        if xi <= lam_minus:
            out[i] = 0.0
        elif xi >= lam_plus:
            out[i] = 1.0
        else:
            val, _ = quad(_wachter_pdf_scalar, lam_minus, xi, args=(a, b, lam_minus, lam_plus))
            out[i] = val
    return out
