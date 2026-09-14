"""Moments of the CUE characteristic polynomial (Keating-Snaith).

Reference: J. P. Keating, N. C. Snaith, "Random matrix theory and
zeta(1/2+it)", Commun. Math. Phys. 214 (2000) 57 -- moments of the CUE
characteristic polynomial as a model for moments of the Riemann zeta
function on the critical line (following C. P. Hughes, and the earlier
exact CUE moment formula of P. Diaconis, M. Shahshahani, "On the
eigenvalues of random matrices", J. Appl. Probab. 31A (1994) 49).

For U an n x n Haar-random unitary matrix (CUE) and its characteristic
polynomial evaluated on the unit circle, Z_n(theta) = det(I - U*e^{-i*theta}),
the exact moments (for positive integer k, any theta -- the Haar measure
is rotation-invariant, so the value does not depend on theta) are

    ``E[|Z_n(theta)|^(2k)] = prod_{j=0}^{n-1} j! * (j+2k)! / (j+k)!^2``

verified numerically here against direct Monte Carlo (CUE via
``physicskit.rmt.utils.haar.haar_unitary``) at several (n, k) pairs (matching to
within 1-2%, consistent with Monte Carlo noise) before being trusted --
see ``tests/test_characteristic_polynomial.py``. A useful special case
noticed and checked during that verification: k=1 gives exactly n+1 (the
product telescopes), a simple closed form worth knowing independently of
the general product formula.
"""

from math import factorial

import numpy as np


def keating_snaith_moment(n: int, k: int) -> float:
    """Exact CUE characteristic polynomial moment
    ``E[|Z_n(theta)|^(2k)] = prod_{j=0}^{n-1} j!*(j+2k)! / (j+k)!^2``.

    Parameters
    ----------
    n : int
        Matrix dimension.
    k : int
        Moment order (positive integer).

    Returns
    -------
    float
    """
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")
    product = 1.0
    for j in range(n):
        product *= factorial(j) * factorial(j + 2 * k) / factorial(j + k) ** 2
    return product


def characteristic_polynomial_empirical_moment(eigenvalues: np.ndarray, k: int, theta: float = 0.0) -> float:
    """Empirical mean of ``|det(I - U*e^{-i*theta})|^(2k)``, computed
    directly from a CUE spectrum's eigenvalues (the phases of U):
    det(I - U*e^{-i*theta}) = prod_j (1 - e^{i*(phi_j - theta)}) for
    eigenvalue phases phi_j.

    Parameters
    ----------
    eigenvalues : numpy.ndarray, shape (n_samples, n)
        CUE eigenvalue phases (e.g. ``CUE.sample(...).eigenvalues``, or
        ``.rescaled`` -- either works, since only the phase matters).
    k : int
        Moment order (positive integer).
    theta : float, optional

    Returns
    -------
    float
    """
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")
    eigenvalues = np.asarray(eigenvalues)
    z = np.prod(1.0 - np.exp(1j * (eigenvalues - theta)), axis=-1)
    return float(np.mean(np.abs(z) ** (2 * k)))
