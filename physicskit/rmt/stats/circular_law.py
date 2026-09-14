"""The circular law: limiting eigenvalue density of the Ginibre
ensembles (uniform on the unit disk after the sqrt(n) rescaling).

Reference: J. Ginibre, J. Math. Phys. 6 (1965) 440.

Note on naming: this is a completely different concept from the
"circular ensembles" (COE/CUE/CSE, Dyson 1962) in
``physicskit.rmt.ensembles.circular`` -- the shared word "circular" refers to
two unrelated things in the RMT literature (a disk-shaped eigenvalue
density here, vs. eigenvalues literally on the unit circle there). Do
not confuse the modules.

For a uniform distribution on the unit disk, the radial marginal
(density of eigenvalue *magnitude* ``|lambda|``, integrating out the
angle) has PDF f(r) = 2r and CDF F(r) = r^2 on [0, 1] -- an ordinary 1-D
distribution, which is what makes it possible to validate via the
existing KS/Wasserstein ``Benchmark`` machinery (unlike the raw 2-D
disk density itself).
"""

import numpy as np


def circular_law_radial_pdf(r: np.ndarray) -> np.ndarray:
    """Radial marginal density of the circular law: f(r) = 2r on [0, 1].

    Parameters
    ----------
    r : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(r)
    mask = (r >= 0) & (r <= 1)
    out[mask] = 2.0 * r[mask]
    return out


def circular_law_radial_cdf(r: np.ndarray) -> np.ndarray:
    """Radial marginal CDF of the circular law: F(r) = r^2 on [0, 1].

    Parameters
    ----------
    r : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """
    r = np.asarray(r, dtype=float)
    return np.clip(r, 0.0, 1.0) ** 2
