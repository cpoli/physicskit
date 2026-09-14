"""The single ring theorem's radii formula (see
``physicskit.rmt.ensembles.single_ring`` for the full construction and
references): a bi-unitarily-invariant non-Hermitian ensemble's
eigenvalues fill the annulus ``r_in <= |z| <= r_out``, with

    r_out = sqrt(<s^2>)          r_in = 1 / sqrt(<1/s^2>)

for singular values s.
"""

import numpy as np


def single_ring_radii(singular_values: np.ndarray) -> tuple[float, float]:
    """Theoretical single-ring radii (r_in, r_out) from a set of
    singular values (or samples from the limiting singular-value
    distribution).

    Parameters
    ----------
    singular_values : numpy.ndarray

    Returns
    -------
    r_in : float
    r_out : float
    """
    singular_values = np.asarray(singular_values, dtype=float)
    r_out = np.sqrt(np.mean(singular_values**2))
    r_in = 1.0 / np.sqrt(np.mean(1.0 / singular_values**2))
    return float(r_in), float(r_out)


def single_ring_radii_wishart_theory(gamma: float) -> tuple[float, float]:
    """Exact single-ring radii for ``NonHermitianWishartEnsemble`` at
    aspect ratio gamma = n/m < 1: r_out = 1, r_in = sqrt(1 - gamma), from
    the Marchenko-Pastur distribution's exact moments E[X] = 1 and
    E[1/X] = 1/(1-gamma) -- verified numerically during development (see
    ``tests/test_single_ring.py``) before being used here.

    Parameters
    ----------
    gamma : float
        Aspect ratio n/m, must be in (0, 1).

    Returns
    -------
    r_in : float
    r_out : float
    """
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must be in (0, 1), got {gamma}")
    return float(np.sqrt(1.0 - gamma)), 1.0


def annulus_radial_cdf(r: np.ndarray, r_in: float, r_out: float) -> np.ndarray:
    """Radial marginal CDF of the uniform distribution on the annulus
    ``r_in <= |z| <= r_out``: F(r) = (r^2 - r_in^2) / (r_out^2 - r_in^2),
    the direct generalization of
    ``physicskit.rmt.stats.circular_law.circular_law_radial_cdf`` (its r_in=0,
    r_out=1 special case).

    Parameters
    ----------
    r : numpy.ndarray
    r_in : float
    r_out : float

    Returns
    -------
    numpy.ndarray
    """
    r = np.clip(np.asarray(r, dtype=float), r_in, r_out)
    return (r**2 - r_in**2) / (r_out**2 - r_in**2)


def annulus_radial_pdf(r: np.ndarray, r_in: float, r_out: float) -> np.ndarray:
    """Radial marginal density of the uniform distribution on the
    annulus ``r_in <= |z| <= r_out``: f(r) = 2r / (r_out^2 - r_in^2).

    Parameters
    ----------
    r : numpy.ndarray
    r_in : float
    r_out : float

    Returns
    -------
    numpy.ndarray
    """
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(r)
    mask = (r >= r_in) & (r <= r_out)
    out[mask] = 2.0 * r[mask] / (r_out**2 - r_in**2)
    return out
