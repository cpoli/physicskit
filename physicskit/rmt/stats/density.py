"""Empirical spectral density (ESD) and the Wigner semicircle reference
density/CDF."""

import numpy as np

from ..spectrum import Spectrum


def empirical_density(spectrum: Spectrum, bins: int = 100, rescaled: bool = True, density: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Histogram-based empirical spectral density of a Spectrum.

    Parameters
    ----------
    spectrum : physicskit.rmt.spectrum.Spectrum
    bins : int
    rescaled : bool
        If True (default), histogram the ensemble-normalized eigenvalues
        (``spectrum.rescaled``) rather than the raw ones.
    density : bool
        Passed through to ``numpy.histogram``.

    Returns
    -------
    centers, counts : numpy.ndarray
        Bin centers and (normalized, if density=True) counts.
    """
    data = spectrum.rescaled.ravel() if rescaled else spectrum.flat
    counts, edges = np.histogram(data, bins=bins, density=density)  # type: ignore[call-overload]
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def semicircle_pdf(x: np.ndarray, radius: float = 2.0) -> np.ndarray:
    """Wigner semicircle probability density on [-radius, radius].

    f(x) = 2 / (pi * radius^2) * sqrt(radius^2 - x^2)

    Parameters
    ----------
    x : numpy.ndarray
    radius : float, optional

    Returns
    -------
    numpy.ndarray
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    mask = np.abs(x) <= radius
    out[mask] = 2.0 / (np.pi * radius**2) * np.sqrt(radius**2 - x[mask] ** 2)
    return out


def semicircle_cdf(x: np.ndarray, radius: float = 2.0) -> np.ndarray:
    """Wigner semicircle cumulative distribution function on
    [-radius, radius].

    F(x) = 1/2 + x*sqrt(radius^2 - x^2) / (pi * radius^2) + arcsin(x/radius) / pi

    Parameters
    ----------
    x : numpy.ndarray
    radius : float, optional

    Returns
    -------
    numpy.ndarray
    """
    x = np.clip(np.asarray(x, dtype=float), -radius, radius)
    return 0.5 + x * np.sqrt(radius**2 - x**2) / (np.pi * radius**2) + np.arcsin(x / radius) / np.pi
