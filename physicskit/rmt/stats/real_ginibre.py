"""The real-Ginibre real-eigenvalue-count anomaly: a real Ginibre matrix
(GinOE) has a mix of purely real eigenvalues and complex-conjugate
pairs, with the expected number of real eigenvalues growing like
sqrt(n) -- a celebrated result with no Hermitian-ensemble analogue (see
``physicskit.rmt.ensembles.ginibre`` module docstring).

Reference: A. Edelman, E. Kostlan, S. Shub, "How many eigenvalues of a
random matrix are real?", J. Amer. Math. Soc. 7 (1994) 247.

Precision note: Edelman-Kostlan-Shub's finite-n result is an exact
closed form built from the real-Ginibre skew-orthogonal Hermite-
polynomial kernel -- machinery not otherwise needed anywhere else in
this package, and which this implementation does not reproduce. What IS
implemented and verified here is the large-n asymptotic expansion

    E[N_real](n) ~ sqrt(2*n/pi) + 1/2 + O(1/sqrt(n))

The leading sqrt(2n/pi) term is the well-known leading-order asymptotic
already noted in ``physicskit.rmt.ensembles.ginibre``; the +1/2 correction was
fit against, and confirmed by, direct Monte Carlo real-eigenvalue
counts on ``GinOE`` at n = 2, 4, ..., 128 during development (residual
E[N_real] - sqrt(2n/pi) plateaus near 0.5, consistent with this
correction, rather than assumed from memory) -- see
``tests/test_real_ginibre.py``. This is therefore an asymptotic
formula, not the exact EKS closed form, and is documented as such:
expect several-percent-level disagreement at small n (n well below 30),
narrowing as n grows.
"""

import numpy as np


def real_eigenvalue_count_empirical(eigenvalues: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Count (near-)real eigenvalues per sample.

    Parameters
    ----------
    eigenvalues : numpy.ndarray, shape (n_samples, n), complex
        Raw complex eigenvalues, e.g. ``GinOE`` ``Spectrum.eigenvalues``.
    tol : float, optional
        An eigenvalue counts as real if ``abs(Im(lambda))`` is below
        this (accounts for floating-point residue from
        ``numpy.linalg.eigvals`` on an exactly-real matrix).

    Returns
    -------
    numpy.ndarray, shape (n_samples,)
        Integer count of real eigenvalues in each sample.
    """
    eigenvalues = np.asarray(eigenvalues)
    return np.sum(np.abs(eigenvalues.imag) < tol, axis=-1)


def real_eigenvalue_count_asymptotic(n: int) -> float:
    """Large-n asymptotic expected number of real eigenvalues of an
    n x n real Ginibre (GinOE) matrix: sqrt(2*n/pi) + 1/2.

    See module docstring for precision caveats (asymptotic, not the
    exact Edelman-Kostlan-Shub 1994 finite-n closed form).

    Parameters
    ----------
    n : int

    Returns
    -------
    float
    """
    return np.sqrt(2.0 * n / np.pi) + 0.5


def real_eigenvalue_density_empirical(eigenvalues: np.ndarray, n: int, bins: int = 60, x_max: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    """Empirical density of real eigenvalue positions (rescaled by
    ``sqrt(n)``, matching ``GinOE.natural_scale``): expected number of
    real eigenvalues per unit x per sample, histogrammed over the
    (near-)real eigenvalues pooled across all samples.

    Parameters
    ----------
    eigenvalues : numpy.ndarray, shape (n_samples, n), complex
        Raw (unrescaled) ``GinOE`` eigenvalues.
    n : int
        Matrix dimension (for the sqrt(n) rescaling).
    bins : int, optional
    x_max : float, optional
        Histogram range is ``[-x_max, x_max]``.

    Returns
    -------
    centers, density : numpy.ndarray
    """
    eigenvalues = np.asarray(eigenvalues)
    n_samples = eigenvalues.shape[0]
    real_values = eigenvalues.real[np.abs(eigenvalues.imag) < 1e-8] / np.sqrt(n)
    edges = np.linspace(-x_max, x_max, bins + 1)
    counts, _ = np.histogram(real_values, bins=edges)
    bin_width = edges[1] - edges[0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = counts / (n_samples * bin_width)
    return centers, density


def real_eigenvalue_density_asymptotic(x: np.ndarray, n: int) -> np.ndarray:
    """Large-n asymptotic expected density of real eigenvalue positions
    (rescaled by sqrt(n)): uniform on (-1, 1) at height
    ``real_eigenvalue_count_asymptotic(n) / 2`` (i.e. the total expected
    real-eigenvalue count spread evenly over the rescaled real-axis
    support), zero outside.

    Verified numerically during development (not assumed): pooled,
    rescaled real eigenvalues from direct Monte Carlo GinOE samples show
    a flat bulk histogram (no visible x-dependence within statistical
    noise) with a clean cutoff at |x|=1 matching the circular law's disk
    radius, rather than any elevated edge density or bulk curvature --
    see ``tests/test_real_ginibre.py``. Like
    ``real_eigenvalue_count_asymptotic``, this is a verified large-n
    asymptotic description, not the exact finite-n
    Edelman-Kostlan-Shub density (which has additional fine structure,
    e.g. a smooth rather than sharp edge, not captured here).

    Parameters
    ----------
    x : numpy.ndarray
    n : int

    Returns
    -------
    numpy.ndarray
    """
    x = np.asarray(x, dtype=float)
    height = real_eigenvalue_count_asymptotic(n) / 2.0
    out = np.zeros_like(x)
    out[np.abs(x) <= 1.0] = height
    return out
