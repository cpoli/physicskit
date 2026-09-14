"""Two-point (pair) correlation function of a point process, and the
exact sine-kernel result for the Circular Unitary Ensemble.

Scope note: the exact bulk two-point correlation function for CUE
(beta=2) has the simple closed determinantal form used here. The
corresponding beta=1 (COE) and beta=4 (CSE) results also reduce to
sine-kernel-type expressions, but through the fuller Pfaffian
point-process machinery: a 2x2 matrix kernel built from companion
functions "S" (the same sine kernel as beta=2), "D" = S', and
"I" = integral of S (Mehta, "Random Matrices", Ch. 6-7; Dyson's
original papers). ``kernel_s``, ``kernel_d``, ``kernel_i`` below
implement these three individually exact, independently verifiable
building blocks (D and I are just the derivative and antiderivative of
the already-validated sine kernel S).

The COMBINATION of S, D, I into R_2 for beta=1, 4 is NOT implemented
here: an attempted reconstruction (from memory) of the standard
"R_2 = 1 - S^2 + D*I"-type combination was checked here two ways before
being ruled out -- (1) a small-r Taylor expansion shows that
combination can only ever produce a QUADRATIC leading term in r
(D*I ~ r^2 near r=0, same order as 1-S^2), which is structurally
incompatible with the well-established LINEAR level-repulsion exponent
required for beta=1; and (2) fitting against direct Monte Carlo COE
pair-correlation data (see ``pair_correlation_estimate``) confirmed no
value of an overall coefficient on the D*I term brings it into
agreement. Rather than ship a formula known to be structurally wrong,
only the qualitative, robustly-verified physics is asserted here (see
``tests/test_correlations.py``): level repulsion strength orders
CSE >> CUE >> COE >> Poisson (uncorrelated) near r=0, and COE's own
empirical correlation is consistent with genuine (sub-quadratic) level
repulsion. The exact beta=1, 4 closed forms remain deferred.

Reference: F. J. Dyson, J. Math. Phys. 3 (1962) 140, 157, 166.
"""

import numpy as np
from scipy.special import sici

from ..spectrum import Spectrum


def pair_correlation_estimate(spectrum: Spectrum, r_max: float = 4.0, n_bins: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the two-point correlation function R_2(r) of an unfolded
    point process (e.g. a circular ensemble's phases) by histogramming
    pairwise circular distances.

    Convention note: pairwise distances are unsigned, so a bin at
    separation r receives contributions from *both* sides of each
    reference point; the standard R_2(r) convention is defined per
    signed separation, so raw counts must be divided by 2 in addition to
    the usual (n_points * bin_width) normalization -- verified during
    development against a Poisson (uncorrelated) reference process,
    where R_2(r) = 1 identically and the un-halved estimator gave
    exactly 2.0 instead.

    Parameters
    ----------
    spectrum : physicskit.rmt.spectrum.Spectrum
        A circular-ensemble spectrum (``spectrum.rescaled`` already
        unfolded to unit mean spacing on a circle of circumference n).
    r_max : float
        Maximum separation to histogram.
    n_bins : int

    Returns
    -------
    centers, r2_estimate : numpy.ndarray
    """
    edges = np.linspace(0.0, r_max, n_bins + 1)
    counts = np.zeros(n_bins)
    total_points = 0
    for row in spectrum.rescaled:
        n = len(row)
        diffs = np.abs(np.subtract.outer(row, row))
        diffs = np.minimum(diffs, n - diffs)  # circular distance
        np.fill_diagonal(diffs, np.inf)
        flat = diffs[diffs <= r_max]
        counts += np.histogram(flat, bins=edges)[0]
        total_points += n
    bin_width = edges[1] - edges[0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    r2_estimate = counts / (2.0 * total_points * bin_width)
    return centers, r2_estimate


def sine_kernel_r2(r: np.ndarray) -> np.ndarray:
    """Exact CUE (beta=2) two-point correlation function:
    R_2(r) = 1 - [sin(pi*r) / (pi*r)]^2, with the removable singularity
    at r=0 handled via its limit (R_2(0) = 0, full level repulsion)."""
    r = np.asarray(r, dtype=float)
    out = np.ones_like(r)
    mask = np.abs(r) > 1e-8
    out[mask] = 1.0 - (np.sin(np.pi * r[mask]) / (np.pi * r[mask])) ** 2
    out[~mask] = 0.0
    return out


def kernel_s(r: np.ndarray) -> np.ndarray:
    """The "S" Pfaffian-point-process kernel: S(r) = sin(pi*r)/(pi*r),
    the same sine kernel as beta=2 (see module docstring's scope note
    on why only this building block, not the full beta=1/4 R_2
    combination, is implemented)."""
    r = np.asarray(r, dtype=float)
    out = np.ones_like(r)
    mask = np.abs(r) > 1e-10
    out[mask] = np.sin(np.pi * r[mask]) / (np.pi * r[mask])
    return out


def kernel_d(r: np.ndarray) -> np.ndarray:
    """The "D" Pfaffian-point-process kernel: D(r) = dS/dr (an odd
    function, since S is even)."""
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(r)
    mask = np.abs(r) > 1e-6
    x = np.pi * r[mask]
    out[mask] = np.pi * (x * np.cos(x) - np.sin(x)) / x**2
    return out


def kernel_i(r: np.ndarray) -> np.ndarray:
    """The "I" Pfaffian-point-process kernel: I(r) = integral_0^r S(t) dt
    = Si(pi*r)/pi, where Si is the sine integral function."""
    si, _ = sici(np.pi * np.asarray(r, dtype=float))
    return si / np.pi


def k_point_correlation(positions: np.ndarray) -> float:
    """Exact CUE (beta=2) k-point correlation function at the given
    positions: R_k(x_1, ..., x_k) = det[S(x_i - x_j)]_{i,j=1}^k, the
    defining property of a determinantal point process (Soshnikov,
    "Determinantal random point fields", Russian Math. Surveys 55
    (2000) 923; also Mehta Ch. 5) with kernel S (``kernel_s``, the same
    sine kernel already validated for k=2 via ``sine_kernel_r2``).

    Reduces EXACTLY to ``sine_kernel_r2`` at k=2 (checked directly:
    ``det[[S(0),S(r)],[S(-r),S(0)]] = 1 - S(r)^2`` since S is even) --
    this is the anchor this generalization rests on, since a fresh
    Monte Carlo estimate of a k=3 (or higher) correlation value from
    ensemble samples turned out to have a subtle reference-point-
    selection bias that made a first attempt at direct verification
    unreliable. What IS verified here (see ``tests/test_correlations.py``):
    the formula vanishes exactly whenever two positions coincide (level
    repulsion at all orders -- an algebraic property of a determinant
    with a repeated row) and correctly factorizes,
    ``R_3(x1, x2, x3) -> R_2(x1, x2)`` as ``x3`` moves to infinity
    (cluster decomposition, checked numerically to 6 significant figures).

    Parameters
    ----------
    positions : numpy.ndarray, shape (k,)
        The k positions (unfolded, i.e. in units of the mean spacing).

    Returns
    -------
    float
    """
    positions = np.asarray(positions, dtype=float)
    k = len(positions)
    diff = positions[:, None] - positions[None, :]
    kernel_matrix = kernel_s(diff.ravel()).reshape(k, k)
    return float(np.linalg.det(kernel_matrix))
