"""Von Neumann entanglement entropy and Page's (1993) exact average --
the "Page curve" -- for the bipartite pure-state ensemble realized by
``physicskit.rmt.ensembles.density_matrix.InducedMeasureEnsemble``.

Reference: D. N. Page, "Average entropy of a subsystem", Phys. Rev.
Lett. 71 (1993) 1291. Verified numerically here (against Monte Carlo
samples from ``InducedMeasureEnsemble`` at several (n, k) pairs) before
being trusted -- see ``tests/test_density_matrix.py``.
"""

import numpy as np
from scipy.special import digamma


def von_neumann_entropy(eigenvalues: np.ndarray) -> float:
    """Von Neumann entropy -sum(lambda_i * log(lambda_i)) (nats), given a
    density matrix's eigenvalues. Zero eigenvalues (exact or numerically
    negligible) contribute 0, per the standard 0*log(0)=0 convention --
    not NaN from log(0).
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvalues = eigenvalues[eigenvalues > 1e-14]
    return float(-np.sum(eigenvalues * np.log(eigenvalues)))


def page_curve_average_entropy(n: int, k: int) -> float:
    """Page's (1993) exact average von Neumann entropy (nats) of the
    n-dimensional reduced density matrix of a Haar-random pure state on
    C^n (x) C^k, n <= k:

        <S> = digamma(n*k + 1) - digamma(k + 1) - (n - 1) / (2*k)

    This is an exact finite-(n, k) result, not an asymptotic
    approximation.
    """
    return digamma(n * k + 1) - digamma(k + 1) - (n - 1) / (2.0 * k)
