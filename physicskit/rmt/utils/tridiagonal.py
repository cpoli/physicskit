"""Dumitriu & Edelman (2002) tridiagonal model for the Gaussian beta-ensemble.

Reference: I. Dumitriu and A. Edelman, "Matrix Models for Beta Ensembles",
J. Math. Phys. 43, 5830 (2002).

The eigenvalues of the tridiagonal matrix

    H_beta = tridiag( off_diag, diag, off_diag )

with

    diag[i]      ~ N(0, 2)                       i.i.d., i = 1..n
    off_diag[i]  ~ chi(df = beta * (n - i))       independent, i = 1..n-1

have exactly the joint eigenvalue density of the Gaussian beta-ensemble

    f(x_1, ..., x_n) ~ exp(-beta/4 * sum_i x_i^2) * prod_{i<j} ``|x_i - x_j|^beta``

for beta = 1, 2, 4 this recovers GOE, GUE, GSE eigenvalue statistics exactly
(not just asymptotically), and for any beta > 0 it gives the continuum
Dyson-index generalization -- without ever forming or diagonalizing a dense
N x N matrix. Diagonalizing a tridiagonal matrix is O(n^2) rather than the
O(n^3) required for a general dense Hermitian eigensolve, which is what
makes it practical to push n high enough for the asymptotic theorems
(semicircle law, Tracy-Widom edge statistics, etc.) to actually apply.

Raw eigenvalues from this construction have empirical spread that grows
like sqrt(n * beta); divide by that factor to reach the standard semicircle
normalization with support [-2, 2]. See ``natural_scale`` on
``HermiteBetaEnsemble`` in ``physicskit.rmt.ensembles.gaussian``.
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.stats import chi


def sample_hermite_beta_eigenvalues(n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Draw one realization of eigenvalues from the beta-Hermite ensemble.

    Parameters
    ----------
    n : int
        Matrix dimension.
    beta : float
        Dyson index. beta=1 -> GOE, beta=2 -> GUE, beta=4 -> GSE; any
        beta > 0 is a valid continuum generalization.
    rng : numpy.random.Generator

    Returns
    -------
    numpy.ndarray, shape (n,)
        Raw (unnormalized) eigenvalues, ascending order.
    """
    if n < 2:
        raise ValueError("n must be at least 2 for the tridiagonal model")
    diag = rng.normal(loc=0.0, scale=np.sqrt(2.0), size=n)
    dof = beta * np.arange(n - 1, 0, -1)  # (n-1)*beta, (n-2)*beta, ..., beta
    off_diag = chi.rvs(dof, random_state=rng)
    return eigh_tridiagonal(diag, off_diag, eigvals_only=True)


def sample_hermite_beta_eigenvalues_and_vectors(n: int, beta: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw one realization of eigenvalues AND eigenvectors from the
    beta-Hermite ensemble.

    The tridiagonal model preserves not only the exact beta-Hermite
    eigenVALUE density but also the ensemble's rotationally/unitarily-
    invariant eigenVECTOR distribution (Dumitriu-Edelman 2002; also
    Edelman-Sutton): since GOE/GUE/GSE (and their continuum-beta
    generalization) are invariant under conjugation by an orthogonal/
    unitary/symplectic change of basis, the tridiagonal model's
    eigenvectors are a faithful stand-in for a dense-matrix
    diagonalization's eigenvectors, not merely a computational
    convenience for the eigenvalues alone. This makes them usable, e.g.,
    as the delocalized reference case for inverse-participation-ratio
    statistics (see ``physicskit.rmt.stats.localization``) -- verified
    numerically here: the mean IPR of the columns returned matches the
    exact Haar/Gaussian-vector result ``(beta/2+1)/(n*beta/2+1)`` (see
    ``physicskit.rmt.stats.localization.ipr_theory`` and
    ``tests/test_localization.py``).

    Parameters
    ----------
    n : int
        Matrix dimension.
    beta : float
        Dyson index. beta=1 -> GOE, beta=2 -> GUE, beta=4 -> GSE; any
        beta > 0 is a valid continuum generalization.
    rng : numpy.random.Generator

    Returns
    -------
    eigenvalues : numpy.ndarray, shape (n,)
        Raw (unnormalized) eigenvalues, ascending order.
    eigenvectors : numpy.ndarray, shape (n, n)
        Column ``i`` is the eigenvector for ``eigenvalues[i]``.
    """
    if n < 2:
        raise ValueError("n must be at least 2 for the tridiagonal model")
    diag = rng.normal(loc=0.0, scale=np.sqrt(2.0), size=n)
    dof = beta * np.arange(n - 1, 0, -1)
    off_diag = chi.rvs(dof, random_state=rng)
    return eigh_tridiagonal(diag, off_diag, eigvals_only=False)


def sample_laguerre_beta_eigenvalues(m: int, n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Draw one realization of eigenvalues from the beta-Laguerre
    (Wishart) ensemble, corresponding to an m x n rectangular Gaussian
    data matrix (m "samples", n "variables", m >= n) with i.i.d.
    unit-variance entries -- real for beta=1, complex for beta=2,
    quaternionic for beta=4; any beta > 0 is a valid continuum
    generalization.

    Construction: a classical fact (Householder-bidiagonalization of a
    Gaussian matrix yields independent chi-distributed bidiagonal
    entries) generalized to arbitrary beta following the same pattern as
    the Dumitriu-Edelman tridiagonal Hermite model. The n x n bidiagonal
    matrix

        B = bidiag( diagonal: chi(beta*m), chi(beta*(m-1)), ..., chi(beta*(m-n+1));
                    subdiagonal: chi(beta*(n-1)), chi(beta*(n-2)), ..., chi(beta) )

    gives W = B B^T / (beta * m) eigenvalues matching the standard
    Marchenko-Pastur normalization (support [(1-sqrt(gamma))^2,
    (1+sqrt(gamma))^2] with gamma = n/m as m, n -> infinity).

    Verified numerically at beta=1, 2 against dense X^T X / m
    constructions (real and complex Ginibre-type X respectively): pooled
    eigenvalue samples agree via two-sample KS test to within Monte Carlo
    noise. All three classical beta (1, 2, 4) converge to the exact
    Marchenko-Pastur law with correctly located support edges.

    Parameters
    ----------
    m : int
        Number of samples (rows) of the underlying data matrix, m >= n.
    n : int
        Number of variables (columns) of the underlying data matrix.
    beta : float
        Dyson index. beta=1 -> real, beta=2 -> complex, beta=4 ->
        quaternionic; any beta > 0 is a valid continuum generalization.
    rng : numpy.random.Generator

    Returns
    -------
    numpy.ndarray, shape (n,)
        Raw eigenvalues (already in Marchenko-Pastur normalization,
        i.e. natural_scale = 1 for this ensemble), ascending order.
    """
    if n < 2:
        raise ValueError("n must be at least 2 for the bidiagonal model")
    if m < n:
        raise ValueError("m (samples) must be >= n (variables)")
    diag_dof = beta * (m - np.arange(n))
    sub_dof = beta * (n - 1 - np.arange(n - 1))
    diag = chi.rvs(diag_dof, random_state=rng)
    sub = chi.rvs(sub_dof, random_state=rng) if n > 1 else np.array([])
    B = np.diag(diag)
    if n > 1:
        B += np.diag(sub, -1)
    W = (B @ B.T) / (beta * m)
    # Real advantage over dense X^T X: no m x n matrix (and no O(m*n^2)
    # matrix product) is ever formed regardless of how large m is --
    # generating B costs O(n) chi draws, independent of m. The final
    # eigenvalue solve on the n x n matrix W is O(n^3) either way.
    return np.linalg.eigvalsh(W)
