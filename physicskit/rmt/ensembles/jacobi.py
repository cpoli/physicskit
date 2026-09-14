"""Jacobi (MANOVA) ensembles -- JOE (beta=1), JUE (beta=2).

References
----------
K. W. Wachter, "The limiting empirical measure of multiple discriminant
ratios", Ann. Statist. 8 (1980) 937 -- the limiting law (see
``physicskit.rmt.stats.wachter``).
L. Erdos, B. Farrell, "Local Eigenvalue Density for General MANOVA
Matrices" (arXiv:1207.0031) -- the construction and formula used here,
verified numerically during development (see design notes and
``tests/test_jacobi_wachter.py``).

Construction: given two independent Wishart-type matrices A, B (n x n,
built from m1 and m2 "samples" respectively, m1, m2 >= n), the Jacobi
ensemble matrix is

    J = A (A + B)^{-1}

Unlike Gaussian/Wishart, this requires the actual dense matrices A and B
(not just their eigenvalues), since the ratio construction mixes them --
none of the tridiagonal/bidiagonal eigenvalue-only tricks used elsewhere
in this package apply here. J itself is generally NOT symmetric, but is
similar to the manifestly symmetric matrix

    M = (A+B)^{-1/2} A (A+B)^{-1/2}

(via conjugation by (A+B)^{1/2}), which has the same eigenvalues and is
what's actually diagonalized here for numerical stability. Eigenvalues
are real and confined to [0, 1] (a "double hard edge", unlike the
semicircle/Marchenko-Pastur soft edges).

Parametrization: following Erdos-Farrell/Wachter, a = m1/n, b = m2/n
(inverse aspect ratios, i.e. "samples per variable" -- note this is the
reciprocal of the gamma = n/m convention used in
``physicskit.rmt.ensembles.wishart``).

beta=4 (quaternion, "JSE"): A and B are built from n x m quaternion
matrices, embedded as 2n x 2m complex matrices via the standard
quaternion-to-2x2-complex-block map (the same convention as
``physicskit.rmt.ensembles.ginibre.GinSE`` and
``physicskit.rmt.ensembles.effective_hamiltonian``'s beta=4 case). Since that
embedding is an algebra homomorphism, A, B, S = A+B, and
M = S^{-1/2} A S^{-1/2} are all quaternionic self-dual Hermitian, so M's
2n eigenvalues come in exact Kramers double-degenerate pairs -- verified
numerically during development to machine precision (~1e-15) before
being trusted here, the same true degeneracy GSE/CSE/EffGSE have (as
opposed to GinSE's merely-conjugate, genuinely distinct pairs). Only the
n distinct values are returned (one representative per pair), mirroring
those other beta=4 ensembles.
"""

import numpy as np

from .base import MatrixEnsemble


def _dense_wishart(n: int, m: int, rng: np.random.Generator, complex_entries: bool) -> np.ndarray:
    if complex_entries:
        x = (rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))) / np.sqrt(2.0)
    else:
        x = rng.standard_normal((n, m))
    return x @ x.conj().T


def _quaternion_gaussian_block(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """N x M quaternion matrix, embedded as a 2N x 2M complex matrix
    (same convention as ``physicskit.rmt.ensembles.ginibre.GinSE``)."""
    a = rng.standard_normal((n, m))
    b = rng.standard_normal((n, m))
    c = rng.standard_normal((n, m))
    d = rng.standard_normal((n, m))
    w = np.zeros((2 * n, 2 * m), dtype=complex)
    w[0::2, 0::2] = a + 1j * b
    w[0::2, 1::2] = c + 1j * d
    w[1::2, 0::2] = -c + 1j * d
    w[1::2, 1::2] = a - 1j * b
    return w / np.sqrt(2.0)


def _dense_quaternion_wishart(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    x = _quaternion_gaussian_block(n, m, rng)
    return x @ x.conj().T


class JacobiBetaEnsemble(MatrixEnsemble):
    """General (beta=1, 2, or 4) Jacobi/MANOVA ensemble.

    ``n`` is the ensemble/matrix size (per the ``MatrixEnsemble`` base
    class, matching the other ensembles' convention -- for beta=4 this
    is the number of *distinct* eigenvalues, per the module docstring);
    ``m1``, ``m2`` are the two Wishart "sample sizes" (each must be >= n).
    """

    def __init__(
        self,
        n: int,
        m1: int,
        m2: int,
        beta: int,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        """
        Parameters
        ----------
        n : int
            Number of variables (matrix dimension).
        m1 : int
            Sample size of the first Wishart-type matrix; must be >= n.
        m2 : int
            Sample size of the second Wishart-type matrix; must be >= n.
        beta : int
            Dyson index; must be 1, 2, or 4.
        seed : int, numpy.random.Generator, or None, optional
        """
        if beta not in (1, 2, 4):
            raise ValueError(f"JacobiBetaEnsemble only supports beta in (1, 2, 4), got {beta}")
        if m1 < n or m2 < n:
            raise ValueError(f"m1 (samples={m1}) and m2 (samples={m2}) must both be >= n (variables={n})")
        super().__init__(n, seed=seed)
        self.m1 = m1
        self.m2 = m2
        self.beta: float = beta

    @property
    def a(self) -> float:
        """Inverse aspect ratio m1/n (Erdos-Farrell/Wachter parametrization)."""
        return self.m1 / self.n

    @property
    def b(self) -> float:
        """Inverse aspect ratio m2/n."""
        return self.m2 / self.n

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        if self.beta == 4:
            mat_a = _dense_quaternion_wishart(self.n, self.m1, rng)
            mat_b = _dense_quaternion_wishart(self.n, self.m2, rng)
        else:
            complex_entries = self.beta == 2
            mat_a = _dense_wishart(self.n, self.m1, rng, complex_entries)
            mat_b = _dense_wishart(self.n, self.m2, rng, complex_entries)
        s = mat_a + mat_b
        w, v = np.linalg.eigh(s)
        s_inv_sqrt = (v * (1.0 / np.sqrt(w))) @ v.conj().T
        m = s_inv_sqrt @ mat_a @ s_inv_sqrt
        m = (m + m.conj().T) / 2.0  # symmetrize away numerical asymmetry
        eigenvalues = np.linalg.eigvalsh(m)
        if self.beta == 4:
            # Exact Kramers double degeneracy (see module docstring) --
            # keep one representative per pair, as GSE/CSE/EffGSE do.
            eigenvalues = np.sort(eigenvalues)[0::2]
        return eigenvalues

    def natural_scale(self) -> float:
        # Eigenvalues already live in the natural [0, 1] range; no rescaling.
        return 1.0


class JOE(JacobiBetaEnsemble):
    """Jacobi Orthogonal Ensemble (beta=1): real double-Wishart ratio."""

    def __init__(self, n: int, m1: int, m2: int, seed: int | np.random.Generator | None = None) -> None:
        super().__init__(n, m1, m2, beta=1, seed=seed)


class JUE(JacobiBetaEnsemble):
    """Jacobi Unitary Ensemble (beta=2): complex double-Wishart ratio."""

    def __init__(self, n: int, m1: int, m2: int, seed: int | np.random.Generator | None = None) -> None:
        super().__init__(n, m1, m2, beta=2, seed=seed)


class JSE(JacobiBetaEnsemble):
    """Jacobi Symplectic Ensemble (beta=4): quaternionic double-Wishart
    ratio. ``n`` is the quaternionic dimension; each sample produces n
    distinct eigenvalues (each an exact Kramers pair internally -- see
    module docstring)."""

    def __init__(self, n: int, m1: int, m2: int, seed: int | np.random.Generator | None = None) -> None:
        super().__init__(n, m1, m2, beta=4, seed=seed)
