"""Gaussian (Hermite) ensembles -- GOE, GUE, GSE -- plus the continuum
Dyson-index generalization, all built on the Dumitriu-Edelman tridiagonal
model (see ``physicskit.rmt.utils.tridiagonal``).

References
----------
E. Wigner, Ann. Math. 62 (1955) 548; Ann. Math. 67 (1958) 325.
M. L. Mehta, "Random Matrices" (3rd ed.), Academic Press, 2004.
I. Dumitriu, A. Edelman, J. Math. Phys. 43, 5830 (2002).
"""

from __future__ import annotations

import numpy as np

from ..utils.tridiagonal import sample_hermite_beta_eigenvalues
from .base import MatrixEnsemble


def _dense_hermite(n: int, beta: int, rng: np.random.Generator) -> np.ndarray:
    """Dense n x n GOE/GUE matrix (beta=1 real symmetric, beta=2 complex
    Hermitian), same off-diagonal-variance-1/diagonal-variance-2
    convention as the tridiagonal model above.

    Used only for ``_sample_eigenvalues_and_vectors``: unlike the
    tridiagonal model, whose eigenVALUES are exact in distribution but
    whose eigenVECTORS are an artifact of that particular sparse
    representation (verified numerically during development -- their
    inverse participation ratio was 2-3x too large relative to the exact
    Haar-vector result below, and the gap grew with n), a genuinely
    dense, rotationally/unitarily-invariant construction is needed to
    get Haar-distributed eigenvectors. See ``_dense_hermite_quaternion``
    for the beta=4 (GSE) analogue.
    """
    if beta == 1:
        x = rng.standard_normal((n, n))
        return (x + x.T) / np.sqrt(2.0)
    x = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2.0)
    return (x + x.conj().T) / 2.0


def _dense_hermite_quaternion(n: int, rng: np.random.Generator) -> np.ndarray:
    """Dense n x n quaternionic self-dual Hermitian matrix (GSE),
    embedded as a 2n x 2n complex matrix via the standard
    quaternion-to-2x2-complex-block map (same convention as
    ``physicskit.rmt.ensembles.ginibre.GinSE``).

    Its 2n eigenvalues come in exact Kramers double-degenerate pairs
    (like GSE/CSE/EffGSE/JSE); ``_sample_eigenvalues_and_vectors`` keeps
    one representative eigenvalue AND its (2n-dimensional complex, i.e.
    n-quaternion) eigenvector per pair. Verified numerically during
    development that folding each such eigenvector into n per-site
    quaternion weights ``|v[2i]|^2 + |v[2i+1]|^2`` gives an inverse
    participation ratio matching the exact beta=4 Haar-vector theory
    (``physicskit.rmt.stats.localization.ipr_theory(n, beta=4)``) to four
    significant figures -- see
    ``physicskit.rmt.stats.localization.inverse_participation_ratio_quaternionic``
    and ``tests/test_localization.py``.
    """
    a = rng.standard_normal((n, n))
    b = rng.standard_normal((n, n))
    c = rng.standard_normal((n, n))
    d = rng.standard_normal((n, n))
    m = np.zeros((2 * n, 2 * n), dtype=complex)
    m[0::2, 0::2] = a + 1j * b
    m[0::2, 1::2] = c + 1j * d
    m[1::2, 0::2] = -c + 1j * d
    m[1::2, 1::2] = a - 1j * b
    return (m + m.conj().T) / 2.0


class HermiteBetaEnsemble(MatrixEnsemble):
    """General beta-Hermite (Gaussian) ensemble.

    beta=1, 2, 4 recover GOE, GUE, GSE exactly (same joint eigenvalue
    density, not merely the same limiting law); any beta > 0 is a valid
    continuum generalization, useful for studying the Dyson index as a
    continuous "inverse temperature" parameter.

    Use the named subclasses (:class:`GOE`, :class:`GUE`, :class:`GSE`)
    for the classical cases; use this class directly for other beta.
    """

    def __init__(self, n: int, beta: float, seed: int | np.random.Generator | None = None) -> None:
        """
        Parameters
        ----------
        n : int
            Matrix dimension.
        beta : float
            Dyson index; must be positive.
        seed : int, numpy.random.Generator, or None, optional
        """
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        super().__init__(n, seed=seed)
        self.beta: float = float(beta)

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        return sample_hermite_beta_eigenvalues(self.n, self.beta, rng)

    def _sample_eigenvalues_and_vectors(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if self.beta not in (1, 2, 4):
            raise NotImplementedError("return_eigenvectors=True is only supported for beta in (1, 2, 4) (GOE, GUE, GSE).")
        if self.beta == 4:
            h = _dense_hermite_quaternion(self.n, rng)
            eigenvalues, eigenvectors = np.linalg.eigh(h)
            # Exact Kramers double degeneracy: keep one eigenvalue and its
            # (2n-dim complex) eigenvector per pair -- see
            # _dense_hermite_quaternion's docstring.
            return eigenvalues[0::2], eigenvectors[:, 0::2]
        h = _dense_hermite(self.n, int(self.beta), rng)
        eigenvalues, eigenvectors = np.linalg.eigh(h)
        return eigenvalues, eigenvectors

    def natural_scale(self) -> float:
        # Verified empirically (see design notes): raw eigenvalue spread
        # grows as sqrt(n * beta); dividing by this recovers the standard
        # semicircle normalization with support [-2, 2] for all beta.
        return np.sqrt(self.n * self.beta)


class GOE(HermiteBetaEnsemble):
    """Gaussian Orthogonal Ensemble (beta=1).

    Real symmetric matrices; time-reversal symmetric with T^2 = +1.
    Altland-Zirnbauer class AI.
    """

    def __init__(self, n: int, seed: int | np.random.Generator | None = None) -> None:
        super().__init__(n, beta=1, seed=seed)


class GUE(HermiteBetaEnsemble):
    """Gaussian Unitary Ensemble (beta=2).

    Complex Hermitian matrices; no time-reversal symmetry.
    Altland-Zirnbauer class A.
    """

    def __init__(self, n: int, seed: int | np.random.Generator | None = None) -> None:
        super().__init__(n, beta=2, seed=seed)


class GSE(HermiteBetaEnsemble):
    """Gaussian Symplectic Ensemble (beta=4).

    Quaternionic self-dual Hermitian matrices; time-reversal symmetric
    with T^2 = -1. Altland-Zirnbauer class AII.
    """

    def __init__(self, n: int, seed: int | np.random.Generator | None = None) -> None:
        super().__init__(n, beta=4, seed=seed)
