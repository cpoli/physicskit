"""Circular ensembles -- COE, CUE, CSE.

References
----------
F. J. Dyson, "Statistical Theory of the Energy Levels of Complex Systems
I-III", J. Math. Phys. 3 (1962) 140, 157, 166 -- the original
circular-ensemble classification, and the source of Dyson's beta=1,2,4
threefold way.
F. Mezzadri, Notices Amer. Math. Soc. 54 (2007) 592 -- the Haar-random
sampling recipe used here.

Important construction note (this is easy to get wrong): COE and CSE
are NOT simply "eigenvalues of a Haar-random matrix from O(n) / Sp(n)".
A real orthogonal matrix's eigenvalues are either +-1 or complex-conjugate
pairs -- not the generic unit-circle structure a circular ensemble
requires. Dyson's actual constructions::

    CUE: eigenvalues of U, U Haar-random from U(n).
    COE: eigenvalues of U = V^T V, V Haar-random from U(n) -- a
         *symmetric* unitary matrix, built from a unitary building block.
    CSE: eigenvalues of U = V^R V, V Haar-random from U(2n), where
         V^R = Z V^T Z^{-1} is the symplectic dual (Z the symplectic
         form). U is self-dual unitary; its 2n eigenvalues come in n
         doubly-degenerate pairs, of which n distinct values are the CSE
         eigenvalues.

All three were verified during development by checking their unfolded
level-spacing statistics against the already-validated (generalized)
Wigner surmise at the matching beta -- see ``tests/test_circular.py``.
This is a meaningful cross-family check: spacing statistics are
universal in beta across the Gaussian and circular families, so
agreement here is independent evidence the constructions are correct,
not just that "some points ended up on a circle".

Eigenvalue representation: ``Spectrum.eigenvalues`` for these ensembles
holds the eigenvalue *phases* theta in [0, 2*pi), not the complex
eigenvalues e^{i*theta} themselves -- this keeps the rest of the
(real-array-based) stats machinery directly usable. ``natural_scale`` is
set to 2*pi/n, so ``Spectrum.rescaled`` is already fully unfolded with
exactly unit mean spacing -- exactly, not just asymptotically, since
Haar measure is exactly rotation-invariant at any finite n. This is a
structural simplification relative to the Gaussian/Wishart ensembles,
which only have unit mean spacing after unfolding against their
(asymptotic) limiting density.
"""

import numpy as np

from ..utils.haar import haar_unitary, symplectic_form
from .base import MatrixEnsemble


class COE(MatrixEnsemble):
    """Circular Orthogonal Ensemble (beta=1)."""

    beta: float = 1

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        v = haar_unitary(self.n, rng)
        u = v.T @ v  # symmetric unitary
        eigs = np.linalg.eigvals(u)
        return np.sort(np.angle(eigs) % (2.0 * np.pi))

    def natural_scale(self) -> float:
        return 2.0 * np.pi / self.n


class CUE(MatrixEnsemble):
    """Circular Unitary Ensemble (beta=2)."""

    beta: float = 2

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        u = haar_unitary(self.n, rng)
        eigs = np.linalg.eigvals(u)
        return np.sort(np.angle(eigs) % (2.0 * np.pi))

    def natural_scale(self) -> float:
        return 2.0 * np.pi / self.n


class CSE(MatrixEnsemble):
    """Circular Symplectic Ensemble (beta=4).

    ``n`` here is the number of *distinct* CSE eigenvalues; internally a
    Haar-random matrix of size 2n is used (each eigenvalue of the dual
    construction is doubly degenerate).
    """

    beta: float = 4

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        n2 = 2 * self.n
        v = haar_unitary(n2, rng)
        z = symplectic_form(n2)
        v_dual = (-z) @ v.T @ z  # Z^{-1} = -Z since Z^2 = -I
        u = v_dual @ v
        eigs = np.linalg.eigvals(u)
        theta = np.sort(np.angle(eigs) % (2.0 * np.pi))
        # Degenerate pairs are adjacent after sorting; keep one of each.
        return theta[0::2]

    def natural_scale(self) -> float:
        return 2.0 * np.pi / self.n
