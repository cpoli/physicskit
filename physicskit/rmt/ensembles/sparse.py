"""Sparse random matrix ensembles: Erdos-Renyi graph "Hamiltonians" and
sparse Bernoulli Wigner-type matrices.

References
----------
P. Erdos, A. Renyi, "On random graphs I", Publ. Math. Debrecen 6 (1959)
290 -- the G(n, p) random graph model.
G. J. Rodgers, A. J. Bray, "Density of states of a sparse random
matrix", Phys. Rev. B 37 (1988) 3557 -- sparse random matrix spectral
theory; the truly sparse regime (mean degree c = n*p fixed as n -> inf)
has no simple closed-form limiting density (unlike the dense/crossover
regime below), the celebrated hard case in this literature.
Z. Furedi, J. Komlos, "The eigenvalues of random symmetric matrices",
Combinatorica 1 (1981) 233 -- the large-outlier eigenvalue (~ mean
degree) of a dense-ish Erdos-Renyi adjacency matrix, separate from its
semicircle-shaped bulk.
F. L. Metz, I. Neri, T. Rogers, "Spectral theory of sparse non-Hermitian
random matrices", J. Phys. A 52 (2019) 434003 -- review of the broader
sparse-RMT landscape.

Two constructions, deliberately different in character:

- :class:`ErdosRenyiEnsemble`: the RAW 0/1 adjacency matrix, used
  directly as a tight-binding "Hamiltonian". Its mean entry is p != 0,
  so (Furedi-Komlos) it has one large POSITIVE OUTLIER eigenvalue near
  the mean degree n*p, separate from a bulk that only approaches the
  semicircle law (radius 2*sqrt(n*p*(1-p))) once n*p -> infinity (the
  "dense-sparse crossover" regime) -- at fixed, non-growing mean degree
  (the truly sparse regime), neither the outlier-subtracted rescaling
  nor the semicircle law describes the bulk correctly (Rodgers-Bray).
  ``natural_scale`` uses the crossover-regime scale sqrt(n*p*(1-p)) --
  meaningful there, but explicitly NOT claimed to hold at fixed mean
  degree; see ``tests/test_sparse.py`` for what is and isn't checked.

- :class:`BernoulliWignerEnsemble`: a MEAN-ZERO sparse analogue of the
  Gaussian Wigner ensemble -- each off-diagonal entry is
  +-1/sqrt(n*p) with probability p/2 each, 0 otherwise. Because the
  mean is exactly zero (no Furedi-Komlos outlier), this ensemble's bulk
  converges to the EXACT semicircle law as long as n*p -> infinity, even
  arbitrarily slowly -- verified directly against the already-validated
  ``WignerSemicircle`` benchmark at moderate sparsity in the test suite,
  not merely asserted from the universality literature.
"""

from __future__ import annotations

import numpy as np

from .base import MatrixEnsemble


class ErdosRenyiEnsemble(MatrixEnsemble):
    """Adjacency matrix of an Erdos-Renyi G(n, p) random graph, used
    directly as a random tight-binding Hamiltonian (see module
    docstring for the outlier/bulk caveat)."""

    def __init__(self, n: int, p: float, seed: int | np.random.Generator | None = None) -> None:
        if not 0.0 < p <= 1.0:
            raise ValueError(f"p must be in (0, 1], got {p}")
        super().__init__(n, seed=seed)
        self.p = float(p)

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        upper = np.triu(rng.random((self.n, self.n)) < self.p, k=1)
        a = upper.astype(float) + upper.T.astype(float)
        return np.linalg.eigvalsh(a)

    def natural_scale(self) -> float:
        return np.sqrt(self.n * self.p * (1.0 - self.p))


class BernoulliWignerEnsemble(MatrixEnsemble):
    """Sparse, mean-zero Bernoulli-entry Wigner-type ensemble: each
    off-diagonal entry is +-1/sqrt(n*p) with probability p/2 each, 0
    otherwise (diagonal exactly zero). See module docstring: unlike
    :class:`ErdosRenyiEnsemble`, this converges to the exact semicircle
    law whenever n*p -> infinity."""

    def __init__(self, n: int, p: float, seed: int | np.random.Generator | None = None) -> None:
        if not 0.0 < p <= 1.0:
            raise ValueError(f"p must be in (0, 1], got {p}")
        super().__init__(n, seed=seed)
        self.p = float(p)

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        n, p = self.n, self.p
        scale = 1.0 / np.sqrt(n * p)
        signs = rng.choice([-1.0, 1.0], size=(n, n))
        mask = rng.random((n, n)) < p
        upper = np.triu(signs * mask * scale, k=1)
        h = upper + upper.T
        return np.linalg.eigvalsh(h)

    def natural_scale(self) -> float:
        return 1.0
