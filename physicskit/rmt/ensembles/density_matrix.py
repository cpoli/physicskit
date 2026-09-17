"""Random quantum state (density matrix) ensembles: the induced measure
(Hilbert-Schmidt as its special case) and the Bures-Hall measure.

References
----------
K. Zyczkowski, H.-J. Sommers, "Induced measures in the space of mixed
quantum states", J. Phys. A 34 (2001) 7111 -- the induced-measure
construction, its joint eigenvalue density, and the mean-purity formula
used for validation here.
D. N. Page, "Average entropy of a subsystem", Phys. Rev. Lett. 71
(1993) 1291 -- the exact average-entanglement-entropy formula (the
"Page curve"), realized directly by this ensemble at general k -- see
below.
K. Zyczkowski, K. A. Penson, I. Nechita, B. Collins, "Generating random
density matrices", J. Math. Phys. 52 (2011) 062201 -- survey covering
both measures used here.
V. Osipov, H.-J. Sommers, K. Zyczkowski, "Random Bures mixed states and
the distribution of their purity", J. Phys. A 43 (2010) 055302 -- the
Bures-Hall construction and joint eigenvalue density.
J. Hall, "Random quantum correlations and density operator
distributions", Phys. Lett. A 242 (1998) 123 -- the original Bures-Hall
measure.

Induced measure (and its Hilbert-Schmidt special case)
--------------------------------------------------------
A Haar-random pure state on C^n (x) C^k (n <= k WLOG), traced out over
the k-dimensional second factor, gives a reduced density matrix on the
n-dimensional first factor distributed EXACTLY as

    rho = A @ A^dagger / Tr(A @ A^dagger)

for A an n x k complex Ginibre matrix -- the "induced measure" with
environment dimension k (Zyczkowski-Sommers 2001). k=n is the
Hilbert-Schmidt measure (the flat/uniform measure induced by the
natural volume element on the space of density matrices);
k -> infinity concentrates rho near the maximally mixed state I/n; k=1
forces A to be a single column, so A @ A^dagger is rank 1 and rho is
exactly a Haar-random PURE state (a rank-1 projector).

This SAME construction is, with no modification, exactly the standard
matrix model used to study the "Page curve": since A n x k IS the
(unnormalized) reduced density matrix of a random bipartite pure state
on C^n (x) C^k, ``InducedMeasureEnsemble(n, k)`` directly generates that
bipartite ensemble too -- ``physicskit.rmt.stats.entanglement`` provides the
matching von Neumann entropy calculation and Page's (1993) exact average
-- see ``tests/test_density_matrix.py`` for the verification (both the
mean-purity formula (n+k)/(n*k+1) and Page's average-entropy formula
were checked directly against Monte Carlo, at several (n, k) pairs,
before being trusted here).

Bures-Hall measure
-------------------
A genuinely different, non-Hilbert-Schmidt measure on density matrices,
induced by the Bures (fidelity) metric rather than the flat
Hilbert-Schmidt one. Realized here as

    rho = (I + U) @ G @ G^dagger @ (I + U)^dagger / Tr(...)

for G an independent n x n complex Ginibre matrix and U an independent
Haar-random unitary (Osipov-Sommers-Zyczkowski 2010). This construction
was verified directly here (not assumed from a remembered formula, given
how easy this measure is to get subtly wrong) by comparing Monte Carlo
samples against the EXACT joint eigenvalue density

    P(lambda) ~ prod_{i<j} (lambda_i-lambda_j)^2 / (lambda_i+lambda_j)
                * prod_i lambda_i^(-1/2),           sum_i lambda_i = 1

at n=2 (closed-form 1-D marginal, checked via a KS test) and n=3
(2-D numerical integration over the simplex, checked via mean purity) --
both matched to within Monte Carlo noise; see ``tests/test_density_matrix.py``.

Unlike the induced measure (whose joint density is a pure Vandermonde
power ``|Delta(lambda)|^2``, exactly the classical Dyson beta=2 form, so
``beta=2`` is used below), the Bures-Hall joint density has the EXTRA
prod 1/(lambda_i+lambda_j) factor -- not one of Dyson's classical
beta-ensembles, so ``beta`` is left ``None`` for it.

All three ensembles return eigenvalues already in the natural [0, 1]
range, summing to 1 (a genuine probability distribution) --
``natural_scale`` is 1.0 throughout, no rescaling needed.
"""

from __future__ import annotations

import numpy as np

from ..utils.haar import haar_unitary
from .base import MatrixEnsemble


class InducedMeasureEnsemble(MatrixEnsemble):
    """Induced-measure random density matrix: rho = A @ A^dagger / Tr(...)
    for an n x k complex Ginibre matrix A.

    Parameters
    ----------
    n : int
        Dimension of the density matrix (the traced-IN subsystem).
    k : int, optional
        Environment/other-subsystem dimension (default: n, the
        Hilbert-Schmidt measure). k=1 gives Haar-random pure states.
    """

    beta: float = 2

    def __init__(self, n: int, k: int | None = None, seed: int | np.random.Generator | None = None) -> None:
        k = n if k is None else k
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        super().__init__(n, seed=seed)
        self.k = k

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        a = (rng.standard_normal((self.n, self.k)) + 1j * rng.standard_normal((self.n, self.k))) / np.sqrt(2.0)
        rho = a @ a.conj().T
        rho = rho / np.trace(rho).real
        rho = (rho + rho.conj().T) / 2.0
        return np.linalg.eigvalsh(rho)

    def natural_scale(self) -> float:
        return 1.0


class HilbertSchmidtEnsemble(InducedMeasureEnsemble):
    """Hilbert-Schmidt random density matrix: the k=n special case of
    :class:`InducedMeasureEnsemble` (the flat/uniform measure on the
    space of n x n density matrices)."""

    def __init__(self, n: int, seed: int | np.random.Generator | None = None) -> None:
        super().__init__(n, k=n, seed=seed)


class BuresHallEnsemble(MatrixEnsemble):
    """Bures-Hall random density matrix (see module docstring)."""

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        n = self.n
        g = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2.0)
        w = g @ g.conj().T
        u = haar_unitary(n, rng)
        ip = np.eye(n) + u
        rho = ip @ w @ ip.conj().T
        rho = rho / np.trace(rho).real
        rho = (rho + rho.conj().T) / 2.0
        return np.linalg.eigvalsh(rho)

    def natural_scale(self) -> float:
        return 1.0
