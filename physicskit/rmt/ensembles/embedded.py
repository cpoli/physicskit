"""Embedded Gaussian ensembles (EGOE(k)/EGUE(k)) and the Two-Body Random
Ensemble (TBRE, k=2) -- random k-body interactions among m fermions
distributed over N single-particle levels.

References
----------
K. K. Mon, J. B. French, "Statistical properties of many-particle
spectra", Ann. Phys. 95 (1975) 90 -- the k-body embedded ensemble
construction and the celebrated result that the many-body density of
states approaches a GAUSSIAN (not the semicircle law) as m, N -> infinity
at fixed k >= 2, k < m.
T. A. Brody, J. Flores, J. B. French, P. A. Mello, A. Pandey,
S. S. M. Wong, "Random-matrix physics: spectrum and strength
fluctuations", Rev. Mod. Phys. 53 (1981) 385 -- review.
V. K. B. Kota, "Embedded Random Matrix Ensembles in Quantum Physics",
Lecture Notes in Physics 884, Springer, 2014 -- comprehensive reference.
T. Papenbrock, H. A. Weidenmuller, "Colloquium: Random matrices and
chaos in nuclear physics", Rev. Mod. Phys. 79 (2007) 997.

Unlike every classical ensemble in this package (i.i.d. matrix entries),
the many-body Hamiltonian matrix here is built by EMBEDDING a random
k-body interaction into the m-particle Fock space: the interaction
itself is a Hermitian random (GOE/GUE-type) matrix V on the space of
k-particle Slater determinants (dimension C(N, k)), and the m-particle
Hamiltonian is

    H = sum_{I, J} V_{I, J} * a^dagger_I a_J

summed over all k-index tuples I, J (I = i_1 < ... < i_k, similarly J),
with a^dagger_I = a^dagger_{i_1} ... a^dagger_{i_k} and
a_J = a_{j_k} ... a_{j_1} (reversed order, so a^dagger_I a_J is built
from creation operators applied in ASCENDING index order and
annihilation operators in DESCENDING index order -- get this backwards
and Hermiticity silently survives but the exact k=m identity below does
not, which is exactly how an initial, subtly wrong sign convention
during development was caught).

Because H is built from a SHARED set of k-body matrix elements rather
than independent entries, the resulting many-body matrix elements are
strongly correlated (unlike GOE) -- this correlation structure is
exactly what drives the many-body density of states toward a Gaussian
rather than a semicircle as m, N grow (Mon-French 1975), the qualitative
signature this ensemble exists to demonstrate.

Implementation: creation/annihilation operators are realized via the
standard Jordan-Wigner tensor-product construction over N qubits (one
per single-particle level, ``|0>``/``|1>`` = unoccupied/occupied) -- the SAME
technique already used for SYK's Majorana operators
(``physicskit.rmt.ensembles.syk``), just for fermionic number-conserving ladder
operators instead. H is built on the full 2^N Fock space, then restricted
to the fixed-m-particle sector (dimension C(N, m)) by selecting the
appropriate basis states -- exact, not an approximation, since
number-conserving k-body terms never mix different particle-number
sectors.

Verified during development (see ``tests/test_embedded.py``), independent
of trusting the construction's well-known status in the literature:

- H is exactly Hermitian and exactly commutes with the total particle
  number operator (both to machine precision).
- The celebrated special case k = m: with only m particles present, a
  k=m-body interaction acts on all of them at once, so the embedding is
  trivial and the m-particle Hamiltonian reduces EXACTLY (not merely
  statistically) to the k-body matrix V itself -- checked directly for
  k=m=2 and k=m=3 to machine precision, catching the operator-ordering
  sign bug mentioned above during development.

Cost note (genuinely different from the classical ensembles, exactly
like SYK): the Fock-space dimension is 2^N, and building H requires
O(C(N,k)^2) dense Hilbert-space operator products -- exponential in N,
not polynomial. Keep ``n_levels`` small (roughly <= 10) for interactive
use.

``beta`` in {1, 2} selects a real-symmetric (EGOE, "orthogonal") or
complex-Hermitian (EGUE, "unitary") random two-body interaction V; no
simple universal ``natural_scale`` rescaling is attempted (unlike GOE's
sqrt(n), the many-body spectral width here depends on N, m, and k
through a nontrivial combinatorial moment formula -- see Mon-French
1975 -- not implemented here).
"""

from itertools import combinations

import numpy as np

from .base import MatrixEnsemble

_I2 = np.eye(2, dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_CREATE_LOCAL = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)  # |0>=unoccupied -> |1>=occupied
_ANNIHILATE_LOCAL = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


def _fermion_ladder_operators(n_levels: int) -> list[np.ndarray]:
    """The N fermionic creation operators a^dagger_1, ..., a^dagger_N as
    dense 2^N x 2^N complex matrices (Jordan-Wigner construction; the
    corresponding annihilation operators are their conjugate transposes).
    """
    creators = []
    for k in range(n_levels):
        factors = [_Z] * k + [_CREATE_LOCAL] + [_I2] * (n_levels - k - 1)
        op = factors[0]
        for factor in factors[1:]:
            op = np.kron(op, factor)
        creators.append(op)
    return creators


class EmbeddedGaussianEnsemble(MatrixEnsemble):
    """EGOE(k)/EGUE(k): m fermions in N single-particle levels,
    interacting via a random k-body interaction (see module docstring).

    Parameters
    ----------
    n_particles : int
        Number of fermions m.
    n_levels : int
        Number of single-particle levels N (>= n_particles). Keep small
        (roughly <= 10) -- see the module docstring's cost note.
    k : int, optional
        Interaction body-rank (default 2, i.e. TBRE). Must satisfy
        2 <= k <= n_particles.
    beta : int, optional
        1 (real symmetric interaction, EGOE) or 2 (complex Hermitian,
        EGUE).
    """

    def __init__(
        self,
        n_particles: int,
        n_levels: int,
        k: int = 2,
        beta: int = 1,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        if beta not in (1, 2):
            raise ValueError(f"beta must be 1 or 2, got {beta}")
        if not 2 <= k <= n_particles <= n_levels:
            raise ValueError(f"require 2 <= k <= n_particles <= n_levels; got k={k}, n_particles={n_particles}, n_levels={n_levels}")
        super().__init__(n_particles, seed=seed)
        self.n_levels = n_levels
        self.k = k
        self.beta: float = beta
        self._p_ops: list[np.ndarray] | None = None
        self._basis: list[int] | None = None

    def _operators(self) -> tuple[list[np.ndarray], list[int]]:
        """The k-body creation operators P_a (one per k-subset of levels)
        and the fixed-m-particle basis indices -- independent of the
        random interaction V, so computed once and cached rather than
        redone on every ``_sample_eigenvalues`` call.
        """
        if self._p_ops is None:
            n_levels, k, m = self.n_levels, self.k, self.n
            creators = _fermion_ladder_operators(n_levels)
            tuples = list(combinations(range(n_levels), k))

            def multi_create(idxs: tuple[int, ...]) -> np.ndarray:
                op = creators[idxs[0]]
                for idx in idxs[1:]:
                    op = op @ creators[idx]
                return op

            self._p_ops = [multi_create(t) for t in tuples]
            dim = self._p_ops[0].shape[0]
            self._basis = [state for state in range(dim) if bin(state).count("1") == m]
        return self._p_ops, self._basis

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        p_ops, basis = self._operators()
        n_tuples = len(p_ops)

        if self.beta == 1:
            x = rng.standard_normal((n_tuples, n_tuples))
            v = (x + x.T) / np.sqrt(2.0)
        else:
            x = (rng.standard_normal((n_tuples, n_tuples)) + 1j * rng.standard_normal((n_tuples, n_tuples))) / np.sqrt(2.0)
            v = (x + x.conj().T) / 2.0

        dim = p_ops[0].shape[0]
        h = np.zeros((dim, dim), dtype=complex)
        for a_idx in range(n_tuples):
            q = sum(v[a_idx, b_idx] * p_ops[b_idx].conj().T for b_idx in range(n_tuples))
            h += p_ops[a_idx] @ q

        h_sub = h[np.ix_(basis, basis)]
        h_sub = (h_sub + h_sub.conj().T) / 2.0
        return np.linalg.eigvalsh(h_sub)

    def natural_scale(self) -> float:
        return 1.0


class TwoBodyRandomEnsemble(EmbeddedGaussianEnsemble):
    """TBRE: the k=2 special case of :class:`EmbeddedGaussianEnsemble`,
    by far the most commonly studied (nuclear shell-model-type) case."""

    def __init__(
        self,
        n_particles: int,
        n_levels: int,
        beta: int = 1,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(n_particles, n_levels, k=2, beta=beta, seed=seed)
