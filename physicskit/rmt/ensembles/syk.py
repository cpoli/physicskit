"""Sachdev-Ye-Kitaev (SYK) ensemble -- a maximally chaotic, all-to-all
random-interaction Majorana fermion model, holographically dual (in its
low-energy, large-N limit) to Jackiw-Teitelboim gravity in 1+1
dimensions.

References
----------
S. Sachdev, J. Ye, "Gapless spin fluid ground state in a random,
quantum Heisenberg magnet", Phys. Rev. Lett. 70 (1993) 3339.
A. Kitaev, unpublished KITP talks (2015) -- the Majorana-fermion
reformulation now called "the SYK model".
J. Maldacena, D. Stanford, "Remarks on the Sachdev-Ye-Kitaev model",
Phys. Rev. D 94 (2016) 106002 -- the q=4 Hamiltonian and coupling
normalization used here.
Y.-Z. You, A. W. W. Ludwig, C. Xu, "Sachdev-Ye-Kitaev Model and
Thermalization on the Boundary of Many-Body Localized Fermionic
Symmetry Protected Topological States", Phys. Rev. B 95 (2017) 115150
-- the N mod 8 random-matrix symmetry-class periodicity mentioned below.

Unlike every other ensemble in this package, this is NOT built from
i.i.d. matrix entries: it is an explicit many-body Hamiltonian,

    H = i**(q/2) * sum_{i_1 < ... < i_q} J_{i_1...i_q} * gamma_{i_1} ... gamma_{i_q}

acting on the 2**(N/2)-dimensional Hilbert space of N Majorana fermions
gamma_1, ..., gamma_N (Hermitian operators satisfying the Clifford
algebra {gamma_a, gamma_b} = 2*delta_ab), with couplings

    J_{i_1...i_q} ~ N(0, (q-1)! * J**2 / N**(q-1))    i.i.d. over index tuples

The i**(q/2) prefactor is exactly what is needed to make H Hermitian for
any even q: reversing the order of q distinct anticommuting operators
picks up a sign (-1)**(q*(q-1)/2), and i**(q/2) is precisely the phase
that cancels it -- verified both algebraically and numerically (not
merely trusted from the literature) in ``tests/test_syk.py``: q=4 (the
default, most commonly studied case -- maximally chaotic, holographically
dual to JT gravity) needs no net phase (i**2 = -1, a real overall sign),
while q=2 (a free-fermion, NON-chaotic special case, included only for
structural comparison in the tests) needs a genuine factor of i.

The Majorana operators are realized via the standard Jordan-Wigner
tensor-product construction over N/2 qubits:

    gamma_{2k-1} = Z^(x)(k-1) (x) X (x) I^(x)(N/2-k)
    gamma_{2k}   = Z^(x)(k-1) (x) Y (x) I^(x)(N/2-k)      k = 1, ..., N/2

(x) denoting the Kronecker product -- verified in the test suite to
satisfy {gamma_a, gamma_b} = 2*delta_ab exactly, to machine precision,
rather than trusted from the construction's well-known status in the
literature.

Computational cost note (genuinely different from every other ensemble
in this package): the Hilbert space dimension is 2**(N/2), and building
H sums C(N, q) random terms, each an explicit dense matrix PRODUCT of q
Hilbert-space operators -- exponential in N, not polynomial. This
package's other ensembles scale to N in the thousands; SYK numerics
(here and in the research literature alike) are limited to N of order
10-20 for exact diagonalization. Keep ``n`` (the Majorana count) small
-- roughly <= 16 -- for interactive use.

``beta`` is left ``None``: the SYK model's random-matrix symmetry class
is known to cycle through an 8-fold pattern in N mod 8 (You-Ludwig-Xu
2017) rather than being fixed the way the classical ensembles' beta is
fixed by their construction -- a genuinely interesting further structure
this module does not classify or assert.
"""

import math
from itertools import combinations

import numpy as np

from .base import MatrixEnsemble

_PAULI_I = np.eye(2, dtype=complex)
_PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def majorana_operators(n_majorana: int) -> list[np.ndarray]:
    """The N Hermitian Majorana operators gamma_1, ..., gamma_N as dense
    2**(N/2) x 2**(N/2) complex matrices, via the standard Jordan-Wigner
    tensor-product construction (see module docstring). N must be even.
    """
    if n_majorana % 2 != 0:
        raise ValueError(f"n_majorana must be even, got {n_majorana}")
    n_qubits = n_majorana // 2
    gammas = []
    for k in range(n_qubits):
        for pauli in (_PAULI_X, _PAULI_Y):
            factors = [_PAULI_Z] * k + [pauli] + [_PAULI_I] * (n_qubits - k - 1)
            op = factors[0]
            for factor in factors[1:]:
                op = np.kron(op, factor)
            gammas.append(op)
    return gammas


def _sample_syk_hamiltonian(
    n_majorana: int,
    q: int,
    coupling: float,
    rng: np.random.Generator,
    gammas: list[np.ndarray],
) -> np.ndarray:
    dim = gammas[0].shape[0]
    h = np.zeros((dim, dim), dtype=complex)
    variance = math.factorial(q - 1) * coupling**2 / n_majorana ** (q - 1)
    std = math.sqrt(variance)
    for idx in combinations(range(n_majorana), q):
        j_coeff = rng.normal(0.0, std)
        term = gammas[idx[0]]
        for i in idx[1:]:
            term = term @ gammas[i]
        h += j_coeff * term
    return (1j ** (q // 2)) * h


class SYKEnsemble(MatrixEnsemble):
    """Sachdev-Ye-Kitaev ensemble: q-body random-interaction Majorana
    fermion Hamiltonian (see module docstring). ``n`` (inherited from
    ``MatrixEnsemble``) is the number of Majorana fermions N -- each
    sample has ``2 ** (N // 2)`` real eigenvalues, not N (matching the
    precedent set by ``physicskit.rmt.ensembles.bdg`` and
    ``physicskit.rmt.ensembles.ginibre.GinSE`` for ensembles whose eigenvalue
    count differs from the constructor's size parameter).

    Parameters
    ----------
    n : int
        Number of Majorana fermions N (must be even, and >= q). Keep
        small (roughly <= 16) -- see the module docstring's cost note.
    q : int, optional
        Interaction range: H sums products of q distinct Majorana
        operators (default 4, the standard maximally-chaotic case).
        Must be even.
    coupling : float, optional
        Overall coupling scale J (default 1.0); couplings are drawn
        with variance (q-1)! * J**2 / N**(q-1) (Maldacena-Stanford
        normalization), so the spectral width stays controlled as N
        grows at fixed J.
    """

    def __init__(
        self,
        n: int,
        q: int = 4,
        coupling: float = 1.0,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        if n % 2 != 0:
            raise ValueError(f"n (number of Majorana fermions) must be even, got {n}")
        if q % 2 != 0:
            raise ValueError(f"q must be even, got {q}")
        if q < 2 or q > n:
            raise ValueError(f"q must satisfy 2 <= q <= n, got q={q}, n={n}")
        super().__init__(n, seed=seed)
        self.q = q
        self.coupling = float(coupling)

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        gammas = majorana_operators(self.n)
        h = _sample_syk_hamiltonian(self.n, self.q, self.coupling, rng, gammas)
        return np.linalg.eigvalsh(h)

    def natural_scale(self) -> float:
        return 1.0
