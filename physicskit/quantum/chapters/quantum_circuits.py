r"""A small state-vector quantum-circuit simulator and three textbook algorithms.

An :math:`n`-qubit register is a vector of :math:`2^n` amplitudes, stored as
an array of shape ``(2,) * n`` so that a gate on :math:`k` qubits is a
tensor contraction over those :math:`k` axes, at cost :math:`O(2^n)` per
gate. Qubit 0 is the most significant bit of a basis-state index:
:math:`|q_0 q_1 \dots q_{n-1}\rangle = |x\rangle`,
:math:`x = \sum_j q_j 2^{n-1-j}`.

Three algorithms show what a quantum computer does differently:

- Deutsch-Jozsa (1992): one query of a phase oracle :math:`(-1)^{f(x)}`
  tells whether :math:`f` is constant or balanced, where a classical
  deterministic algorithm needs :math:`2^{n-1} + 1` queries.
- The quantum Fourier transform (Coppersmith 1994; Shor 1994),
  :math:`|x\rangle \to 2^{-n/2}\sum_y e^{2\pi ixy/2^n}|y\rangle`, built from
  :math:`n(n+1)/2` gates, at the heart of Shor's period finding.
- Grover search (1996): about :math:`\tfrac{\pi}{4}\sqrt{N/M}` oracle
  queries find one of :math:`M` marked items among :math:`N`, against
  :math:`O(N/M)` classically.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "GATES",
    "QuantumCircuit",
    "deutsch_jozsa_circuit",
    "grover_circuit",
    "grover_optimal_iterations",
    "grover_success_probability",
    "modular_exponentiation_circuit",
    "qft_circuit",
    "shor_period_from_measurement",
]

_S2 = 1.0 / np.sqrt(2.0)
GATES: dict[str, NDArray[np.complex128]] = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    "H": np.array([[_S2, _S2], [_S2, -_S2]], dtype=complex),
    "S": np.array([[1, 0], [0, 1j]], dtype=complex),
    "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
}


def _phase(theta: float) -> NDArray[np.complex128]:
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)


class QuantumCircuit:
    """A list of gates on ``n_qubits`` qubits, simulated on a state vector.

    Gates are appended with the methods below, which return the circuit so
    calls can be chained, and :meth:`run` applies them in order.

    Parameters
    ----------
    n_qubits : int

    Examples
    --------
    A Bell pair:

    >>> qc = QuantumCircuit(2).h(0).cx(0, 1)
    >>> np.round(qc.run(), 4).tolist()
    [(0.7071+0j), 0j, 0j, (0.7071+0j)]
    """

    def __init__(self, n_qubits: int):
        if n_qubits < 1:
            raise ValueError("need at least one qubit")
        self.n = int(n_qubits)
        # (name, gate matrix / diagonal phases / permutation, qubits)
        self.ops: list[tuple[str, NDArray, tuple[int, ...]]] = []

    # --- building -------------------------------------------------------------------

    def gate(self, U: ArrayLike, *qubits: int, name: str = "U") -> QuantumCircuit:
        """Append a :math:`2^k \\times 2^k` unitary acting on ``qubits`` (first is most significant)."""
        U = np.asarray(U, dtype=complex)
        k = len(qubits)
        if U.shape != (2**k, 2**k):
            raise ValueError(f"a gate on {k} qubits must be {2**k}x{2**k}")
        if len(set(qubits)) != k or not all(0 <= q < self.n for q in qubits):
            raise ValueError("qubits must be distinct and in range")
        self.ops.append((name, U, tuple(qubits)))
        return self

    def h(self, q: int) -> QuantumCircuit:
        """Hadamard."""
        return self.gate(GATES["H"], q, name="H")

    def x(self, q: int) -> QuantumCircuit:
        """Pauli X (NOT)."""
        return self.gate(GATES["X"], q, name="X")

    def z(self, q: int) -> QuantumCircuit:
        """Pauli Z."""
        return self.gate(GATES["Z"], q, name="Z")

    def phase(self, theta: float, q: int) -> QuantumCircuit:
        """Phase gate :math:`\\mathrm{diag}(1, e^{i\\theta})`."""
        return self.gate(_phase(theta), q, name="P")

    def cx(self, control: int, target: int) -> QuantumCircuit:
        """Controlled NOT."""
        U = np.eye(4, dtype=complex)
        U[2:, 2:] = GATES["X"]
        return self.gate(U, control, target, name="CX")

    def cphase(self, theta: float, control: int, target: int) -> QuantumCircuit:
        """Controlled phase :math:`\\mathrm{diag}(1, 1, 1, e^{i\\theta})`."""
        return self.gate(np.diag([1, 1, 1, np.exp(1j * theta)]), control, target, name="CP")

    def swap(self, a: int, b: int) -> QuantumCircuit:
        """Swap two qubits."""
        return self.gate(np.eye(4)[[0, 2, 1, 3]], a, b, name="SWAP")

    def diagonal(self, phases: ArrayLike, qubits: tuple[int, ...] | None = None) -> QuantumCircuit:
        """Diagonal unitary :math:`|x\\rangle \\to e^{i\\phi_x}|x\\rangle` on ``qubits`` (default all)."""
        qubits = tuple(range(self.n)) if qubits is None else tuple(qubits)
        phases = np.asarray(phases, dtype=float)
        if phases.size != 2 ** len(qubits):
            raise ValueError("need one phase per basis state")
        self.ops.append(("D", np.exp(1j * phases), qubits))
        return self

    def permutation(self, perm: ArrayLike, qubits: tuple[int, ...] | None = None) -> QuantumCircuit:
        """Reversible classical gate :math:`|x\\rangle \\to |\\pi(x)\\rangle`."""
        qubits = tuple(range(self.n)) if qubits is None else tuple(qubits)
        perm = np.asarray(perm, dtype=int)
        if sorted(perm.tolist()) != list(range(2 ** len(qubits))):
            raise ValueError("perm must be a permutation of the basis states")
        self.ops.append(("PERM", perm, qubits))
        return self

    def append(self, other: QuantumCircuit, offset: int = 0) -> QuantumCircuit:
        """Append the gates of ``other`` with its qubit ``j`` mapped to ``j + offset``."""
        for name, U, qubits in other.ops:
            self.ops.append((name, U, tuple(q + offset for q in qubits)))
        return self

    def inverse(self) -> QuantumCircuit:
        """The adjoint circuit: gates in reverse order, each inverted."""
        inv = QuantumCircuit(self.n)
        for name, U, qubits in reversed(self.ops):
            if name == "D":
                inv.ops.append((name, np.conj(U), qubits))
            elif name == "PERM":
                inv.ops.append((name, np.argsort(U), qubits))
            else:
                inv.ops.append((name + "^-1", U.conj().T, qubits))
        return inv

    # --- simulation -----------------------------------------------------------------

    def _apply(self, psi: NDArray[np.complex128], name: str, U: NDArray, qubits: tuple[int, ...]) -> NDArray[np.complex128]:
        k = len(qubits)
        moved = np.moveaxis(psi, qubits, range(k))
        shape = moved.shape
        flat = moved.reshape(2**k, -1)
        if name == "D":
            flat = U[:, None] * flat
        elif name == "PERM":
            out = np.empty_like(flat)
            out[U] = flat
            flat = out
        else:
            flat = U @ flat
        return np.moveaxis(flat.reshape(shape), range(k), qubits)

    def run(self, initial: ArrayLike | int = 0) -> NDArray[np.complex128]:
        """Apply the circuit to a basis state (an integer) or a state vector.

        Parameters
        ----------
        initial : int or array_like, default 0
            Basis-state index, or a vector of :math:`2^n` amplitudes.

        Returns
        -------
        ndarray of complex, shape (2**n,)
        """
        if isinstance(initial, (int, np.integer)):
            psi = np.zeros(2**self.n, dtype=complex)
            psi[int(initial)] = 1.0
        else:
            psi = np.asarray(initial, dtype=complex).copy()
        psi = psi.reshape((2,) * self.n)
        for name, U, qubits in self.ops:
            psi = self._apply(psi, name, U, qubits)
        return psi.reshape(-1)

    def unitary(self) -> NDArray[np.complex128]:
        """The full :math:`2^n \\times 2^n` matrix of the circuit (small ``n`` only)."""
        return np.column_stack([self.run(x) for x in range(2**self.n)])

    @staticmethod
    def probabilities(psi: ArrayLike, n_qubits: int, qubits: tuple[int, ...] | None = None) -> NDArray[np.float64]:
        """Measurement probabilities of ``qubits`` (default all), marginalizing the rest.

        Parameters
        ----------
        psi : array_like, shape (2**n_qubits,)
        n_qubits : int
        qubits : tuple of int, optional

        Returns
        -------
        ndarray of float, shape (2**len(qubits),)
        """
        p = np.abs(np.asarray(psi).reshape((2,) * n_qubits)) ** 2
        if qubits is None:
            return p.reshape(-1)
        others = tuple(q for q in range(n_qubits) if q not in qubits)
        p = p.sum(axis=others)
        # sum keeps the remaining axes in increasing order; reorder to `qubits`
        order = np.argsort(np.argsort(qubits))
        return np.transpose(p, order).reshape(-1) if len(qubits) > 1 else p.reshape(-1)

    @staticmethod
    def sample(probabilities: ArrayLike, shots: int, seed: int | None = None) -> NDArray[np.int64]:
        """Counts of each outcome in ``shots`` projective measurements."""
        p = np.asarray(probabilities, dtype=float)
        return np.random.default_rng(seed).multinomial(shots, p / p.sum())

    def gate_count(self) -> dict[str, int]:
        """Number of gates of each kind."""
        counts: dict[str, int] = {}
        for name, _, _ in self.ops:
            counts[name] = counts.get(name, 0) + 1
        return counts


def deutsch_jozsa_circuit(f: ArrayLike) -> QuantumCircuit:
    """Deutsch-Jozsa circuit for a Boolean function given by its truth table.

    :math:`H^{\\otimes n}`, the phase oracle :math:`|x\\rangle \\to
    (-1)^{f(x)}|x\\rangle` (one query), and :math:`H^{\\otimes n}` again. The
    probability of reading :math:`|0\\dots0\\rangle` is
    :math:`|2^{-n}\\sum_x(-1)^{f(x)}|^2`: 1 if :math:`f` is constant and 0 if
    it is balanced (Deutsch and Jozsa, Proc. R. Soc. A 439, 553 (1992);
    Cleve et al., Proc. R. Soc. A 454, 339 (1998)).

    Parameters
    ----------
    f : array_like of {0, 1}, shape (2**n,)

    Returns
    -------
    QuantumCircuit

    Examples
    --------
    >>> balanced = [0, 1, 1, 0, 1, 0, 0, 1]
    >>> psi = deutsch_jozsa_circuit(balanced).run()
    >>> round(float(abs(psi[0]) ** 2), 12)
    0.0
    """
    f = np.asarray(f, dtype=int)
    n = int(np.log2(f.size))
    if 2**n != f.size:
        raise ValueError("truth table length must be a power of two")
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    qc.diagonal(np.pi * f)
    for q in range(n):
        qc.h(q)
    return qc


def qft_circuit(n: int, swaps: bool = True) -> QuantumCircuit:
    """Quantum Fourier transform on ``n`` qubits from Hadamards and controlled phases.

    Qubit :math:`j` gets a Hadamard followed by controlled phases
    :math:`2\\pi/2^{k-j+1}` from each later qubit :math:`k`; the final swaps
    reverse the qubit order (Coppersmith, IBM Research Report RC19642
    (1994); Nielsen and Chuang, sec. 5.1).

    Parameters
    ----------
    n : int
    swaps : bool, default True

    Returns
    -------
    QuantumCircuit

    Examples
    --------
    >>> N = 8
    >>> F = np.exp(2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N) / np.sqrt(N)
    >>> bool(np.allclose(qft_circuit(3).unitary(), F))
    True
    """
    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            qc.cphase(2 * np.pi / 2 ** (k - j + 1), k, j)
    if swaps:
        for j in range(n // 2):
            qc.swap(j, n - 1 - j)
    return qc


def grover_optimal_iterations(n_items: int, n_marked: int = 1) -> int:
    """:math:`\\lfloor \\pi/(4\\theta) \\rfloor` with :math:`\\sin\\theta = \\sqrt{M/N}`."""
    theta = np.arcsin(np.sqrt(n_marked / n_items))
    return int(np.floor(np.pi / (4 * theta)))


def grover_success_probability(k: ArrayLike, n_items: int, n_marked: int = 1) -> NDArray[np.float64]:
    """Probability of measuring a marked item after ``k`` iterations, :math:`\\sin^2((2k+1)\\theta)`.

    Examples
    --------
    >>> round(float(grover_success_probability(1, 4)), 12)  # N = 4: one query suffices
    1.0
    """
    theta = np.arcsin(np.sqrt(n_marked / n_items))
    return np.sin((2 * np.asarray(k) + 1) * theta) ** 2


def grover_circuit(n: int, marked: ArrayLike, iterations: int | None = None) -> QuantumCircuit:
    """Grover search for the ``marked`` basis states of ``n`` qubits.

    Starting from the uniform superposition, each iteration applies the
    phase oracle :math:`O|x\\rangle = -|x\\rangle` for marked :math:`x` and the
    diffusion operator :math:`2|s\\rangle\\langle s| - I = H^{\\otimes n}(2|0\\rangle\\langle 0| - I)H^{\\otimes n}`,
    rotating the state by :math:`2\\theta` toward the marked subspace (Grover,
    Phys. Rev. Lett. 79, 325 (1997)).

    Parameters
    ----------
    n : int
    marked : array_like of int
        Indices of the marked items.
    iterations : int, optional
        Defaults to :func:`grover_optimal_iterations`.

    Returns
    -------
    QuantumCircuit

    Examples
    --------
    >>> psi = grover_circuit(5, [19]).run()
    >>> bool(abs(psi[19]) ** 2 > 0.99)
    True
    """
    N = 2**n
    marked = np.atleast_1d(np.asarray(marked, dtype=int))
    if iterations is None:
        iterations = grover_optimal_iterations(N, marked.size)
    oracle = np.zeros(N)
    oracle[marked] = np.pi
    reflect0 = np.full(N, np.pi)
    reflect0[0] = 0.0  # diag(1, -1, ..., -1) = 2|0><0| - I
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for _ in range(iterations):
        qc.diagonal(oracle)
        for q in range(n):
            qc.h(q)
        qc.diagonal(reflect0)
        for q in range(n):
            qc.h(q)
    return qc


def modular_exponentiation_circuit(a: int, N: int, n_count: int) -> tuple[QuantumCircuit, int]:
    """Prepare :math:`2^{-t/2}\\sum_x |x\\rangle|a^x \\bmod N\\rangle` for Shor's period finding.

    The counting register (``n_count`` qubits) is put in uniform
    superposition, and the reversible map :math:`|x\\rangle|y\\rangle \\to
    |x\\rangle|y \\oplus (a^x \\bmod N)\\rangle` is applied as one permutation
    gate, which stands in for the arithmetic circuit a real device would
    compile.

    Parameters
    ----------
    a : int
        Base, coprime to ``N``.
    N : int
        Number to factor.
    n_count : int
        Counting qubits :math:`t`, typically with :math:`N^2 \\le 2^t`.

    Returns
    -------
    circuit : QuantumCircuit
        On ``n_count + m`` qubits, counting register first.
    m : int
        Work-register size, :math:`\\lceil\\log_2 N\\rceil`.
    """
    if gcd(a, N) != 1:
        raise ValueError("a must be coprime to N")
    m = int(np.ceil(np.log2(N)))
    n_total = n_count + m
    x = np.arange(2**n_count)[:, None]
    y = np.arange(2**m)[None, :]
    fx = np.array([pow(a, int(v), N) for v in range(2**n_count)])[:, None]
    perm = (x * 2**m + (y ^ fx)).reshape(-1)
    qc = QuantumCircuit(n_total)
    for q in range(n_count):
        qc.h(q)
    qc.permutation(perm)
    return qc, m


def shor_period_from_measurement(y: int, n_count: int, N: int) -> int:
    """Period candidate from a counting-register outcome, by continued fractions.

    :math:`y/2^t \\approx s/r`; the denominator of the best approximation with
    :math:`r < N` is returned.

    Examples
    --------
    >>> shor_period_from_measurement(192, 8, 15)  # 192/256 = 3/4
    4
    """
    from fractions import Fraction

    return Fraction(y, 2**n_count).limit_denominator(N - 1).denominator
