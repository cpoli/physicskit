"""Non-Hermitian effective Hamiltonians for open quantum systems (Feshbach
projection / chaotic-scattering resonance ensembles).

References
----------
V.V. Sokolov, V.G. Zelevinsky, "Dynamics and statistics of unstable
quantum states", Nucl. Phys. A 504 (1989) 562.
Y.V. Fyodorov, H.-J. Sommers, "Statistics of resonance poles, phase
shifts and time delays in quantum chaotic scattering: Random matrix
approach for systems with broken time-reversal invariance", J. Math.
Phys. 38 (1997) 1918.
Y.V. Fyodorov, D.V. Savin, H.-J. Sommers, "Scattering, reflection and
impedance of waves in chaotic and disordered systems with absorption",
J. Phys. A 38 (2005) 10731 (review).

Physical picture
-----------------
A quantum system with N internal levels, coupled to M open decay/
scattering channels, is no longer described by a Hermitian Hamiltonian
once the continuum is eliminated via a Feshbach projection: the
resulting effective Hamiltonian is

    H_eff = H - (i / 2) * coupling**2 * V @ V^dagger

where H (N x N) is drawn from the closed-system ensemble (GOE/GUE/GSE)
and V (N x M) collects the amplitudes coupling each internal level to
each open channel, drawn from the matching Dyson symmetry class.
Complex eigenvalues E_k - i*Gamma_k/2 of H_eff are resonance poles:
Re(E_k) are resonance positions, Gamma_k >= 0 are widths (decay rates).
beta = 1, 2, 4 fixes both H's and V's symmetry class together
(time-reversal-invariant / broken / self-dual), so this is the direct
non-Hermitian analogue of GOE/GUE/GSE, not an unrelated construction --
see :class:`EffGOE`, :class:`EffGUE`, :class:`EffGSE`.

``m`` (number of open channels) controls the openness of the system::

    m=0:              recovers the parent Hermitian ensemble exactly
                       (H_eff = H, eigenvalues real to machine precision).
    m << n:            weak-coupling/isolated-resonance regime.
    m ~ n or m >> n:   strong-coupling/overlapping-resonance regime --
                       "resonance trapping", where a few short-lived
                       states absorb most of the total decay width and
                       n-m long-lived states remain.

Causality (widths are never negative): for ANY complex vector x with
H_eff @ x = lam * x,

    lam = (x^dagger @ H @ x - (i/2) * x^dagger @ Gamma @ x) / (x^dagger @ x)

where Gamma = coupling**2 * V @ V^dagger. x^dagger @ H @ x and
x^dagger @ x are real (H Hermitian), and x^dagger @ Gamma @ x is real
and >= 0 (Gamma Hermitian positive semi-definite), so
Im(lam) = -(x^dagger @ Gamma @ x) / (2 * x^dagger @ x) <= 0 always --
this holds for every eigenvalue of every realization, not just on
average, regardless of beta, n, or m. See ``tests/test_effective_hamiltonian.py``.

beta=4 (:class:`EffGSE`) construction note: both H and V are built from
2x2 quaternion blocks (the same embedding as :class:`~physicskit.rmt.ensembles.ginibre.GinSE`),
so H is quaternionic self-dual Hermitian and Gamma = V @ V^dagger is
quaternionic self-dual Hermitian PSD (the quaternion embedding is an
algebra homomorphism, so products and adjoints of embedded matrices stay
embedded). H_eff = H - (i/2)*Gamma is therefore a sum of two self-dual
matrices and, verified numerically during development (to ~1e-14,
machine precision -- see the test suite), its 2n eigenvalues retain
Kramers' exact two-fold degeneracy even though H_eff itself is not
Hermitian -- the same true degeneracy GSE itself has, unlike
:class:`~physicskit.rmt.ensembles.ginibre.GinSE`'s *distinct* conjugate pairs.
Only the n distinct values are returned (one representative per
degenerate pair), mirroring :class:`~physicskit.rmt.ensembles.circular.CSE`.
"""

from __future__ import annotations

import numpy as np

from .base import MatrixEnsemble


def _quaternion_block_embedding(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """N x M quaternion matrix, embedded as a 2N x 2M complex matrix via
    the standard quaternion-to-2x2-complex-block map (same convention as
    ``physicskit.rmt.ensembles.ginibre.GinSE``)."""
    a = rng.standard_normal((rows, cols))
    b = rng.standard_normal((rows, cols))
    c = rng.standard_normal((rows, cols))
    d = rng.standard_normal((rows, cols))
    m = np.zeros((2 * rows, 2 * cols), dtype=complex)
    m[0::2, 0::2] = a + 1j * b
    m[0::2, 1::2] = c + 1j * d
    m[1::2, 0::2] = -c + 1j * d
    m[1::2, 1::2] = a - 1j * b
    return m


def _sample_hamiltonian(n: int, rng: np.random.Generator, beta: float) -> np.ndarray:
    """Dense closed-system Hamiltonian H for the given Dyson index."""
    if beta == 1:
        x = rng.standard_normal((n, n))
        return (x + x.T) / np.sqrt(2.0)
    if beta == 2:
        x = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2.0)
        return (x + x.conj().T) / 2.0
    if beta == 4:
        m = _quaternion_block_embedding(n, n, rng)
        return (m + m.conj().T) / 2.0  # quaternionic self-dual Hermitian
    raise ValueError(f"beta must be 1, 2, or 4, got {beta}")


def _sample_coupling(n: int, m: int, rng: np.random.Generator, beta: float) -> np.ndarray:
    """Dense N x M (or, for beta=4, 2N x 2M) coupling matrix V."""
    if beta == 1:
        return rng.standard_normal((n, m))
    if beta == 2:
        return (rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))) / np.sqrt(2.0)
    if beta == 4:
        return _quaternion_block_embedding(n, m, rng) / np.sqrt(2.0)
    raise ValueError(f"beta must be 1, 2, or 4, got {beta}")


class EffectiveHamiltonianEnsemble(MatrixEnsemble):
    """Non-Hermitian effective Hamiltonian H_eff = H - (i/2) * coupling**2 * V @ V^dagger.

    Use the named subclasses (:class:`EffGOE`, :class:`EffGUE`,
    :class:`EffGSE`) for the classical beta=1,2,4 cases.

    Parameters
    ----------
    n : int
        Dimension of the closed system (number of resonances).
    m : int
        Number of open decay/scattering channels (m=0 recovers the
        parent Hermitian ensemble exactly).
    beta : {1, 2, 4}
        Dyson index shared by the parent Hamiltonian H and the coupling
        matrix V.
    coupling : float, optional
        Overall coupling strength scaling V (default 1.0); larger values
        drive the system from the isolated-resonance regime towards
        resonance trapping without changing beta or the number of
        channels m.
    seed : int, numpy.random.Generator, or None, optional
        Seed for reproducible sampling. See
        :func:`~physicskit.rmt.utils.random_state.as_generator`.
    """

    def __init__(
        self,
        n: int,
        m: int,
        beta: int,
        coupling: float = 1.0,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        if beta not in (1, 2, 4):
            raise ValueError(f"beta must be 1, 2, or 4, got {beta}")
        if m < 0:
            raise ValueError(f"m (number of open channels) must be >= 0, got {m}")
        super().__init__(n, seed=seed)
        self.m = m
        self.beta: float = beta
        self.coupling = float(coupling)

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        n, m, beta = self.n, self.m, self.beta
        h = _sample_hamiltonian(n, rng, beta)
        if m > 0:
            v = self.coupling * _sample_coupling(n, m, rng, beta)
            gamma = v @ v.conj().T
        else:
            gamma = np.zeros_like(h)
        h_eff = h - 0.5j * gamma
        eigs = np.linalg.eigvals(h_eff)
        if beta == 4:
            # Exact Kramers double degeneracy (see module docstring) --
            # keep one representative per pair, as CSE does.
            order = np.lexsort((eigs.imag, eigs.real))
            eigs = eigs[order][0::2]
        return eigs.astype(complex)

    def natural_scale(self) -> float:
        # Re(eigenvalues) inherit the parent Hermitian ensemble's level
        # scale (exactly, at m=0; approximately for weak-to-moderate
        # coupling) -- see HermiteBetaEnsemble.natural_scale. Widths
        # (Im part) have no comparably universal n-independent scale --
        # they depend on m and coupling as well as n -- so this factor
        # should be treated as a Re-axis convenience, not a full
        # normalization of the complex plane.
        return np.sqrt(self.n * self.beta)


class EffGOE(EffectiveHamiltonianEnsemble):
    """Effective Hamiltonian built on GOE (beta=1): time-reversal
    invariant closed system, real coupling to the open channels."""

    def __init__(
        self,
        n: int,
        m: int,
        coupling: float = 1.0,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(n, m, beta=1, coupling=coupling, seed=seed)


class EffGUE(EffectiveHamiltonianEnsemble):
    """Effective Hamiltonian built on GUE (beta=2): broken time-reversal
    symmetry, complex coupling to the open channels."""

    def __init__(
        self,
        n: int,
        m: int,
        coupling: float = 1.0,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(n, m, beta=2, coupling=coupling, seed=seed)


class EffGSE(EffectiveHamiltonianEnsemble):
    """Effective Hamiltonian built on GSE (beta=4): self-dual
    (Kramers-degenerate) closed system, quaternionic coupling to the
    open channels. ``n`` is the quaternionic dimension; each sample
    produces n distinct complex resonances (each an exact Kramers pair
    internally -- see module docstring)."""

    def __init__(
        self,
        n: int,
        m: int,
        coupling: float = 1.0,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(n, m, beta=4, coupling=coupling, seed=seed)
