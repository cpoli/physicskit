r"""Matrix product states and time-evolving block decimation (TEBD) for spin-1/2 chains.

A state of :math:`N` spins needs :math:`2^N` amplitudes, but the low-energy
states of gapped 1D Hamiltonians obey an area law: the entanglement across
any cut is bounded. Such states are efficiently written as matrix product
states (MPS; Fannes, Nachtergaele and Werner 1992; Östlund and Rommer 1995,
explaining White's 1992 DMRG),

.. math::

    |\psi\rangle = \sum_{s_1\dots s_N} B^{s_1}_1 B^{s_2}_2 \cdots B^{s_N}_N\,
    |s_1 \dots s_N\rangle,

with matrices of bond dimension :math:`\chi`. Vidal's TEBD (Phys. Rev. Lett.
91, 147902 (2003); 93, 040502 (2004)) evolves an MPS under a nearest-neighbour
Hamiltonian :math:`H = \sum_j h_{j,j+1}` by a Trotter splitting into even
and odd bonds,

.. math::

    e^{-iH\delta t} \approx e^{-iH_{\text{even}}\delta t/2}\, e^{-iH_{\text{odd}}\delta t}\,
    e^{-iH_{\text{even}}\delta t/2} + O(\delta t^3),

applying each two-site gate and truncating the bond back to :math:`\chi`
by a singular value decomposition. In imaginary time
(:math:`\delta t \to -i\delta\tau`) the same algorithm projects onto the
ground state.

This module keeps the MPS in right-canonical form with the singular values
on every bond (Hauschild and Pollmann, SciPost Phys. Lect. Notes 5 (2018)).
Conventions match :mod:`physicskit.condensed.spin_chains`: physical index 0
is spin up (:math:`\sigma^z = +1`), and :meth:`MPS.to_dense` orders basis
states with site 0 as the most significant bit and bit 1 meaning up.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

__all__ = ["MPS", "tebd", "tfim_bond_hamiltonians", "xxz_bond_hamiltonians"]

_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


def xxz_bond_hamiltonians(N: int, J: float = 1.0, delta: float = 1.0, h: float = 0.0) -> list[NDArray[np.complex128]]:
    r"""Two-site terms of the open XXZ chain, :math:`H = J\sum(S^xS^x + S^yS^y + \Delta S^zS^z) - h\sum S^z`.

    Each bond operator is a :math:`4\times4` matrix on (site :math:`j`, site
    :math:`j+1`); the field of each site is split between its bonds so that
    :math:`\sum_j h_{j,j+1}` is exactly the Hamiltonian of
    :func:`physicskit.condensed.spin_chains.xxz_hamiltonian` with
    ``periodic=False``.

    Parameters
    ----------
    N : int
    J, delta, h : float

    Returns
    -------
    list of ndarray, length ``N - 1``
    """
    Sx, Sy, Sz = _SX / 2, _SY / 2, _SZ / 2
    coupling = J * (np.kron(Sx, Sx) + np.kron(Sy, Sy) + delta * np.kron(Sz, Sz))
    return _with_fields(N, coupling, -h * Sz)


def tfim_bond_hamiltonians(N: int, J: float = 1.0, h: float = 1.0) -> list[NDArray[np.complex128]]:
    r"""Two-site terms of the open transverse-field Ising chain, :math:`H = -J\sum\sigma^x\sigma^x - h\sum\sigma^z`.

    Matches :func:`physicskit.condensed.spin_chains.tfim_hamiltonian` with
    ``periodic=False``.

    Parameters
    ----------
    N : int
    J, h : float

    Returns
    -------
    list of ndarray, length ``N - 1``
    """
    return _with_fields(N, -J * np.kron(_SX, _SX), -h * _SZ)


def _with_fields(N, coupling, onsite):
    bonds = []
    for j in range(N - 1):
        wl = 1.0 if j == 0 else 0.5
        wr = 1.0 if j == N - 2 else 0.5
        bonds.append(coupling + wl * np.kron(onsite, _I2) + wr * np.kron(_I2, onsite))
    return bonds


class MPS:
    """An open-boundary matrix product state in right-canonical form.

    Attributes
    ----------
    B : list of ndarray
        Site tensors of shape ``(chi_left, 2, chi_right)``, each right-canonical.
    S : list of ndarray
        Singular values on the bond to the left of each site (``S[0] = [1]``).

    Examples
    --------
    >>> psi = MPS.product_state([0, 1, 0, 1])  # up, down, up, down
    >>> v = psi.to_dense()
    >>> int(np.argmax(np.abs(v))), float(abs(v).max())
    (10, 1.0)
    """

    def __init__(self, B: list[NDArray], S: list[NDArray]):
        self.B = [np.asarray(b, dtype=complex) for b in B]
        self.S = [np.asarray(s, dtype=float) for s in S]
        self.N = len(self.B)

    @classmethod
    def product_state(cls, spins) -> MPS:
        """Product state with physical index ``spins[j]`` (0 up, 1 down) on site ``j``."""
        B, S = [], []
        for s in spins:
            b = np.zeros((1, 2, 1), dtype=complex)
            b[0, int(s), 0] = 1.0
            B.append(b)
            S.append(np.ones(1))
        return cls(B, S)

    def copy(self) -> MPS:
        """Deep copy."""
        return MPS([b.copy() for b in self.B], [s.copy() for s in self.S])

    @property
    def bond_dimensions(self) -> list[int]:
        """Bond dimensions :math:`\\chi` of the ``N - 1`` internal bonds."""
        return [b.shape[2] for b in self.B[:-1]]

    def to_dense(self) -> NDArray[np.complex128]:
        """The full :math:`2^N` state vector, in the basis order of :mod:`physicskit.condensed.spin_chains`."""
        psi = self.B[0]
        for b in self.B[1:]:
            psi = np.tensordot(psi, b, axes=(psi.ndim - 1, 0))
        psi = psi.reshape(-1)
        # physical index 0 is up, but condensed uses bit 1 for up: flip all bits
        return psi[::-1] * self.S[0][0]

    def entanglement_entropy(self) -> NDArray[np.float64]:
        """Von Neumann entropy :math:`-\\sum_\\alpha s_\\alpha^2\\ln s_\\alpha^2` of each internal bond."""
        out = []
        for s in self.S[1:]:
            p = s[s > 1e-15] ** 2
            out.append(float(-np.sum(p * np.log(p))))
        return np.array(out)

    def expectation_local(self, op: NDArray) -> NDArray[np.float64]:
        """:math:`\\langle O_j \\rangle` of a single-site operator on every site."""
        out = []
        for j in range(self.N):
            theta = np.tensordot(np.diag(self.S[j]), self.B[j], axes=(1, 0))  # (vL, p, vR)
            out.append(np.real(np.tensordot(theta.conj(), np.tensordot(op, theta, axes=(1, 1)), axes=([0, 1, 2], [1, 0, 2]))))
        return np.array(out)

    def _theta(self, j: int) -> NDArray[np.complex128]:
        th = np.tensordot(np.diag(self.S[j]), self.B[j], axes=(1, 0))
        return np.tensordot(th, self.B[j + 1], axes=(2, 0))  # (vL, p1, p2, vR)

    def bond_energies(self, H_bonds: list[NDArray]) -> NDArray[np.float64]:
        """:math:`\\langle h_{j,j+1}\\rangle` for every bond."""
        out = []
        for j, h in enumerate(H_bonds):
            th = self._theta(j)
            hth = np.tensordot(h.reshape(2, 2, 2, 2), th, axes=([2, 3], [1, 2]))  # (p1, p2, vL, vR)
            out.append(np.real(np.tensordot(th.conj(), hth, axes=([0, 1, 2, 3], [2, 0, 1, 3]))))
        return np.array(out)

    def canonicalize(self) -> None:
        """Restore the normalized right-canonical form with exact Schmidt values on every bond.

        A right-to-left sweep makes every tensor right-orthonormal, then a
        left-to-right sweep of SVDs reads off the Schmidt values
        :math:`S_{j}` and sets :math:`B_j = S_j^{-1} A_j S_{j+1}` (Vidal's
        relation). Non-unitary gates, as in imaginary-time evolution,
        break the canonical form that :meth:`bond_energies` and the TEBD
        truncation rely on.
        """
        B = [b.copy() for b in self.B]
        for j in range(self.N - 1, 0, -1):
            vL, d, vR = B[j].shape
            U, Y, V = np.linalg.svd(B[j].reshape(vL, d * vR), full_matrices=False)
            B[j] = V.reshape(-1, d, vR)
            B[j - 1] = np.tensordot(B[j - 1], U * Y[None, :], axes=(2, 0))
        B[0] = B[0] / np.linalg.norm(B[0])
        # left-to-right: the orthogonality centre moves along as theta = Y V B_{j+1}
        S = [np.ones(1)]
        for j in range(self.N - 1):
            vL, d, vR = B[j].shape
            U, Y, V = np.linalg.svd(B[j].reshape(vL * d, vR), full_matrices=False)
            keep = max(1, int(np.sum(Y > 1e-14 * Y[0])))
            U, Y, V = U[:, :keep], Y[:keep] / np.linalg.norm(Y[:keep]), V[:keep, :]
            B[j] = np.tensordot(np.diag(1.0 / S[j]), U.reshape(vL, d, keep), axes=(1, 0)) * Y[None, None, :]
            B[j + 1] = np.tensordot(Y[:, None] * V, B[j + 1], axes=(1, 0))
            S.append(Y)
        B[-1] = np.tensordot(np.diag(1.0 / S[-1]), B[-1], axes=(1, 0))
        self.B = B
        self.S = S

    def apply_two_site(self, j: int, U: NDArray, chi_max: int, eps: float = 1e-12) -> float:
        """Apply a :math:`4\\times4` gate to sites ``j, j+1`` and truncate; returns the discarded weight."""
        th = self._theta(j)
        vL, vR = th.shape[0], th.shape[3]
        th = np.tensordot(U.reshape(2, 2, 2, 2), th, axes=([2, 3], [1, 2]))  # (p1, p2, vL, vR)
        th = th.transpose(2, 0, 1, 3).reshape(vL * 2, 2 * vR)
        X, Y, Z = np.linalg.svd(th, full_matrices=False)
        keep = min(chi_max, int(np.sum(Y > eps * Y[0])))
        discarded = float(np.sum(Y[keep:] ** 2) / np.sum(Y**2))
        X, Y, Z = X[:, :keep], Y[:keep], Z[:keep, :]
        Y = Y / np.linalg.norm(Y)
        A = X.reshape(vL, 2, keep)
        # B_j = S_j^{-1} A diag(Y), which is right-canonical
        self.B[j] = np.tensordot(np.diag(1.0 / self.S[j]), A, axes=(1, 0)) * Y[None, None, :]
        self.S[j + 1] = Y
        self.B[j + 1] = Z.reshape(keep, 2, vR)
        return discarded


def tebd(
    psi: MPS,
    H_bonds: list[NDArray],
    dt: float,
    n_steps: int,
    chi_max: int = 64,
    imaginary: bool = False,
    eps: float = 1e-12,
    measure_every: int = 1,
    observables: dict | None = None,
) -> dict[str, NDArray]:
    """Evolve an MPS with second-order TEBD.

    Parameters
    ----------
    psi : MPS
        Modified in place.
    H_bonds : list of ndarray
        Two-site Hamiltonian terms, e.g. from :func:`xxz_bond_hamiltonians`.
    dt : float
        Time step (imaginary-time step if ``imaginary``).
    n_steps : int
    chi_max : int, default 64
        Maximum bond dimension.
    imaginary : bool, default False
        Imaginary-time evolution :math:`e^{-H\\tau}` toward the ground state.
    eps : float, default 1e-12
        Relative singular-value cutoff.
    measure_every : int, default 1
    observables : dict of str -> ndarray, optional
        Single-site operators whose site-resolved expectation values are recorded.

    Returns
    -------
    dict
        ``"t"``, the total ``"energy"``, the half-chain ``"entropy"``, the
        accumulated ``"discarded_weight"``, and one array per observable.

    Examples
    --------
    Imaginary-time TEBD finds the ground state of an open 8-site Heisenberg chain:

    >>> from physicskit.condensed.spin_chains import lowest_eigenstates, xxz_hamiltonian
    >>> H = xxz_bond_hamiltonians(8)
    >>> psi = MPS.product_state([0, 1] * 4)
    >>> out = tebd(psi, H, dt=0.05, n_steps=400, chi_max=16, imaginary=True)
    >>> E0 = lowest_eigenstates(xxz_hamiltonian(8, periodic=False, n_up=4)[0], k=1)[0][0]
    >>> bool(abs(out["energy"][-1] - E0) < 1e-6)
    True
    """
    observables = observables or {}
    factor = -1.0 if imaginary else -1j
    half = [expm(factor * h * dt / 2) for h in H_bonds]
    full = [expm(factor * h * dt) for h in H_bonds]
    even = range(0, psi.N - 1, 2)
    odd = range(1, psi.N - 1, 2)
    t, energy, entropy, discarded = [0.0], [psi.bond_energies(H_bonds).sum()], [psi.entanglement_entropy()[psi.N // 2 - 1]], [0.0]
    obs = {k: [psi.expectation_local(op)] for k, op in observables.items()}
    total = 0.0
    for step in range(1, n_steps + 1):
        for j in even:
            total += psi.apply_two_site(j, half[j], chi_max, eps)
        for j in odd:
            total += psi.apply_two_site(j, full[j], chi_max, eps)
        for j in even:
            total += psi.apply_two_site(j, half[j], chi_max, eps)
        if imaginary:
            psi.canonicalize()
        if step % measure_every == 0:
            t.append(step * dt)
            energy.append(psi.bond_energies(H_bonds).sum())
            entropy.append(psi.entanglement_entropy()[psi.N // 2 - 1])
            discarded.append(total)
            for k, op in observables.items():
                obs[k].append(psi.expectation_local(op))
    out = {"t": np.array(t), "energy": np.array(energy), "entropy": np.array(entropy), "discarded_weight": np.array(discarded)}
    out.update({k: np.array(v) for k, v in obs.items()})
    return out
