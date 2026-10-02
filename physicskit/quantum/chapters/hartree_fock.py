r"""Restricted (closed-shell) Hartree-Fock for small atoms and molecules in s-type Gaussian bases.

Atomic units throughout (:math:`\hbar = m_e = e = 4\pi\varepsilon_0 = 1`;
energies in hartree, lengths in bohr), matching the :math:`\hbar = 1`
convention of :mod:`physicskit.quantum`.

The Roothaan-Hall equations :math:`FC = SC\varepsilon` (Roothaan, Rev.
Mod. Phys. 23, 69 (1951); Hall, Proc. R. Soc. A 205, 541 (1951)) are
solved self-consistently for :math:`N` electrons in :math:`N/2` doubly
occupied spatial orbitals, following Szabo and Ostlund, *Modern Quantum
Chemistry* (1989), Ch. 3. Basis functions are contracted s-type Gaussians,
enough for the classic minimal-basis examples (:math:`\mathrm H_2`,
:math:`\mathrm{HeH^+}`, He, :math:`\mathrm H_3^+`) and for large
even-tempered s bases that converge to the Hartree-Fock limit of
two-electron atoms. All integrals are in closed form (Szabo and Ostlund,
Appendix A).

- :class:`GaussianS`, :func:`sto3g_1s`, :func:`even_tempered_s_basis` --
  basis functions.
- :func:`overlap_matrix`, :func:`kinetic_matrix`, :func:`nuclear_attraction_matrix`,
  :func:`electron_repulsion_tensor` -- one- and two-electron integrals.
- :func:`restricted_hartree_fock`, :class:`HartreeFockResult` -- the SCF
  procedure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import erf

__all__ = [
    "GaussianS",
    "sto3g_1s",
    "even_tempered_s_basis",
    "overlap_matrix",
    "kinetic_matrix",
    "nuclear_attraction_matrix",
    "electron_repulsion_tensor",
    "HartreeFockResult",
    "restricted_hartree_fock",
    "STO3G_ZETA",
]

# STO-3G fit of a zeta = 1 Slater 1s function (Hehre, Stewart and Pople,
# J. Chem. Phys. 51, 2657 (1969); Szabo and Ostlund Eq. 3.225).
_STO3G_ALPHA = np.array([0.109818, 0.405771, 2.22766])
_STO3G_D = np.array([0.444635, 0.535328, 0.154329])

STO3G_ZETA = {"H": 1.24, "He": 1.69}
"""Standard molecular Slater exponents for the STO-3G 1s functions of H and He."""


@dataclass
class GaussianS:
    r"""A contracted s-type Gaussian :math:`\phi(\mathbf r) = \sum_p d_p\,(2\alpha_p/\pi)^{3/4} e^{-\alpha_p|\mathbf r - \mathbf A|^2}`.

    Parameters
    ----------
    exponents : array_like
        Primitive exponents :math:`\alpha_p`.
    coefficients : array_like
        Contraction coefficients :math:`d_p` of the *normalized* primitives.
    center : array_like
        Center :math:`\mathbf A` (bohr), a 3-vector.
    """

    exponents: np.ndarray
    coefficients: np.ndarray
    center: np.ndarray

    def __post_init__(self) -> None:
        self.exponents = np.atleast_1d(np.asarray(self.exponents, dtype=float))
        self.coefficients = np.atleast_1d(np.asarray(self.coefficients, dtype=float))
        self.center = np.asarray(self.center, dtype=float).reshape(3)

    @property
    def weights(self) -> np.ndarray:
        """Coefficients times primitive normalization constants."""
        return self.coefficients * (2 * self.exponents / np.pi) ** 0.75


def sto3g_1s(zeta: float, center: Sequence[float]) -> GaussianS:
    r"""STO-3G contraction of a Slater 1s orbital with exponent :math:`\zeta`.

    The :math:`\zeta = 1` exponents are scaled as :math:`\alpha = \zeta^2\alpha_{(1)}`
    (Szabo and Ostlund, Eq. 3.224).

    Parameters
    ----------
    zeta : float
        Slater exponent (see :data:`STO3G_ZETA`).
    center : sequence of float
        Position (bohr).

    Returns
    -------
    GaussianS

    Examples
    --------
    >>> import numpy as np
    >>> phi = sto3g_1s(1.24, [0.0, 0.0, 0.0])
    >>> round(float(overlap_matrix([phi])[0, 0]), 5)
    1.0
    """
    return GaussianS(_STO3G_ALPHA * zeta**2, _STO3G_D, np.asarray(center, dtype=float))


def even_tempered_s_basis(center: Sequence[float], n: int, alpha_min: float, ratio: float) -> list[GaussianS]:
    r"""Uncontracted even-tempered s basis, :math:`\alpha_k = \alpha_{\min}\,\beta^{k}`, :math:`k = 0,\dots,n-1`.

    Even-tempered sequences (Ruedenberg, Raffenetti and Bardo, 1973)
    approach the complete s basis as :math:`n` grows, and with it the
    Hartree-Fock limit of a two-electron atom.

    Parameters
    ----------
    center : sequence of float
        Center (bohr).
    n : int
        Number of functions.
    alpha_min : float
        Smallest exponent.
    ratio : float
        Geometric ratio :math:`\beta > 1`.

    Returns
    -------
    list of GaussianS

    Examples
    --------
    >>> len(even_tempered_s_basis([0, 0, 0], 8, 0.1, 3.0))
    8
    """
    return [GaussianS(np.array([alpha_min * ratio**k]), np.array([1.0]), np.asarray(center, dtype=float)) for k in range(n)]


def _boys0(t: np.ndarray) -> np.ndarray:
    r"""Boys function :math:`F_0(t) = \tfrac12\sqrt{\pi/t}\,\mathrm{erf}\sqrt t`, with :math:`F_0(0) = 1`."""
    t = np.asarray(t, dtype=float)
    small = t < 1e-10
    ts = np.where(small, 1.0, t)
    return np.where(small, 1.0 - t / 3.0, 0.5 * np.sqrt(np.pi / ts) * erf(np.sqrt(ts)))


def _pair(f: GaussianS, g: GaussianS):
    """Primitive-pair quantities: exponent sums, Gaussian-product centers, prefactors."""
    a = f.exponents[:, None]
    b = g.exponents[None, :]
    p = a + b
    ab2 = np.sum((f.center - g.center) ** 2)
    K = np.exp(-a * b / p * ab2)
    P = (a[..., None] * f.center + b[..., None] * g.center) / p[..., None]
    w = f.weights[:, None] * g.weights[None, :]
    return a, b, p, ab2, K, P, w


def overlap_matrix(basis: Sequence[GaussianS]) -> np.ndarray:
    r"""Overlap matrix :math:`S_{\mu\nu} = \langle\phi_\mu|\phi_\nu\rangle`, with :math:`(\pi/p)^{3/2}e^{-ab|A-B|^2/p}` per primitive pair.

    Parameters
    ----------
    basis : sequence of GaussianS

    Returns
    -------
    numpy.ndarray
    """
    n = len(basis)
    S = np.empty((n, n))
    for i, f in enumerate(basis):
        for j, g in enumerate(basis):
            _, _, p, _, K, _, w = _pair(f, g)
            S[i, j] = np.sum(w * (np.pi / p) ** 1.5 * K)
    return S


def kinetic_matrix(basis: Sequence[GaussianS]) -> np.ndarray:
    r"""Kinetic-energy matrix :math:`T_{\mu\nu} = \langle\phi_\mu|-\tfrac12\nabla^2|\phi_\nu\rangle`.

    Per primitive pair :math:`\frac{ab}{p}\left(3 - \frac{2ab}{p}|A-B|^2\right)(\pi/p)^{3/2}e^{-ab|A-B|^2/p}`
    (Szabo and Ostlund, Eq. A.11).

    Parameters
    ----------
    basis : sequence of GaussianS

    Returns
    -------
    numpy.ndarray
    """
    n = len(basis)
    T = np.empty((n, n))
    for i, f in enumerate(basis):
        for j, g in enumerate(basis):
            a, b, p, ab2, K, _, w = _pair(f, g)
            mu = a * b / p
            T[i, j] = np.sum(w * mu * (3 - 2 * mu * ab2) * (np.pi / p) ** 1.5 * K)
    return T


def nuclear_attraction_matrix(basis: Sequence[GaussianS], charges: Sequence[float] | np.ndarray, positions: np.ndarray) -> np.ndarray:
    r"""Electron-nuclear attraction :math:`V_{\mu\nu} = -\sum_C Z_C\langle\phi_\mu|1/|\mathbf r - \mathbf R_C||\phi_\nu\rangle`.

    Per primitive pair :math:`-\frac{2\pi}{p}Z_C\,e^{-ab|A-B|^2/p}F_0(p|P - C|^2)`
    (Szabo and Ostlund, Eq. A.33).

    Parameters
    ----------
    basis : sequence of GaussianS
    charges : sequence of float
        Nuclear charges :math:`Z_C`.
    positions : numpy.ndarray
        Nuclear positions, shape ``(n_nuclei, 3)``.

    Returns
    -------
    numpy.ndarray
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    n = len(basis)
    V = np.zeros((n, n))
    for i, f in enumerate(basis):
        for j, g in enumerate(basis):
            _, _, p, _, K, P, w = _pair(f, g)
            for Z, C in zip(charges, positions):
                pc2 = np.sum((P - C) ** 2, axis=-1)
                V[i, j] -= Z * np.sum(w * 2 * np.pi / p * K * _boys0(p * pc2))
    return V


def electron_repulsion_tensor(basis: Sequence[GaussianS]) -> np.ndarray:
    r"""Two-electron integrals :math:`(\mu\nu|\lambda\sigma)` in chemists' notation.

    Per primitive quartet

    .. math::

        \frac{2\pi^{5/2}}{pq\sqrt{p+q}}\,K_{AB}K_{CD}\,
        F_0\!\left(\frac{pq}{p+q}|P - Q|^2\right)

    (Szabo and Ostlund, Eq. A.41), filled using the 8-fold permutational
    symmetry.

    Parameters
    ----------
    basis : sequence of GaussianS

    Returns
    -------
    numpy.ndarray
        Shape ``(n, n, n, n)``.
    """
    n = len(basis)
    pairs = {(i, j): _pair(basis[i], basis[j]) for i in range(n) for j in range(i + 1)}
    eri = np.zeros((n, n, n, n))
    keys = sorted(pairs)
    for ij_idx, (i, j) in enumerate(keys):
        _, _, p, _, K1, P, w1 = pairs[(i, j)]
        p_, K1_, P_ = p.ravel(), (K1 * w1).ravel(), P.reshape(-1, 3)
        for k, l in keys[: ij_idx + 1]:
            _, _, q, _, K2, Q, w2 = pairs[(k, l)]
            q_, K2_, Q_ = q.ravel(), (K2 * w2).ravel(), Q.reshape(-1, 3)
            pq = p_[:, None] * q_[None, :]
            s = p_[:, None] + q_[None, :]
            d2 = np.sum((P_[:, None, :] - Q_[None, :, :]) ** 2, axis=-1)
            val = np.sum(2 * np.pi**2.5 / (pq * np.sqrt(s)) * K1_[:, None] * K2_[None, :] * _boys0(pq / s * d2))
            for a, b in ((i, j), (j, i)):
                for c, d in ((k, l), (l, k)):
                    eri[a, b, c, d] = eri[c, d, a, b] = val
    return eri


@dataclass
class HartreeFockResult:
    r"""Output of :func:`restricted_hartree_fock`.

    Attributes
    ----------
    energy : float
        Total energy (electronic plus nuclear repulsion), hartree.
    electronic_energy : float
        :math:`E_0 = \tfrac12\sum_{\mu\nu}P_{\nu\mu}(H^\mathrm{core}_{\mu\nu} + F_{\mu\nu})`.
    nuclear_repulsion : float
        :math:`\sum_{A<B} Z_AZ_B/R_{AB}`.
    orbital_energies : numpy.ndarray
        Canonical orbital energies :math:`\varepsilon_i`, ascending.
    coefficients : numpy.ndarray
        MO coefficients (columns), :math:`C^TSC = I`.
    density : numpy.ndarray
        Closed-shell density matrix :math:`P = 2C_\mathrm{occ}C_\mathrm{occ}^T`.
    n_iterations : int
        SCF iterations used.
    converged : bool
    """

    energy: float
    electronic_energy: float
    nuclear_repulsion: float
    orbital_energies: np.ndarray
    coefficients: np.ndarray
    density: np.ndarray
    n_iterations: int
    converged: bool


def restricted_hartree_fock(
    basis: Sequence[GaussianS],
    charges: Sequence[float] | np.ndarray,
    positions: np.ndarray,
    n_electrons: int,
    max_iter: int = 200,
    tol: float = 1e-10,
    damping: float = 0.0,
) -> HartreeFockResult:
    r"""Closed-shell Hartree-Fock by the Roothaan self-consistent-field procedure.

    With core Hamiltonian :math:`H^\mathrm{core} = T + V`, the Fock matrix

    .. math::

        F_{\mu\nu} = H^\mathrm{core}_{\mu\nu} + \sum_{\lambda\sigma}P_{\lambda\sigma}
            \left[(\mu\nu|\sigma\lambda) - \tfrac12(\mu\lambda|\sigma\nu)\right]

    is diagonalized in the symmetrically orthogonalized basis
    :math:`X = S^{-1/2}`, the :math:`N/2` lowest orbitals are doubly
    occupied, and the density :math:`P = 2C_\mathrm{occ}C_\mathrm{occ}^T` is
    iterated to self-consistency, starting from the core-Hamiltonian guess
    (Szabo and Ostlund, Sec. 3.4.6, Eqs. 3.154 and 3.184). By the variational
    principle the energy is an upper bound, which approaches the
    Hartree-Fock limit from above as the basis grows.

    Parameters
    ----------
    basis : sequence of GaussianS
        Basis functions.
    charges : sequence of float
        Nuclear charges.
    positions : numpy.ndarray
        Nuclear positions (bohr), shape ``(n_nuclei, 3)``.
    n_electrons : int
        Even number of electrons.
    max_iter : int, default=200
        Maximum SCF iterations.
    tol : float, default=1e-10
        Convergence threshold on the density (RMS change) and energy.
    damping : float, default=0.0
        Fraction of the previous density mixed into each update.

    Returns
    -------
    HartreeFockResult

    Raises
    ------
    ValueError
        For an odd electron count or too small a basis.

    Examples
    --------
    :math:`\mathrm H_2` at :math:`R = 1.4` bohr in STO-3G
    (Szabo and Ostlund, Sec. 3.5.2: :math:`E = -1.1167` hartree):

    >>> import numpy as np
    >>> R = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    >>> basis = [sto3g_1s(1.24, r) for r in R]
    >>> res = restricted_hartree_fock(basis, [1, 1], R, n_electrons=2)
    >>> round(res.energy, 4)
    -1.1167
    """
    if n_electrons % 2:
        raise ValueError("restricted closed-shell Hartree-Fock needs an even number of electrons")
    n_occ = n_electrons // 2
    if n_occ > len(basis):
        raise ValueError("basis too small for the number of electrons")
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    charges = np.asarray(charges, dtype=float)

    S = overlap_matrix(basis)
    H = kinetic_matrix(basis) + nuclear_attraction_matrix(basis, charges, positions)
    eri = electron_repulsion_tensor(basis)
    s, U = np.linalg.eigh(S)
    X = U @ np.diag(s**-0.5) @ U.T

    E_nuc = 0.0
    for A in range(len(charges)):
        for B in range(A):
            E_nuc += charges[A] * charges[B] / np.linalg.norm(positions[A] - positions[B])

    def solve(F):
        eps, Cp = np.linalg.eigh(X.T @ F @ X)
        return eps, X @ Cp

    def fock(P):
        return H + np.einsum("ls,mnsl->mn", P, eri) - 0.5 * np.einsum("ls,mlsn->mn", P, eri)

    eps, C = solve(H)
    P = 2 * C[:, :n_occ] @ C[:, :n_occ].T
    E_old = np.inf
    converged = False
    n_iterations = 0
    while n_iterations < max_iter:
        n_iterations += 1
        F = fock(P)
        E_el = 0.5 * np.sum(P * (H + F))
        eps, C = solve(F)
        P_new = (1 - damping) * 2 * C[:, :n_occ] @ C[:, :n_occ].T + damping * P
        dP = np.sqrt(np.mean((P_new - P) ** 2))
        P = P_new
        if dP < tol and abs(E_el - E_old) < tol:
            converged = True
            break
        E_old = E_el
    F = fock(P)
    E_el = 0.5 * np.sum(P * (H + F))
    eps, C = solve(F)
    return HartreeFockResult(
        energy=float(E_el + E_nuc),
        electronic_energy=float(E_el),
        nuclear_repulsion=float(E_nuc),
        orbital_energies=eps,
        coefficients=C,
        density=P,
        n_iterations=n_iterations,
        converged=converged,
    )
