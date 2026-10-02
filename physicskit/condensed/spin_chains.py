r"""Exact diagonalization of spin-1/2 chains: Heisenberg XXZ and transverse-field Ising.

Sparse Hamiltonians are built directly in a symmetry sector -- fixed
magnetization for the :math:`U(1)`-symmetric XXZ chain, fixed
:math:`\mathbb{Z}_2` spin-flip parity for the transverse-field Ising model
(TFIM) -- and diagonalized with Lanczos
(:func:`scipy.sparse.linalg.eigsh`). Ground-state energies, gaps, and
bipartite entanglement entropies are checked against the two classic exact
solutions: Bethe's 1931 ansatz for the Heisenberg chain and Pfeuty's 1970
free-fermion solution of the TFIM.

Conventions: energies in units of the exchange coupling. A basis state is
an integer whose binary digits are the spins, site 0 being the *most*
significant bit, and bit 1 meaning spin up (:math:`\sigma^z = +1`). A
state vector's entries follow the sorted order of its sector's basis
states (for the full space, simply ``0 .. 2**N - 1``), so the first
:math:`\ell` sites are the leading tensor factor of the
:math:`2^{\ell} \times 2^{N-\ell}` bipartition used by
:func:`entanglement_entropy`.

- :func:`spin_chain_basis`, :func:`xxz_hamiltonian`,
  :func:`tfim_hamiltonian` -- sparse Hamiltonians in symmetry sectors.
- :func:`lowest_eigenstates`, :func:`energy_gap` -- Lanczos
  diagonalization.
- :func:`embed_state`, :func:`entanglement_entropy`,
  :func:`entanglement_profile`, :func:`calabrese_cardy_entropy`,
  :func:`fit_central_charge` -- bipartite entanglement.
- :func:`bethe_ansatz_xxx_ground_energy`, :func:`tfim_free_fermion_spectrum`,
  :func:`tfim_ground_state_energy` -- exact reference results.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import scipy.sparse as sp
from scipy.optimize import fsolve
from scipy.sparse.linalg import eigsh

__all__ = [
    "spin_chain_basis",
    "xxz_hamiltonian",
    "tfim_hamiltonian",
    "lowest_eigenstates",
    "energy_gap",
    "embed_state",
    "entanglement_entropy",
    "entanglement_profile",
    "calabrese_cardy_entropy",
    "fit_central_charge",
    "bethe_ansatz_xxx_ground_energy",
    "tfim_free_fermion_spectrum",
    "tfim_ground_state_energy",
]

_DENSE_LIMIT = 400


def _popcount(states: np.ndarray, N: int) -> np.ndarray:
    return ((states[:, None] >> np.arange(N)) & 1).sum(axis=1)


def _bonds(N: int, periodic: bool) -> list[tuple[int, int]]:
    bonds = [(i, i + 1) for i in range(N - 1)]
    if periodic and N > 2:
        bonds.append((N - 1, 0))
    return bonds


def _bit(i: int, N: int) -> int:
    return 1 << (N - 1 - i)


def spin_chain_basis(N: int, n_up: int | None = None, parity: int | None = None) -> np.ndarray:
    r"""Sorted computational basis states of an :math:`N`-site spin-1/2 chain, optionally restricted to a sector.

    Parameters
    ----------
    N : int
        Number of sites.
    n_up : int, optional
        Keep only states with this many up spins, i.e. total
        :math:`S^z = n_\uparrow - N/2`.
    parity : {+1, -1}, optional
        Keep only states with this eigenvalue of
        :math:`P = \prod_i \sigma^z_i = (-1)^{n_\downarrow}`.

    Returns
    -------
    numpy.ndarray of int64
        Sorted basis states.

    Examples
    --------
    >>> spin_chain_basis(4, n_up=2)
    array([ 3,  5,  6,  9, 10, 12])
    >>> len(spin_chain_basis(6, parity=+1))
    32
    """
    states = np.arange(2**N, dtype=np.int64)
    ups = _popcount(states, N)
    keep = np.ones(states.shape, dtype=bool)
    if n_up is not None:
        keep &= ups == n_up
    if parity is not None:
        keep &= (-1) ** (N - ups) == parity
    return states[keep]


def xxz_hamiltonian(
    N: int,
    J: float = 1.0,
    delta: float = 1.0,
    h: float = 0.0,
    periodic: bool = True,
    n_up: int | None = None,
) -> tuple[sp.csr_matrix, np.ndarray]:
    r"""Sparse spin-1/2 Heisenberg XXZ chain Hamiltonian.

    .. math::

        H = J \sum_{\langle ij \rangle} \left( S^x_i S^x_j + S^y_i S^y_j
            + \Delta\, S^z_i S^z_j \right) - h \sum_i S^z_i,
        \qquad \mathbf{S} = \tfrac12 \boldsymbol{\sigma}.

    The flip-flop term :math:`S^x S^x + S^y S^y = \tfrac12(S^+S^- +
    S^-S^+)` conserves total :math:`S^z`, so passing ``n_up`` builds the
    Hamiltonian directly in that magnetization sector. :math:`\Delta = 1`
    is the isotropic Heisenberg (XXX) antiferromagnet solved by Bethe
    (Z. Phys. 71, 205 (1931)).

    Parameters
    ----------
    N : int
        Number of sites.
    J : float, default=1.0
        Exchange coupling (:math:`J > 0` antiferromagnetic).
    delta : float, default=1.0
        Ising anisotropy :math:`\Delta`.
    h : float, default=0.0
        Longitudinal magnetic field.
    periodic : bool, default=True
        Periodic (ring) or open boundary conditions.
    n_up : int, optional
        Number of up spins of the sector (``None`` for the full space).

    Returns
    -------
    H : scipy.sparse.csr_matrix
        The Hamiltonian in the sector basis.
    basis : numpy.ndarray
        The sector's basis states, from :func:`spin_chain_basis`.

    Examples
    --------
    The two-site singlet has energy :math:`-3J/4`:

    >>> import numpy as np
    >>> H, basis = xxz_hamiltonian(2, periodic=False, n_up=1)
    >>> np.linalg.eigvalsh(H.toarray())
    array([-0.75,  0.25])
    """
    basis = spin_chain_basis(N, n_up=n_up)
    dim = len(basis)
    diag = np.zeros(dim)
    rows, cols, vals = [], [], []
    for i, j in _bonds(N, periodic):
        bi, bj = _bit(i, N), _bit(j, N)
        si = np.where(basis & bi, 0.5, -0.5)
        sj = np.where(basis & bj, 0.5, -0.5)
        diag += J * delta * si * sj
        flip = np.nonzero(si != sj)[0]
        target = basis[flip] ^ (bi | bj)
        rows.append(np.searchsorted(basis, target))
        cols.append(flip)
        vals.append(np.full(flip.size, 0.5 * J))
    for i in range(N):
        diag -= h * np.where(basis & _bit(i, N), 0.5, -0.5)
    rows.append(np.arange(dim))
    cols.append(np.arange(dim))
    vals.append(diag)
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim))
    return H, basis


def tfim_hamiltonian(
    N: int,
    J: float = 1.0,
    h: float = 1.0,
    periodic: bool = True,
    parity: int | None = None,
) -> tuple[sp.csr_matrix, np.ndarray]:
    r"""Sparse transverse-field Ising chain Hamiltonian.

    .. math::

        H = -J \sum_{\langle ij \rangle} \sigma^x_i \sigma^x_j - h \sum_i \sigma^z_i

    (Pfeuty, Ann. Phys. 57, 79 (1970), in Pauli matrices). The
    Ising coupling acts along :math:`x` so that the conserved
    :math:`\mathbb{Z}_2` parity :math:`P = \prod_i \sigma^z_i` is diagonal
    in the computational basis; ``parity`` selects its sector. The chain is
    critical at :math:`h = J`.

    Parameters
    ----------
    N : int
        Number of sites.
    J : float, default=1.0
        Ising coupling.
    h : float, default=1.0
        Transverse field.
    periodic : bool, default=True
        Periodic (ring) or open boundary conditions.
    parity : {+1, -1}, optional
        Parity sector (``None`` for the full space).

    Returns
    -------
    H : scipy.sparse.csr_matrix
        The Hamiltonian in the sector basis.
    basis : numpy.ndarray
        The sector's basis states.

    Examples
    --------
    At zero field the ground state is the doubly degenerate ferromagnet
    with energy :math:`-NJ` on a ring:

    >>> import numpy as np
    >>> H, _ = tfim_hamiltonian(4, J=1.0, h=0.0)
    >>> np.round(np.linalg.eigvalsh(H.toarray())[:2], 12)
    array([-4., -4.])
    """
    basis = spin_chain_basis(N, parity=parity)
    dim = len(basis)
    diag = np.zeros(dim)
    for i in range(N):
        diag -= h * np.where(basis & _bit(i, N), 1.0, -1.0)
    rows, cols, vals = [np.arange(dim)], [np.arange(dim)], [diag]
    for i, j in _bonds(N, periodic):
        target = basis ^ (_bit(i, N) | _bit(j, N))
        rows.append(np.searchsorted(basis, target))
        cols.append(np.arange(dim))
        vals.append(np.full(dim, -J))
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim))
    return H, basis


def lowest_eigenstates(H: sp.spmatrix | np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    r"""The :math:`k` lowest eigenpairs of a Hermitian (sparse) Hamiltonian.

    Uses dense :func:`numpy.linalg.eigh` for small matrices and the
    Lanczos method (:func:`scipy.sparse.linalg.eigsh`, ``which="SA"``)
    otherwise.

    Parameters
    ----------
    H : scipy.sparse matrix or numpy.ndarray
        Hamiltonian, shape ``(dim, dim)``.
    k : int, default=2
        Number of eigenpairs.

    Returns
    -------
    energies : numpy.ndarray
        Ascending, shape ``(k,)``.
    vectors : numpy.ndarray
        Eigenvectors as columns, shape ``(dim, k)``.

    Examples
    --------
    >>> H, _ = xxz_hamiltonian(4, n_up=2)
    >>> E, _ = lowest_eigenstates(H, k=1)
    >>> round(float(E[0]), 12)
    -2.0
    """
    dim = H.shape[0]
    k = min(k, dim)
    if dim <= _DENSE_LIMIT or k >= dim - 1:
        dense = np.asarray(H) if isinstance(H, np.ndarray) else H.toarray()
        E, V = np.linalg.eigh(dense)
        return E[:k], V[:, :k]
    v0 = np.random.default_rng(0).normal(size=dim)
    E, V = eigsh(H, k=k, which="SA", v0=v0, tol=1e-12)
    order = np.argsort(E)
    return E[order], V[:, order]


def energy_gap(H: sp.spmatrix | np.ndarray) -> float:
    r"""Gap :math:`E_1 - E_0` between the two lowest eigenvalues of ``H``.

    Parameters
    ----------
    H : scipy.sparse matrix or numpy.ndarray
        Hamiltonian (typically restricted to one symmetry sector).

    Returns
    -------
    float

    Examples
    --------
    The critical TFIM's gap in the even sector is the two-quasiparticle
    energy :math:`\varepsilon_{\pi/N} + \varepsilon_{-\pi/N}`:

    >>> import numpy as np
    >>> H, _ = tfim_hamiltonian(8, h=1.0, parity=+1)
    >>> bool(np.isclose(energy_gap(H), 8 * np.sin(np.pi / 16)))
    True
    """
    E, _ = lowest_eigenstates(H, k=2)
    return float(E[1] - E[0])


def embed_state(psi: np.ndarray, basis: np.ndarray, N: int) -> np.ndarray:
    r"""Embed a sector-basis state vector into the full :math:`2^N`-dimensional space.

    Parameters
    ----------
    psi : numpy.ndarray
        Amplitudes on ``basis``.
    basis : numpy.ndarray
        Sector basis states.
    N : int
        Number of sites.

    Returns
    -------
    numpy.ndarray
        Shape ``(2**N,)``.

    Examples
    --------
    >>> import numpy as np
    >>> embed_state(np.array([1.0, 0.0]), np.array([1, 2]), 2)
    array([0., 1., 0., 0.])
    """
    full = np.zeros(2**N, dtype=np.result_type(psi, float))
    full[basis] = psi
    return full


def entanglement_entropy(psi: np.ndarray, N: int, n_A: int, basis: np.ndarray | None = None) -> float:
    r"""Von Neumann entanglement entropy of the first ``n_A`` sites.

    .. math::

        S_A = -\operatorname{tr}\rho_A \ln \rho_A = -\sum_\alpha s_\alpha^2 \ln s_\alpha^2,

    with :math:`s_\alpha` the Schmidt coefficients -- the singular values of
    :math:`\psi` reshaped to a :math:`2^{n_A} \times 2^{N - n_A}` matrix
    (Nielsen and Chuang, Sec. 2.5).

    Parameters
    ----------
    psi : numpy.ndarray
        Normalized state, on the full space or on ``basis``.
    N : int
        Number of sites.
    n_A : int
        Number of leading sites in subsystem :math:`A`.
    basis : numpy.ndarray, optional
        Sector basis of ``psi``, if it is not a full-space vector.

    Returns
    -------
    float
        :math:`S_A` in nats.

    Examples
    --------
    A singlet carries :math:`\ln 2` of entanglement:

    >>> import numpy as np
    >>> singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
    >>> bool(np.isclose(entanglement_entropy(singlet, 2, 1), np.log(2)))
    True
    """
    full = embed_state(psi, basis, N) if basis is not None else np.asarray(psi)
    s = np.linalg.svd(full.reshape(2**n_A, 2 ** (N - n_A)), compute_uv=False)
    p = s[s > 1e-14] ** 2
    return float(-(p * np.log(p)).sum()) + 0.0  # + 0.0 turns -0.0 into 0.0


def entanglement_profile(psi: np.ndarray, N: int, basis: np.ndarray | None = None) -> np.ndarray:
    r"""Entanglement entropy :math:`S(\ell)` for every cut :math:`\ell = 1, \dots, N-1`.

    Parameters
    ----------
    psi : numpy.ndarray
        Normalized state.
    N : int
        Number of sites.
    basis : numpy.ndarray, optional
        Sector basis of ``psi``.

    Returns
    -------
    numpy.ndarray
        Shape ``(N - 1,)``.

    Examples
    --------
    A product state is unentangled across every cut:

    >>> import numpy as np
    >>> psi = np.zeros(8); psi[0] = 1.0
    >>> entanglement_profile(psi, 3)
    array([0., 0.])
    """
    full = embed_state(psi, basis, N) if basis is not None else np.asarray(psi)
    return np.array([entanglement_entropy(full, N, ell) for ell in range(1, N)])


def calabrese_cardy_entropy(ell: np.ndarray, N: int, c: float, const: float = 0.0, periodic: bool = True) -> np.ndarray:
    r"""Calabrese-Cardy entanglement entropy of an interval in a critical (CFT) chain.

    .. math::

        S(\ell) = \frac{c}{3\eta}\,\ln\!\left[\frac{\eta N}{\pi}
            \sin\frac{\pi\ell}{N}\right] + c',

    with :math:`\eta = 1` for a periodic chain and :math:`\eta = 2` for an
    open chain cut at distance :math:`\ell` from an edge (Calabrese and
    Cardy, J. Stat. Mech. P06002 (2004), Eqs. 3 and 14). A gapped chain
    instead obeys an area law: :math:`S(\ell)` saturates.

    Parameters
    ----------
    ell : numpy.ndarray
        Subsystem sizes.
    N : int
        Chain length.
    c : float
        Central charge (:math:`1/2` for the critical Ising chain, :math:`1`
        for the Heisenberg chain).
    const : float, default=0.0
        Non-universal constant :math:`c'`.
    periodic : bool, default=True
        Boundary conditions.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> import numpy as np
    >>> S = calabrese_cardy_entropy(np.array([4, 8]), 16, c=3.0)
    >>> bool(np.isclose(S[1], np.log(16 / np.pi)))
    True
    """
    eta = 1.0 if periodic else 2.0
    ell = np.asarray(ell, dtype=float)
    return c / (3 * eta) * np.log(eta * N / np.pi * np.sin(np.pi * ell / N)) + const


def fit_central_charge(profile: np.ndarray, N: int, periodic: bool = True, trim: int = 1) -> tuple[float, float]:
    r"""Least-squares fit of :func:`calabrese_cardy_entropy` to an entanglement profile.

    Parameters
    ----------
    profile : numpy.ndarray
        :math:`S(\ell)` for :math:`\ell = 1, \dots, N-1`
        (from :func:`entanglement_profile`).
    N : int
        Chain length.
    periodic : bool, default=True
        Boundary conditions.
    trim : int, default=1
        Number of cuts dropped at each end, where lattice corrections are
        largest.

    Returns
    -------
    c : float
        Fitted central charge.
    const : float
        Fitted non-universal constant.

    Examples
    --------
    >>> import numpy as np
    >>> ell = np.arange(1, 20)
    >>> c, _ = fit_central_charge(calabrese_cardy_entropy(ell, 20, c=0.5, const=0.3), 20)
    >>> round(c, 10)
    0.5
    """
    ell = np.arange(1, N)[trim : N - 1 - trim]
    S = np.asarray(profile)[trim : N - 1 - trim]
    x = calabrese_cardy_entropy(ell, N, c=1.0, periodic=periodic)
    A = np.column_stack([x, np.ones_like(x)])
    (c, const), *_ = np.linalg.lstsq(A, S, rcond=None)
    return float(c), float(const)


def bethe_ansatz_xxx_ground_energy(N: int, J: float = 1.0) -> float:
    r"""Ground-state energy of the periodic spin-1/2 Heisenberg antiferromagnet from the Bethe ansatz.

    With :math:`M = N/2` magnons of rapidities :math:`\lambda_j`, the
    logarithmic Bethe equations for :math:`H = J\sum_i \mathbf{S}_i\cdot
    \mathbf{S}_{i+1}` read

    .. math::

        2N \arctan(2\lambda_j) = 2\pi I_j
            + 2\sum_{k \ne j} \arctan(\lambda_j - \lambda_k),

    with the ground state given by the consecutive quantum numbers
    :math:`I_j = -\tfrac{M-1}{2}, \dots, \tfrac{M-1}{2}`, and

    .. math::

        E_0 = \frac{JN}{4} - \frac{J}{2}\sum_{j=1}^{M} \frac{1}{\lambda_j^2 + 1/4}

    (Bethe 1931; Karbach and Muller, Computers in Physics 11, 36 (1997),
    Eqs. 20-24). As :math:`N \to \infty`, :math:`E_0/N \to J(1/4 - \ln 2)`
    (Hulthen 1938).

    Parameters
    ----------
    N : int
        Even number of sites, :math:`N \ge 4`.
    J : float, default=1.0
        Exchange coupling.

    Returns
    -------
    float

    Examples
    --------
    >>> round(bethe_ansatz_xxx_ground_energy(4), 12)
    -2.0
    """
    if N % 2 or N < 4:
        raise ValueError("N must be even and >= 4")
    M = N // 2
    I = np.arange(M) - (M - 1) / 2.0

    def residual(lam):
        diff = lam[:, None] - lam[None, :]
        return N * np.arctan(2 * lam) - np.pi * I - np.arctan(diff).sum(axis=1)

    def jacobian(lam):
        kernel = 1.0 / (1.0 + (lam[:, None] - lam[None, :]) ** 2)
        jac = kernel.copy()
        np.fill_diagonal(jac, 0.0)
        jac[np.diag_indices(M)] = 2 * N / (1 + 4 * lam**2) - jac.sum(axis=1)
        return jac

    lam = 0.5 * np.tan(np.pi * I / N)  # non-interacting initial guess
    lam = fsolve(residual, lam, fprime=jacobian, xtol=1e-12)
    return float(J * N / 4 - 0.5 * J * np.sum(1.0 / (lam**2 + 0.25)))


def _tfim_modes(N: int, J: float, h: float, sector: int) -> tuple[np.ndarray, np.ndarray]:
    """Paired (positive) and unpaired (signed) single-fermion energies of one parity sector."""
    k = np.pi * (2 * np.arange(N) + 1) / N if sector == 1 else 2 * np.pi * np.arange(N) / N
    k = np.mod(k, 2 * np.pi)
    unpaired = np.isclose(k, 0.0) | np.isclose(k, np.pi)
    paired_eps = 2 * np.sqrt(J**2 + h**2 - 2 * J * h * np.cos(k[~unpaired]))
    unpaired_eps = 2 * (h - J * np.cos(k[unpaired]))
    return paired_eps, unpaired_eps


def tfim_free_fermion_spectrum(N: int, J: float = 1.0, h: float = 1.0, parity: int | None = None) -> np.ndarray:
    r"""Exact spectrum of the periodic TFIM from its Jordan-Wigner free-fermion solution.

    The Jordan-Wigner transformation maps :func:`tfim_hamiltonian` onto
    free fermions, :math:`H = \sum_k \varepsilon_k (n_k - \tfrac12)` with

    .. math::

        \varepsilon_k = 2\sqrt{J^2 + h^2 - 2Jh\cos k}

    (Pfeuty 1970, Eq. 2.15; Lieb, Schultz and Mattis, Ann. Phys. 16, 407
    (1961)). The parity :math:`P = +1` sector has antiperiodic fermions,
    :math:`k = \pm\pi(2n+1)/N`, and an even number of excitations;
    :math:`P = -1` has periodic fermions, :math:`k = 2\pi n/N`, and an odd
    total fermion number. The unpaired modes :math:`k = 0, \pi` carry the
    signed energy :math:`2(h - J\cos k)` and count as bare fermions.

    Parameters
    ----------
    N : int
        Number of sites (keep :math:`N \lesssim 16`: all :math:`2^N`
        levels are enumerated).
    J : float, default=1.0
        Ising coupling.
    h : float, default=1.0
        Transverse field.
    parity : {+1, -1}, optional
        Restrict to one sector (``None`` for both).

    Returns
    -------
    numpy.ndarray
        Sorted energies.

    Examples
    --------
    >>> import numpy as np
    >>> H, _ = tfim_hamiltonian(5, J=1.0, h=0.6)
    >>> bool(np.allclose(tfim_free_fermion_spectrum(5, 1.0, 0.6), np.linalg.eigvalsh(H.toarray())))
    True
    """
    sectors = (1, -1) if parity is None else (parity,)
    energies = []
    for sector in sectors:
        paired, unpaired = _tfim_modes(N, J, h, sector)
        eps = np.concatenate([paired, unpaired])
        occ = np.array(list(product((0, 1), repeat=eps.size)))
        keep = occ.sum(axis=1) % 2 == (0 if sector == 1 else 1)
        energies.append((occ[keep] - 0.5) @ eps)
    return np.sort(np.concatenate(energies))


def tfim_ground_state_energy(N: int, J: float = 1.0, h: float = 1.0) -> float:
    r"""Exact ground-state energy of the periodic TFIM, the even-sector Bogoliubov vacuum.

    .. math::

        E_0 = -\frac12 \sum_{k = \pm\pi(2n+1)/N} \varepsilon_k,

    which tends to :math:`-\tfrac{N}{2\pi}\int_{-\pi}^{\pi}\varepsilon_k\,dk/2`
    as :math:`N \to \infty` (Pfeuty 1970, Eq. 3.2).

    Parameters
    ----------
    N : int
        Number of sites.
    J : float, default=1.0
        Ising coupling.
    h : float, default=1.0
        Transverse field.

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> bool(np.isclose(tfim_ground_state_energy(8, 1.0, 1.0), -2 / np.sin(np.pi / 16)))
    True
    """
    paired, unpaired = _tfim_modes(N, J, h, 1)
    return float(-0.5 * paired.sum() - 0.5 * np.abs(unpaired).sum())
