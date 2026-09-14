"""Canonical condensed-matter lattice models: SSH, graphene, Haldane, Kane-Mele, BHZ, Kitaev chain.

All 2D Bloch Hamiltonians in this module are functions of the *reduced*
crystal momentum ``(k1, k2)`` described in :mod:`physicskit.condensed.tight_binding`
(periodic gauge, each component periodic on :math:`[0, 2\\pi)`), which is the
convention expected by :func:`physicskit.condensed.topology.compute_chern_number`.
"""

from __future__ import annotations

import numpy as np

from .tight_binding import Hamiltonian, Lattice

__all__ = [
    "ssh_hamiltonian",
    "ssh_lattice_hamiltonian",
    "graphene_hamiltonian",
    "graphene_lattice_hamiltonian",
    "haldane_model",
    "haldane_lattice_hamiltonian",
    "kane_mele_hamiltonian",
    "bhz_hamiltonian",
    "bhz_ribbon_hamiltonian",
    "kitaev_chain_hamiltonian",
    "kitaev_chain_bdg_real_space",
    "harper_hofstadter_hamiltonian",
]

_SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
_SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_SIGMA_0 = np.eye(2, dtype=complex)


# -- SSH ---------------------------------------------------------------------


def ssh_hamiltonian(k: float, v: float = 1.0, w: float = 1.0) -> np.ndarray:
    """Bloch Hamiltonian of the Su-Schrieffer-Heeger (SSH) dimerized chain.

    Parameters
    ----------
    k : float
        Reduced crystal momentum, periodic on :math:`[0, 2\\pi)`.
    v : float, default=1.0
        Intracell hopping amplitude.
    w : float, default=1.0
        Intercell hopping amplitude.

    Returns
    -------
    ndarray, shape (2, 2)
        Bloch Hamiltonian :math:`H(k) = \\begin{pmatrix}0 & v+we^{-ik}\\\\ v+we^{ik} & 0\\end{pmatrix}`.

    Notes
    -----
    The chain is topologically nontrivial (hosts protected zero-energy edge
    states under open boundary conditions) when :math:`v < w`, and trivial
    when :math:`v > w`. See Also :func:`ssh_lattice_hamiltonian` for the
    real-space builder used to expose those edge states.

    Examples
    --------
    >>> import numpy as np
    >>> H = ssh_hamiltonian(k=0.0, v=0.5, w=1.0)
    >>> np.round(np.linalg.eigvalsh(H), 8)
    array([-1.5,  1.5])
    """
    f = v + w * np.exp(-1j * k)
    return np.array([[0, f], [np.conj(f), 0]], dtype=complex)


def ssh_lattice_hamiltonian(v: float = 1.0, w: float = 1.0) -> Hamiltonian:
    """Build the SSH chain as a :class:`~physicskit.condensed.tight_binding.Hamiltonian`.

    Parameters
    ----------
    v : float, default=1.0
        Intracell hopping amplitude (A-B bond within a unit cell).
    w : float, default=1.0
        Intercell hopping amplitude (B-A bond across the cell boundary).

    Returns
    -------
    Hamiltonian
        A two-orbital-per-cell chain, suitable for
        :func:`~physicskit.condensed.tight_binding.build_ribbon` to expose
        edge zero modes.

    See Also
    --------
    ssh_hamiltonian : Closed-form 2x2 Bloch Hamiltonian equivalent to this builder.

    Examples
    --------
    >>> import numpy as np
    >>> H = ssh_lattice_hamiltonian(v=0.5, w=1.0)
    >>> np.round(H.bands([0.0]), 8)
    array([-1.5,  1.5])
    """
    lat = Lattice(lattice_vectors=[[1.0]], orbitals=[[0.0], [0.5]], labels=["A", "B"])
    H = Hamiltonian(lat)
    H.add_hopping(0, 1, (0,), v)
    H.add_hopping(1, 0, (1,), w)
    return H


# -- Graphene ------------------------------------------------------------


def graphene_lattice_hamiltonian(t: float = 1.0) -> Hamiltonian:
    """Nearest-neighbor honeycomb (graphene) tight-binding Hamiltonian.

    Parameters
    ----------
    t : float, default=1.0
        Nearest-neighbor hopping amplitude.

    Returns
    -------
    Hamiltonian

    See Also
    --------
    graphene_hamiltonian : Closed-form 2x2 Bloch Hamiltonian equivalent to this builder.
    """
    lat = Lattice.honeycomb()
    H = Hamiltonian(lat)
    H.add_hopping(0, 1, (0, 0), t)
    H.add_hopping(0, 1, (-1, 0), t)
    H.add_hopping(0, 1, (0, -1), t)
    return H


def graphene_hamiltonian(k1: float, k2: float, t: float = 1.0) -> np.ndarray:
    """Bloch Hamiltonian of nearest-neighbor graphene (honeycomb lattice).

    Parameters
    ----------
    k1, k2 : float
        Reduced crystal momenta conjugate to the two honeycomb primitive
        vectors, each periodic on :math:`[0, 2\\pi)`.
    t : float, default=1.0
        Nearest-neighbor hopping amplitude.

    Returns
    -------
    ndarray, shape (2, 2)
        Bloch Hamiltonian on the A/B sublattice basis.

    Notes
    -----
    The Dirac points sit at :math:`(k_1, k_2) = (2\\pi/3, 4\\pi/3)` and its
    time-reversed partner :math:`(4\\pi/3, 2\\pi/3)`, where the gap closes
    and the dispersion is linear (massless Dirac cone), as verified in
    the Examples.

    Examples
    --------
    >>> import numpy as np
    >>> K = np.array([2 * np.pi / 3, 4 * np.pi / 3])
    >>> np.round(np.linalg.eigvalsh(graphene_hamiltonian(*K)), 8)
    array([-0.,  0.])
    >>> eigs = np.linalg.eigvalsh(graphene_hamiltonian(*(K + [1e-4, 0])))
    >>> round(float(eigs[1] / 1e-4), 4)
    1.0
    """
    return graphene_lattice_hamiltonian(t).bloch([k1, k2])


# -- Haldane -----------------------------------------------------------------


def haldane_lattice_hamiltonian(t: float = 1.0, t2: float = 0.2, phi: float = np.pi / 2, M: float = 0.0) -> Hamiltonian:
    """Haldane model (quantum anomalous Hall) as a :class:`Hamiltonian`.

    Parameters
    ----------
    t : float, default=1.0
        Nearest-neighbor hopping amplitude.
    t2 : float, default=0.2
        Next-nearest-neighbor hopping magnitude.
    phi : float, default=pi/2
        Next-nearest-neighbor hopping phase, breaking time-reversal symmetry.
    M : float, default=0.0
        Sublattice (Semenoff) mass, breaking inversion symmetry.

    Returns
    -------
    Hamiltonian

    See Also
    --------
    haldane_model : Closed-form 2x2 Bloch Hamiltonian equivalent to this builder.
    """
    lat = Lattice.honeycomb()
    H = Hamiltonian(lat, onsite=[-M, M])
    H.add_hopping(0, 1, (0, 0), t)
    H.add_hopping(0, 1, (-1, 0), t)
    H.add_hopping(0, 1, (0, -1), t)
    t2c = t2 * np.exp(1j * phi)
    for R in [(1, 0), (0, -1), (-1, 1)]:
        H.add_hopping(0, 0, R, t2c)
        H.add_hopping(1, 1, R, np.conj(t2c))
    return H


def haldane_model(kx: float, ky: float, t: float = 1.0, t2: float = 0.2, phi: float = np.pi / 2, M: float = 0.0) -> np.ndarray:
    """Bloch Hamiltonian of the Haldane model on the honeycomb lattice.

    The Haldane model realizes the quantum anomalous Hall effect: a Chern
    insulator with zero *net* magnetic flux per unit cell, arising from a
    complex next-nearest-neighbor hopping :math:`t_2 e^{i\\phi}` that breaks
    time-reversal symmetry while preserving the lattice translational symmetry.

    Parameters
    ----------
    kx, ky : float
        Reduced crystal momenta (see :mod:`physicskit.condensed.tight_binding`),
        each periodic on :math:`[0, 2\\pi)`.
    t : float, default=1.0
        Nearest-neighbor hopping amplitude.
    t2 : float, default=0.2
        Next-nearest-neighbor hopping magnitude.
    phi : float, default=pi/2
        Next-nearest-neighbor hopping phase.
    M : float, default=0.0
        Sublattice mass. The model is a Chern insulator
        (:math:`C = \\mathrm{sgn}(\\sin\\phi)`) for :math:`|M| < 3\\sqrt{3}\\,t_2|\\sin\\phi|`,
        and a trivial insulator otherwise.

    Returns
    -------
    ndarray, shape (2, 2)

    See Also
    --------
    physicskit.condensed.topology.compute_chern_number : Computes the Chern number of this model.

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.condensed.topology import compute_chern_number
    >>> H_func = lambda k1, k2: haldane_model(k1, k2, t=1.0, t2=0.2, phi=np.pi / 2, M=0.0)
    >>> compute_chern_number(H_func, grid_size=30)
    [1, -1]
    >>> H_trivial = lambda k1, k2: haldane_model(k1, k2, t=1.0, t2=0.2, phi=np.pi / 2, M=2.0)
    >>> compute_chern_number(H_trivial, grid_size=30)
    [0, 0]
    """
    return haldane_lattice_hamiltonian(t, t2, phi, M).bloch([kx, ky])


# -- Kane-Mele -----------------------------------------------------------


def kane_mele_hamiltonian(
    kx: float,
    ky: float,
    t: float = 1.0,
    lambda_so: float = 0.06,
    lambda_v: float = 0.0,
    lambda_r: float = 0.0,
) -> np.ndarray:
    """Bloch Hamiltonian of the Kane-Mele quantum spin Hall model.

    Two time-reversed Haldane copies (intrinsic spin-orbit coupling with
    opposite Chern-number-generating phase for each spin), realizing a
    :math:`\\mathbb{Z}_2` topological insulator with helical edge states.
    Basis order is ``(A up, B up, A down, B down)``.

    Parameters
    ----------
    kx, ky : float
        Reduced crystal momenta, each periodic on :math:`[0, 2\\pi)`.
    t : float, default=1.0
        Nearest-neighbor hopping amplitude.
    lambda_so : float, default=0.06
        Intrinsic spin-orbit coupling strength.
    lambda_v : float, default=0.0
        Sublattice (staggered) potential.
    lambda_r : float, default=0.0
        Rashba spin-orbit coupling strength. A nonzero value mixes the spin
        blocks (breaking :math:`s_z` conservation) but preserves overall
        time-reversal symmetry.

    Returns
    -------
    ndarray, shape (4, 4)

    Notes
    -----
    When ``lambda_r == 0``, :math:`s_z` is conserved and the Hamiltonian is
    block-diagonal in spin; each block is a Haldane model with
    :math:`t_2 = \\lambda_{so}`, :math:`\\phi = \\pm\\pi/2`. In that case the
    :math:`\\mathbb{Z}_2` invariant equals the spin-up Chern number modulo 2
    (see :func:`physicskit.condensed.topology.z2_invariant`).

    Examples
    --------
    >>> import numpy as np
    >>> H = kane_mele_hamiltonian(0.3, 0.7, lambda_so=0.06)
    >>> np.allclose(H, H.conj().T)
    True
    >>> up = H[:2, :2]; down = H[2:, 2:]
    >>> down_expected = kane_mele_hamiltonian(0.3, 0.7, lambda_so=0.06)[2:, 2:]
    >>> np.allclose(down, down_expected)
    True
    """
    H_up = haldane_lattice_hamiltonian(t=t, t2=lambda_so, phi=np.pi / 2, M=lambda_v).bloch([kx, ky])
    H_down = haldane_lattice_hamiltonian(t=t, t2=lambda_so, phi=-np.pi / 2, M=lambda_v).bloch([kx, ky])
    H = np.zeros((4, 4), dtype=complex)
    H[:2, :2] = H_up
    H[2:, 2:] = H_down

    if lambda_r != 0.0:
        lat = Lattice.honeycomb()
        orb_cart = lat.cartesian_orbitals()
        bond_R = [(0, 0), (-1, 0), (0, -1)]
        bond_vecs = []
        for R in bond_R:
            Rcart = np.array(R, dtype=float) @ lat.lattice_vectors
            bond_vecs.append(orb_cart[1] + Rcart - orb_cart[0])
        R_block = np.zeros((2, 2), dtype=complex)
        for R, d in zip(bond_R, bond_vecs, strict=True):
            d_hat = d / np.linalg.norm(d)
            spin_mat = 1j * lambda_r * (_SIGMA_X * d_hat[1] - _SIGMA_Y * d_hat[0])
            phase = np.exp(1j * (kx * R[0] + ky * R[1]))
            R_block += spin_mat * phase
        # Off-diagonal sublattice block (A up/down <-> B up/down), spin-mixing Rashba term.
        H_AB = np.zeros((2, 2), dtype=complex)
        H_AB[0, 0] = R_block[0, 0]
        H_AB[1, 1] = R_block[1, 1]
        H_AB[0, 1] = R_block[0, 1]
        H_AB[1, 0] = R_block[1, 0]
        H[0, 2] += H_AB[0, 0]
        H[1, 3] += H_AB[1, 1]
        H[0, 3] += H_AB[0, 1]
        H[1, 2] += H_AB[1, 0]
        H[2, 0] += np.conj(H_AB[0, 0])
        H[3, 1] += np.conj(H_AB[1, 1])
        H[3, 0] += np.conj(H_AB[0, 1])
        H[2, 1] += np.conj(H_AB[1, 0])
    return H


# -- BHZ -----------------------------------------------------------------


def bhz_hamiltonian(kx: float, ky: float, A: float = 1.0, B: float = 1.0, M: float = 1.0, D: float = 0.0) -> np.ndarray:
    """Bloch Hamiltonian of the Bernevig-Hughes-Zhang (BHZ) model for HgTe quantum wells.

    A minimal 4-band :math:`\\mathbb{Z}_2` topological insulator model,
    block-diagonal in a time-reversed pair of 2x2 Dirac-like blocks. Basis
    order is ``(E up, H up, E down, H down)``.

    Parameters
    ----------
    kx, ky : float
        Reduced crystal momenta, each periodic on :math:`[0, 2\\pi)`.
    A, B : float, default=1.0
        Model parameters controlling the Dirac velocity and quadratic
        band curvature.
    M : float, default=1.0
        Band inversion mass. The model is topological (band-inverted) for
        :math:`M/B > 0` and trivial for :math:`M/B < 0`.
    D : float, default=0.0
        Particle-hole asymmetry parameter.

    Returns
    -------
    ndarray, shape (4, 4)

    Examples
    --------
    >>> import numpy as np
    >>> H = bhz_hamiltonian(0.1, -0.2, M=1.0, B=1.0)
    >>> np.allclose(H, H.conj().T)
    True
    >>> np.allclose(H[:2, :2], H[2:, 2:].conj())
    True
    """
    eps = -D * (2 - np.cos(kx) - np.cos(ky))
    dx = A * np.sin(kx)
    dy = A * np.sin(ky)
    dz = M - 2 * B * (2 - np.cos(kx) - np.cos(ky))
    h_up = eps * _SIGMA_0 + dx * _SIGMA_X + dy * _SIGMA_Y + dz * _SIGMA_Z
    h_down = np.conj(h_up)
    H = np.zeros((4, 4), dtype=complex)
    H[:2, :2] = h_up
    H[2:, 2:] = h_down
    return H


def bhz_ribbon_hamiltonian(kx: float, n_cells: int, A: float = 1.0, B: float = 1.0, M: float = 1.0, D: float = 0.0) -> np.ndarray:
    """Real-space BHZ ribbon: periodic along ``x``, open (finite) along ``y``.

    Truncating the ``y`` direction exposes the pair of helical, spin-locked
    edge states that make the BHZ model a quantum spin Hall insulator --
    the effect Konig et al. (2007) measured directly in HgTe/CdTe quantum
    wells, by two-terminal conductance quantized at :math:`2e^2/h`.

    Parameters
    ----------
    kx : float
        Reduced crystal momentum along the periodic (``x``) direction.
    n_cells : int
        Number of unit cells stacked along the open (``y``) direction.
    A, B : float, default=1.0
        Model parameters controlling the Dirac velocity and quadratic
        band curvature (see :func:`bhz_hamiltonian`).
    M : float, default=1.0
        Band inversion mass. Topological (edge-state-carrying) for
        :math:`M/B > 0`.
    D : float, default=0.0
        Particle-hole asymmetry parameter.

    Returns
    -------
    ndarray, shape (4 * n_cells, 4 * n_cells)
        Hermitian, open-boundary ribbon Hamiltonian, block-diagonal in the
        ``(E up, H up)`` / ``(E down, H down)`` time-reversed sectors.

    Examples
    --------
    In the topological regime (:math:`M/B > 0`), the ribbon has a pair of
    near-zero-energy states crossing at :math:`k_x = 0` -- the helical edge
    modes -- absent in the trivial regime (:math:`M/B < 0`):

    >>> import numpy as np
    >>> spectrum = np.linalg.eigvalsh(bhz_ribbon_hamiltonian(kx=0.0, n_cells=40, M=1.0, B=1.0))
    >>> bool(np.any(np.abs(spectrum) < 1e-6))
    True
    >>> spectrum_trivial = np.linalg.eigvalsh(bhz_ribbon_hamiltonian(kx=0.0, n_cells=40, M=-1.0, B=1.0))
    >>> bool(np.any(np.abs(spectrum_trivial) < 1e-6))
    False
    """
    eps0 = -D * (2 - np.cos(kx))
    dx = A * np.sin(kx)
    dz0 = M - 2 * B * (2 - np.cos(kx))
    onsite_up = eps0 * _SIGMA_0 + dx * _SIGMA_X + dz0 * _SIGMA_Z
    onsite_down = onsite_up
    hop_up = 0.5 * (D * _SIGMA_0 + 2 * B * _SIGMA_Z) - 0.5j * A * _SIGMA_Y
    hop_down = hop_up.conj().T

    N = n_cells
    H = np.zeros((4 * N, 4 * N), dtype=complex)
    for c in range(N):
        H[4 * c : 4 * c + 2, 4 * c : 4 * c + 2] = onsite_up
        H[4 * c + 2 : 4 * c + 4, 4 * c + 2 : 4 * c + 4] = onsite_down
    for c in range(N - 1):
        H[4 * c : 4 * c + 2, 4 * c + 4 : 4 * c + 6] = hop_up
        H[4 * c + 4 : 4 * c + 6, 4 * c : 4 * c + 2] = hop_up.conj().T
        H[4 * c + 2 : 4 * c + 4, 4 * c + 6 : 4 * c + 8] = hop_down
        H[4 * c + 6 : 4 * c + 8, 4 * c + 2 : 4 * c + 4] = hop_down.conj().T
    return H


# -- Kitaev chain --------------------------------------------------------


def kitaev_chain_hamiltonian(k: float, mu: float = 0.0, t: float = 1.0, delta: float = 1.0) -> np.ndarray:
    """Bogoliubov-de Gennes Bloch Hamiltonian of the Kitaev p-wave superconducting chain.

    Parameters
    ----------
    k : float
        Reduced crystal momentum, periodic on :math:`[0, 2\\pi)`.
    mu : float, default=0.0
        Chemical potential.
    t : float, default=1.0
        Nearest-neighbor hopping amplitude.
    delta : float, default=1.0
        p-wave pairing amplitude.

    Returns
    -------
    ndarray, shape (2, 2)
        BdG Hamiltonian :math:`H(k) = \\xi(k)\\tau_z + \\Delta(k)\\tau_y` in the
        Nambu basis, with :math:`\\xi(k) = -2t\\cos k - \\mu` and
        :math:`\\Delta(k) = 2\\Delta\\sin k`.

    Notes
    -----
    The chain is a topological superconductor, hosting unpaired Majorana
    zero modes at the ends of an open chain, for :math:`|\\mu| < 2t`; it is
    topologically trivial for :math:`|\\mu| > 2t`.

    Examples
    --------
    >>> import numpy as np
    >>> H = kitaev_chain_hamiltonian(k=np.pi / 2, mu=0.0, t=1.0, delta=1.0)
    >>> np.round(np.linalg.eigvalsh(H), 8)
    array([-2.,  2.])
    """
    xi = -2 * t * np.cos(k) - mu
    d = 2 * delta * np.sin(k)
    return xi * _SIGMA_Z + d * _SIGMA_Y


def kitaev_chain_bdg_real_space(n_sites: int, mu: float = 0.0, t: float = 1.0, delta: float = 1.0) -> np.ndarray:
    """Real-space open-boundary BdG Hamiltonian of the Kitaev chain.

    Parameters
    ----------
    n_sites : int
        Number of lattice sites in the finite, open chain.
    mu : float, default=0.0
        Chemical potential.
    t : float, default=1.0
        Nearest-neighbor hopping amplitude.
    delta : float, default=1.0
        p-wave pairing amplitude.

    Returns
    -------
    ndarray, shape (2*n_sites, 2*n_sites)
        BdG Hamiltonian in the Nambu basis
        :math:`(c_1, \\dots, c_N, c_1^\\dagger, \\dots, c_N^\\dagger)`.

    Examples
    --------
    In the topological phase (:math:`|\\mu| < 2t`), diagonalizing this matrix
    yields a pair of near-zero-energy eigenvalues (Majorana end modes),
    exponentially localized at the two ends of the chain:

    >>> import numpy as np
    >>> H = kitaev_chain_bdg_real_space(n_sites=60, mu=0.0, t=1.0, delta=1.0)
    >>> eigs = np.linalg.eigvalsh(H)
    >>> bool(np.abs(eigs[np.argmin(np.abs(eigs))]) < 1e-6)
    True
    """
    N = n_sites
    H = np.zeros((2 * N, 2 * N), dtype=complex)
    for i in range(N):
        H[i, i] = -mu
        H[N + i, N + i] = mu
    for i in range(N - 1):
        H[i, i + 1] += -t
        H[i + 1, i] += -t
        H[N + i, N + i + 1] += t
        H[N + i + 1, N + i] += t
        H[i, N + i + 1] += delta
        H[N + i + 1, i] += np.conj(delta)
        H[i + 1, N + i] += -delta
        H[N + i, i + 1] += -np.conj(delta)
    return H


# -- Harper-Hofstadter --------------------------------------------------------


def harper_hofstadter_hamiltonian(k1: float, k2: float, p: int, q: int, t: float = 1.0) -> np.ndarray:
    r"""Bloch Hamiltonian of the square-lattice Harper-Hofstadter model at flux ``p/q``.

    A charged particle on a square lattice threaded by a uniform magnetic
    flux :math:`p/q` (in units of the flux quantum) per plaquette, in
    Landau gauge :math:`\mathbf{A} = (0, Bx)`. Translational symmetry along
    ``x`` survives only in steps of ``q`` lattice constants, so the ``q``
    inequivalent sublattice sites ``m = 0, ..., q-1`` within one magnetic
    unit cell become an internal band index, coupled by

    .. math::

        H_{mm}(k_1, k_2) = -2t\cos\!\left(k_2 + 2\pi \frac{p}{q} m\right), \qquad
        H_{m, m+1} = -t,

    plus the boundary hopping :math:`H_{q-1, 0} = -t\,e^{ik_1}` that closes
    the magnetic unit cell, with :math:`k_1` the reduced momentum conjugate
    to translation by one magnetic cell (period :math:`q` sites) and
    :math:`k_2` the ordinary reduced momentum along the unbroken direction.
    This is the lattice (Bloch) route to the same physics as
    :mod:`physicskit.condensed.landau_levels`, and, fed one band at a time
    into :func:`~physicskit.condensed.topology.compute_chern_number`, the
    standard numerical verification of the TKNN integer quantum Hall
    formula: summing the Chern numbers of the lowest :math:`r` (non-touching)
    bands gives the exactly quantized Hall conductance
    :math:`\sigma_{xy} = C\,e^2/h` at the filling between band :math:`r` and
    :math:`r+1`.

    Parameters
    ----------
    k1, k2 : float
        Reduced crystal momenta, each periodic on :math:`[0, 2\pi)`. ``k1``
        is conjugate to the enlarged (``q``-site) magnetic unit cell.
    p, q : int
        Flux per plaquette :math:`p/q` (in lowest terms), :math:`0 < p < q`.
        Odd ``q`` avoids the exact band touchings that occur for some even
        ``q``, where individual-band Chern numbers become ill-defined.
    t : float, default=1.0
        Nearest-neighbor hopping amplitude.

    Returns
    -------
    ndarray, shape (q, q)
        Hermitian Bloch Hamiltonian.

    See Also
    --------
    physicskit.condensed.topology.compute_chern_number : Chern number of each resulting band.

    Examples
    --------
    At flux ``1/3`` the three bands carry Chern numbers ``-1, 2, -1``
    (summing to zero, as any complete set of bands of a lattice Hamiltonian
    must), so the Hall conductance is quantized to :math:`-1` and :math:`+1`
    (in units of :math:`e^2/h`) at the two gaps:

    >>> import numpy as np
    >>> from physicskit.condensed.topology import compute_chern_number
    >>> H_func = lambda k1, k2: harper_hofstadter_hamiltonian(k1, k2, p=1, q=3)
    >>> chern_numbers = compute_chern_number(H_func, grid_size=30)
    >>> chern_numbers
    [-1, 2, -1]
    >>> sum(chern_numbers)
    0
    """
    H = np.zeros((q, q), dtype=complex)
    m = np.arange(q)
    H[np.diag_indices(q)] = -2 * t * np.cos(k2 + 2 * np.pi * p * m / q)
    for i in range(q - 1):
        H[i, i + 1] = -t
        H[i + 1, i] = -t
    H[q - 1, 0] += -t * np.exp(1j * k1)
    H[0, q - 1] += -t * np.exp(-1j * k1)
    return H
