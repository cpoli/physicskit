"""Correlated-electron models: Bogoliubov-de Gennes superconductivity and the Hubbard model.

Provides a mean-field BCS/Bogoliubov-de Gennes (BdG) solver for
s-wave superconductivity, and small-cluster exact diagonalization (ED) of
the 1D Fermi-Hubbard model.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

__all__ = ["bdg_bcs_hamiltonian", "bdg_spectrum", "hubbard_1d_exact_diagonalization", "hubbard_spin_correlations"]

_TAU_X = np.array([[0, 1], [1, 0]], dtype=complex)
_TAU_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_TAU_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def bdg_bcs_hamiltonian(k: float, mu: float = 0.0, t: float = 1.0, delta: complex = 0.5) -> np.ndarray:
    """Mean-field Bogoliubov-de Gennes Hamiltonian for a 1D s-wave BCS superconductor.

    Parameters
    ----------
    k : float
        Reduced crystal momentum, periodic on :math:`[0, 2\\pi)`.
    mu : float, default=0.0
        Chemical potential.
    t : float, default=1.0
        Nearest-neighbor hopping amplitude (sets the normal-state band
        :math:`\\xi(k) = -2t\\cos k - \\mu`).
    delta : complex, default=0.5
        s-wave (momentum-independent) pairing amplitude.

    Returns
    -------
    ndarray, shape (2, 2)
        BdG Hamiltonian :math:`H(k) = \\xi(k)\\tau_z + \\mathrm{Re}(\\Delta)\\tau_x - \\mathrm{Im}(\\Delta)\\tau_y`
        in the Nambu basis :math:`(c_k, c_{-k}^\\dagger)`.

    See Also
    --------
    bdg_spectrum : Quasiparticle energies over a grid of k, exposing the gap.

    Examples
    --------
    >>> import numpy as np
    >>> H = bdg_bcs_hamiltonian(k=np.pi / 2, mu=0.0, t=1.0, delta=0.5)
    >>> np.round(np.linalg.eigvalsh(H), 8)
    array([-0.5,  0.5])
    """
    xi = -2 * t * np.cos(k) - mu
    return xi * _TAU_Z + np.real(delta) * _TAU_X - np.imag(delta) * _TAU_Y


def bdg_spectrum(mu: float = 0.0, t: float = 1.0, delta: complex = 0.5, n_k: int = 200) -> tuple:
    """Quasiparticle (positive-energy) BdG spectrum over the 1D Brillouin zone.

    Parameters
    ----------
    mu : float, default=0.0
        Chemical potential.
    t : float, default=1.0
        Nearest-neighbor hopping amplitude.
    delta : complex, default=0.5
        s-wave pairing amplitude.
    n_k : int, default=200
        Number of momentum points.

    Returns
    -------
    k_grid : ndarray, shape (n_k,)
        Momentum grid over :math:`[0, 2\\pi)`.
    energies : ndarray, shape (n_k,)
        Positive quasiparticle branch :math:`E(k) = \\sqrt{\\xi(k)^2 + |\\Delta|^2}`.

    Examples
    --------
    The minimum quasiparticle energy equals the pairing gap :math:`|\\Delta|`:

    >>> k, E = bdg_spectrum(mu=0.0, t=1.0, delta=0.5, n_k=400)
    >>> round(float(E.min()), 3)
    0.5
    """
    k_grid = np.linspace(0, 2 * np.pi, n_k, endpoint=False)
    energies = np.array([np.linalg.eigvalsh(bdg_bcs_hamiltonian(k, mu, t, delta)).max() for k in k_grid])
    return k_grid, energies


# -- 1D Fermi-Hubbard exact diagonalization ----------------------------------


def _sector_states(n_sites: int, n_particles: int) -> list:
    """All occupation bitmasks with exactly ``n_particles`` set bits among ``n_sites``."""
    states = []
    for occ in combinations(range(n_sites), n_particles):
        mask = 0
        for site in occ:
            mask |= 1 << site
        states.append(mask)
    return sorted(states)


def _fermion_sign(mask: int, site: int) -> int:
    """Jordan-Wigner sign from anticommuting a fermion operator past occupied sites below ``site``."""
    below = mask & ((1 << site) - 1)
    return -1 if bin(below).count("1") % 2 else 1


def _hopping_sector_hamiltonian(n_sites: int, n_particles: int, t: float, pbc: bool) -> tuple:
    """Single-spin-species kinetic-energy matrix :math:`-t\\sum_{\\langle ij\\rangle} c_i^\\dagger c_j + \\text{h.c.}`."""
    states = _sector_states(n_sites, n_particles)
    index = {s: i for i, s in enumerate(states)}
    dim = len(states)
    H = np.zeros((dim, dim))
    bonds = [(i, i + 1) for i in range(n_sites - 1)]
    if pbc and n_sites > 2:
        bonds.append((n_sites - 1, 0))
    for n, mask in enumerate(states):
        for i, j in bonds:
            for src, dst in ((i, j), (j, i)):
                if (mask >> src) & 1 and not (mask >> dst) & 1:
                    sign = _fermion_sign(mask, src)
                    m1 = mask & ~(1 << src)
                    sign *= _fermion_sign(m1, dst)
                    m2 = m1 | (1 << dst)
                    H[index[m2], n] += -t * sign
    return H, states, index


def hubbard_1d_exact_diagonalization(
    n_sites: int, n_up: int, n_dn: int, t: float = 1.0, U: float = 4.0, pbc: bool = False, return_eigenvectors: bool = False
) -> dict:
    """Exact diagonalization of the 1D Fermi-Hubbard model on a small cluster.

    .. math::

        H = -t \\sum_{\\langle ij\\rangle, \\sigma} c_{i\\sigma}^\\dagger c_{j\\sigma}
            + U \\sum_i n_{i\\uparrow} n_{i\\downarrow}

    Builds the Fock-space Hamiltonian in the fixed :math:`(n_\\uparrow, n_\\downarrow)`
    sector via full (dense) diagonalization; suitable for small clusters
    (:math:`n_{sites} \\lesssim 8`).

    Parameters
    ----------
    n_sites : int
        Number of lattice sites.
    n_up, n_dn : int
        Number of spin-up and spin-down electrons.
    t : float, default=1.0
        Nearest-neighbor hopping amplitude.
    U : float, default=4.0
        Onsite Coulomb repulsion.
    pbc : bool, default=False
        Periodic (``True``) or open (``False``) boundary conditions.
    return_eigenvectors : bool, default=False
        If ``True``, also return the ground-state vector and the occupation-
        number bases it is expanded in, e.g. for
        :func:`hubbard_spin_correlations`.

    Returns
    -------
    dict
        ``{"ground_state_energy": float, "eigenvalues": ndarray, "dimension": int}``,
        plus ``{"ground_state_vector": ndarray, "states_up": list, "states_dn": list}``
        when ``return_eigenvectors=True``. ``ground_state_vector[a * len(states_dn) + b]``
        is the amplitude of the product basis state ``(states_up[a], states_dn[b])``,
        each an integer bitmask with bit ``i`` set if site ``i`` is occupied.

    Notes
    -----
    In the large-\\ :math:`U` limit at half filling, double occupancy is
    suppressed and the ground state approaches the Mott-insulating,
    singly-occupied regime characteristic of spin-charge separation in 1D.

    Examples
    --------
    The 2-site Hubbard dimer at half filling has the exact ground-state
    energy :math:`E_0 = \\tfrac{1}{2}\\big(U - \\sqrt{U^2 + 16t^2}\\big)`:

    >>> import numpy as np
    >>> result = hubbard_1d_exact_diagonalization(n_sites=2, n_up=1, n_dn=1, t=1.0, U=4.0)
    >>> E_exact = 0.5 * (4.0 - np.sqrt(4.0**2 + 16 * 1.0**2))
    >>> bool(round(result["ground_state_energy"], 8) == round(E_exact, 8))
    True
    """
    H_up, states_up, _ = _hopping_sector_hamiltonian(n_sites, n_up, t, pbc)
    H_dn, states_dn, _ = _hopping_sector_hamiltonian(n_sites, n_dn, t, pbc)
    dim_up, dim_dn = len(states_up), len(states_dn)
    dim = dim_up * dim_dn

    H = np.kron(H_up, np.eye(dim_dn)) + np.kron(np.eye(dim_up), H_dn)

    interaction = np.zeros(dim)
    for a, mask_up in enumerate(states_up):
        for b, mask_dn in enumerate(states_dn):
            n_double = bin(mask_up & mask_dn).count("1")
            interaction[a * dim_dn + b] = U * n_double
    H += np.diag(interaction)

    if return_eigenvectors:
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        return {
            "ground_state_energy": float(eigenvalues[0]),
            "eigenvalues": eigenvalues,
            "dimension": dim,
            "ground_state_vector": eigenvectors[:, 0],
            "states_up": states_up,
            "states_dn": states_dn,
        }
    eigenvalues = np.linalg.eigvalsh(H)
    return {"ground_state_energy": float(eigenvalues[0]), "eigenvalues": eigenvalues, "dimension": dim}


def hubbard_spin_correlations(n_sites: int, ground_state_vector: np.ndarray, states_up: list, states_dn: list) -> np.ndarray:
    """Equal-time z-spin correlations :math:`\\langle S_i^z S_j^z\\rangle` from a Hubbard ground state.

    :math:`S_i^z = \\tfrac{1}{2}(n_{i\\uparrow} - n_{i\\downarrow})` is diagonal
    in the occupation-number basis
    :func:`hubbard_1d_exact_diagonalization` builds its Hamiltonian in, so
    :math:`\\langle S_i^z S_j^z\\rangle` reduces to a weighted sum over basis-state
    probabilities, :math:`\\sum_\\alpha |\\psi_\\alpha|^2\\, s_i(\\alpha)\\, s_j(\\alpha)`,
    with no off-diagonal matrix elements to track.

    Parameters
    ----------
    n_sites : int
        Number of lattice sites.
    ground_state_vector : ndarray
        Ground-state amplitudes, as returned by
        :func:`hubbard_1d_exact_diagonalization` with ``return_eigenvectors=True``.
    states_up, states_dn : list of int
        Occupation-number bases for each spin species, as returned alongside
        ``ground_state_vector``.

    Returns
    -------
    ndarray, shape (n_sites,)
        ``correlations[r]`` is :math:`\\langle S_i^z S_j^z\\rangle` averaged
        over every site pair with :math:`|i-j|=r`; ``correlations[0]`` is the
        onsite :math:`\\langle (S_i^z)^2\\rangle`, suppressed toward 0 by
        double/empty occupancy and toward its :math:`1/4` ceiling by strong
        onsite repulsion.

    Notes
    -----
    A Mott-insulating antiferromagnet's short-range Neel order shows up as
    ``correlations`` alternating in sign with ``r`` -- negative
    (antialigned) at odd separations, positive (aligned) at even ones --
    the same short-range magnetic correlations believed to survive doping
    into the cuprate superconductors' metallic phase.

    Examples
    --------
    At half filling and strong coupling, nearest-neighbor spins are
    antialigned:

    >>> import numpy as np
    >>> result = hubbard_1d_exact_diagonalization(n_sites=6, n_up=3, n_dn=3, t=1.0, U=8.0, pbc=True, return_eigenvectors=True)
    >>> corr = hubbard_spin_correlations(6, result["ground_state_vector"], result["states_up"], result["states_dn"])
    >>> bool(corr[1] < 0)
    True
    """
    dim_dn = len(states_dn)
    weights = np.abs(np.asarray(ground_state_vector)) ** 2
    sz = np.zeros((len(weights), n_sites))
    for a, mask_up in enumerate(states_up):
        for b, mask_dn in enumerate(states_dn):
            idx = a * dim_dn + b
            for site in range(n_sites):
                n_up_site = (mask_up >> site) & 1
                n_dn_site = (mask_dn >> site) & 1
                sz[idx, site] = 0.5 * (n_up_site - n_dn_site)

    correlations = np.zeros(n_sites)
    counts = np.zeros(n_sites)
    for i in range(n_sites):
        for j in range(n_sites):
            r = abs(i - j)
            correlations[r] += np.sum(weights * sz[:, i] * sz[:, j])
            counts[r] += 1
    return correlations / counts
