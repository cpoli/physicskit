"""Berry curvature, Chern numbers, Zak phase, and Z2 invariants.

The workhorse here is the Fukui-Hatsugai-Suzuki (FHS) lattice algorithm,
which computes an exactly quantized, gauge-invariant Chern number from a
finite grid of Bloch eigenvectors -- no smooth gauge choice required. The
per-plaquette flux accumulation is numba-jitted since it is the inner loop
of a double sum over the whole Brillouin-zone grid.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["compute_berry_curvature", "compute_chern_number", "zak_phase", "z2_invariant"]


def _bloch_eigensystem(hamiltonian_func, grid_size: int):
    """Diagonalize ``hamiltonian_func`` on a closed (grid_size+1)^2 k-grid.

    The extra row/column closes the discretized Brillouin-zone torus so
    that plaquette (a, n) reuses the same eigenvectors as plaquette (a, 0).
    """
    n = grid_size
    sample = np.asarray(hamiltonian_func(0.0, 0.0), dtype=complex)
    norb = sample.shape[0]
    U = np.empty((n + 1, n + 1, norb, norb), dtype=complex)
    for a in range(n + 1):
        k1 = 2 * np.pi * a / n
        for b in range(n + 1):
            k2 = 2 * np.pi * b / n
            H = np.asarray(hamiltonian_func(k1, k2), dtype=complex)
            _, v = np.linalg.eigh(H)
            U[a, b] = v.T  # U[a, b, band, :] is the band-th eigenvector (contiguous)
    return U


@njit(cache=True)
def _fhs_curvature_grid(U, n, band):
    F = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            v00 = U[a, b, band, :]
            v10 = U[a + 1, b, band, :]
            v11 = U[a + 1, b + 1, band, :]
            v01 = U[a, b + 1, band, :]
            u1 = np.vdot(v00, v10)
            u1 = u1 / abs(u1)
            u2 = np.vdot(v10, v11)
            u2 = u2 / abs(u2)
            u3 = np.vdot(v11, v01)
            u3 = u3 / abs(u3)
            u4 = np.vdot(v01, v00)
            u4 = u4 / abs(u4)
            F[a, b] = np.angle(u1 * u2 * u3 * u4)
    return F


def compute_berry_curvature(hamiltonian_func, grid_size: int = 50, band_index: int = 0) -> np.ndarray:
    """Compute the discretized Berry curvature of one band over the Brillouin zone.

    Uses the Fukui-Hatsugai-Suzuki (FHS) link-variable formula, which is
    gauge invariant plaquette by plaquette and needs no smooth choice of
    eigenvector phase.

    Parameters
    ----------
    hamiltonian_func : callable
        A function ``H(k1, k2)`` returning the ``(N, N)`` complex Bloch
        Hamiltonian at reduced crystal momentum ``(k1, k2)``, each
        periodic on :math:`[0, 2\\pi)` (see :mod:`physicskit.condensed.tight_binding`).
    grid_size : int, default=50
        Number of plaquettes along each of :math:`k_1, k_2`.
    band_index : int, default=0
        Band index (0 = lowest energy) to compute the curvature for.

    Returns
    -------
    ndarray, shape (grid_size, grid_size)
        Berry flux through each plaquette, in :math:`(-\\pi, \\pi]`. Summing
        and dividing by :math:`2\\pi` gives the band's Chern number.

    See Also
    --------
    compute_chern_number : Integrates this curvature over the full zone for every band.

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.condensed.models import haldane_model
    >>> H_func = lambda k1, k2: haldane_model(k1, k2, t=1.0, t2=0.2, phi=np.pi / 2, M=0.0)
    >>> F = compute_berry_curvature(H_func, grid_size=30, band_index=0)
    >>> F.shape
    (30, 30)
    >>> round(float(F.sum() / (2 * np.pi)))
    1
    """
    U = _bloch_eigensystem(hamiltonian_func, grid_size)
    return _fhs_curvature_grid(U, grid_size, band_index)


def compute_chern_number(hamiltonian_func, grid_size: int = 50) -> list:
    """Compute the Chern number of every band using the Fukui-Hatsugai-Suzuki method.

    Parameters
    ----------
    hamiltonian_func : callable
        A function ``H(k1, k2)`` returning an ``(N, N)`` complex Bloch
        Hamiltonian, using the reduced-momentum convention of
        :mod:`physicskit.condensed.tight_binding`.
    grid_size : int, default=50
        Discretization resolution for :math:`k_1` and :math:`k_2` across
        :math:`[0, 2\\pi)`.

    Returns
    -------
    list of int
        Chern integer for each energy band, ordered from lowest to highest energy.

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.condensed.models import haldane_model
    >>> H_func = lambda k1, k2: haldane_model(k1, k2, t=1.0, t2=0.2, phi=np.pi / 2, M=0.0)
    >>> chern_numbers = compute_chern_number(H_func, grid_size=30)
    >>> print(chern_numbers)
    [1, -1]
    """
    U = _bloch_eigensystem(hamiltonian_func, grid_size)
    norb = U.shape[-1]
    return [int(round(_fhs_curvature_grid(U, grid_size, band).sum() / (2 * np.pi))) for band in range(norb)]


def zak_phase(hamiltonian_func_1d, grid_size: int = 200, band_index: int = 0) -> float:
    """Compute the Zak phase of a 1D band: the Berry phase accumulated across the BZ.

    Parameters
    ----------
    hamiltonian_func_1d : callable
        A function ``H(k)`` returning the ``(N, N)`` complex Bloch Hamiltonian
        at reduced momentum ``k``, periodic on :math:`[0, 2\\pi)`.
    grid_size : int, default=200
        Number of discretization steps around the 1D Brillouin zone.
    band_index : int, default=0
        Band index (0 = lowest energy).

    Returns
    -------
    float
        The Zak phase, wrapped to :math:`(-\\pi, \\pi]`. For a chiral-symmetric
        model such as SSH, this is quantized to :math:`0` (trivial) or
        :math:`\\pi` (topological).

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.condensed.models import ssh_hamiltonian
    >>> topological = lambda k: ssh_hamiltonian(k, v=0.5, w=1.0)
    >>> trivial = lambda k: ssh_hamiltonian(k, v=1.0, w=0.5)
    >>> round(abs(zak_phase(topological)), 4)
    3.1416
    >>> round(abs(zak_phase(trivial)), 4)
    0.0
    """
    n = grid_size
    vecs = np.empty((n + 1, hamiltonian_func_1d(0.0).shape[0]), dtype=complex)
    for i in range(n + 1):
        k = 2 * np.pi * i / n
        _, v = np.linalg.eigh(np.asarray(hamiltonian_func_1d(k), dtype=complex))
        vecs[i] = v[:, band_index]
    prod = 1.0 + 0j
    for i in range(n):
        overlap = np.vdot(vecs[i], vecs[i + 1])
        prod *= overlap / abs(overlap)
    return float(-np.angle(prod))


def z2_invariant(hamiltonian_func, grid_size: int = 30, spin_block: tuple = (0, 2)) -> int:
    """Compute the :math:`\\mathbb{Z}_2` invariant of an :math:`s_z`-conserving spinful model.

    Valid whenever ``hamiltonian_func`` is block-diagonal in spin (e.g. the
    Kane-Mele or BHZ models with Rashba coupling set to zero), in which case
    the :math:`\\mathbb{Z}_2` invariant reduces to the spin-up sector's Chern
    number modulo 2 (Sheng et al. / spin-Chern-number approach).

    Parameters
    ----------
    hamiltonian_func : callable
        A function ``H(k1, k2)`` returning the full ``(2N, 2N)`` Bloch
        Hamiltonian with the spin-up block occupying ``spin_block``.
    grid_size : int, default=30
        Discretization resolution for the Chern-number calculation.
    spin_block : tuple of int, default=(0, 2)
        ``(start, stop)`` slice selecting the spin-up block's rows/columns.

    Returns
    -------
    int
        :math:`\\mathbb{Z}_2` invariant, 0 (trivial) or 1 (topological).

    Examples
    --------
    >>> from physicskit.condensed.models import kane_mele_hamiltonian
    >>> trivial = lambda k1, k2: kane_mele_hamiltonian(k1, k2, lambda_so=0.0)
    >>> topological = lambda k1, k2: kane_mele_hamiltonian(k1, k2, lambda_so=0.06)
    >>> z2_invariant(trivial, grid_size=20)
    0
    >>> z2_invariant(topological, grid_size=20)
    1
    """
    lo, hi = spin_block

    def h_up(k1, k2):
        return np.asarray(hamiltonian_func(k1, k2), dtype=complex)[lo:hi, lo:hi]

    chern_up = compute_chern_number(h_up, grid_size)
    return int(chern_up[0]) % 2
