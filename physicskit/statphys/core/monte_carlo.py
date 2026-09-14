"""Numba-accelerated Monte Carlo kernels for lattice spin models.

This module holds the hot inner loops shared by :mod:`physicskit.statphys.chapters.ising_lattice`:
single-spin-flip Metropolis-Hastings sweeps for the Ising and Potts models, a
continuous-angle Metropolis sweep for the XY model, and the Wolff single-cluster
algorithm for the Ising model. Functions are JIT-compiled with :func:`numba.njit`
so that a full lattice sweep of :math:`L \\times L` sites costs a handful of
microseconds rather than a Python-loop pass.

All lattices use periodic boundary conditions.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = [
    "ising_total_energy",
    "ising_total_magnetization",
    "metropolis_sweep_ising",
    "metropolis_sweep_potts",
    "metropolis_sweep_sk",
    "metropolis_sweep_spin_glass",
    "metropolis_sweep_xy",
    "potts_total_energy",
    "seed_numba_random",
    "sk_total_energy",
    "spin_glass_total_energy",
    "wolff_step_ising",
    "xy_plaquette_vorticity",
    "xy_total_energy",
]


@njit(cache=True)
def seed_numba_random(seed):
    """Seed the random-number generator used by every ``@njit`` kernel in physicskit.statphys.

    Plain ``numpy.random.seed`` only reseeds NumPy's own global generator
    and has **no effect** on Numba-compiled (``@njit``) code, which
    maintains a single separate internal RNG state shared by every
    ``@njit`` function in the process (regardless of which module defines
    them). Calling ``np.random.seed`` from *inside* jitted code, as this
    thin wrapper does, is the only way to obtain reproducible dynamics from
    any of physicskit.statphys's Monte Carlo sweep or random-walk kernels.

    Parameters
    ----------
    seed : int
        Seed value.
    """
    np.random.seed(seed)


@njit(cache=True)
def _ising_neighbor_sum(spins, i, j, L):
    top = spins[(i - 1) % L, j]
    bottom = spins[(i + 1) % L, j]
    left = spins[i, (j - 1) % L]
    right = spins[i, (j + 1) % L]
    return top + bottom + left + right


@njit(cache=True)
def metropolis_sweep_ising(spins, beta, J):
    """Perform one Metropolis-Hastings sweep of an Ising lattice, in place.

    Parameters
    ----------
    spins : ndarray of shape (L, L), dtype int64
        Spin configuration with values in ``{-1, +1}``. Modified in place.
    beta : float
        Inverse temperature :math:`1 / (k_B T)`.
    J : float
        Ferromagnetic (``J > 0``) or antiferromagnetic (``J < 0``) coupling.

    Returns
    -------
    ndarray of shape (L, L)
        The same array passed in (returned for convenience).

    Notes
    -----
    One sweep proposes :math:`L^2` single-spin flips at randomly chosen sites,
    each accepted with probability :math:`\\min(1, e^{-\\beta \\Delta E})`, where
    :math:`\\Delta E = 2 J s_i \\sum_{\\langle i,j \\rangle} s_j`.
    """
    L = spins.shape[0]
    n = L * L
    for _ in range(n):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        s = spins[i, j]
        neighbor_sum = _ising_neighbor_sum(spins, i, j, L)
        dE = 2.0 * J * s * neighbor_sum
        if dE <= 0.0 or np.random.random() < np.exp(-beta * dE):
            spins[i, j] = -s
    return spins


@njit(cache=True)
def wolff_step_ising(spins, beta, J):
    """Grow and flip a single Wolff cluster on an Ising lattice, in place.

    The Wolff algorithm adds a same-spin neighbor to the growing cluster with
    probability :math:`p_{\\text{add}} = 1 - e^{-2 \\beta J}`, then flips the
    entire cluster at once. It suppresses the critical slowing down that
    afflicts single-spin-flip Metropolis dynamics near :math:`T_c`.

    Parameters
    ----------
    spins : ndarray of shape (L, L), dtype int64
        Spin configuration with values in ``{-1, +1}``. Modified in place.
    beta : float
        Inverse temperature.
    J : float
        Ferromagnetic coupling constant (should be positive).

    Returns
    -------
    int
        Number of spins flipped in this cluster update.
    """
    L = spins.shape[0]
    p_add = 1.0 - np.exp(-2.0 * beta * J)

    stack_i = np.empty(L * L, dtype=np.int64)
    stack_j = np.empty(L * L, dtype=np.int64)
    top = 0

    i0 = np.random.randint(0, L)
    j0 = np.random.randint(0, L)
    cluster_spin = spins[i0, j0]

    in_cluster = np.zeros((L, L), dtype=np.bool_)
    in_cluster[i0, j0] = True
    spins[i0, j0] = -cluster_spin
    stack_i[top] = i0
    stack_j[top] = j0
    top += 1
    cluster_size = 1

    while top > 0:
        top -= 1
        i = stack_i[top]
        j = stack_j[top]
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni = (i + di) % L
            nj = (j + dj) % L
            if (not in_cluster[ni, nj]) and spins[ni, nj] == cluster_spin and np.random.random() < p_add:
                in_cluster[ni, nj] = True
                spins[ni, nj] = -cluster_spin
                stack_i[top] = ni
                stack_j[top] = nj
                top += 1
                cluster_size += 1
    return cluster_size


@njit(cache=True)
def metropolis_sweep_potts(spins, beta, J, q):
    """Perform one Metropolis-Hastings sweep of a q-state Potts lattice, in place.

    Parameters
    ----------
    spins : ndarray of shape (L, L), dtype int64
        Spin configuration with values in ``{0, ..., q-1}``. Modified in place.
    beta : float
        Inverse temperature.
    J : float
        Coupling constant. The bond energy is :math:`-J` when neighboring
        spins agree and ``0`` otherwise (Potts Hamiltonian).
    q : int
        Number of Potts states.

    Returns
    -------
    ndarray of shape (L, L)
        The same array passed in.
    """
    L = spins.shape[0]
    n = L * L
    for _ in range(n):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        s = spins[i, j]
        top = spins[(i - 1) % L, j]
        bottom = spins[(i + 1) % L, j]
        left = spins[i, (j - 1) % L]
        right = spins[i, (j + 1) % L]
        new_s = np.random.randint(0, q)
        old_matches = (s == top) + (s == bottom) + (s == left) + (s == right)
        new_matches = (new_s == top) + (new_s == bottom) + (new_s == left) + (new_s == right)
        dE = -J * (new_matches - old_matches)
        if dE <= 0.0 or np.random.random() < np.exp(-beta * dE):
            spins[i, j] = new_s
    return spins


@njit(cache=True)
def metropolis_sweep_xy(theta, beta, J, delta=1.0):
    """Perform one Metropolis-Hastings sweep of a 2D XY lattice, in place.

    Parameters
    ----------
    theta : ndarray of shape (L, L), dtype float64
        Planar spin angles in radians, :math:`\\theta_i \\in [0, 2\\pi)`.
        Modified in place.
    beta : float
        Inverse temperature.
    J : float
        Coupling constant. Bond energy is :math:`-J \\cos(\\theta_i - \\theta_j)`.
    delta : float, default=1.0
        Maximum angular step of a trial move, drawn uniformly from
        ``[-delta, delta]``.

    Returns
    -------
    ndarray of shape (L, L)
        The same array passed in.
    """
    L = theta.shape[0]
    n = L * L
    two_pi = 2.0 * np.pi
    for _ in range(n):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        old = theta[i, j]
        new = old + (2.0 * np.random.random() - 1.0) * delta

        top = theta[(i - 1) % L, j]
        bottom = theta[(i + 1) % L, j]
        left = theta[i, (j - 1) % L]
        right = theta[i, (j + 1) % L]

        old_energy = -J * (np.cos(old - top) + np.cos(old - bottom) + np.cos(old - left) + np.cos(old - right))
        new_energy = -J * (np.cos(new - top) + np.cos(new - bottom) + np.cos(new - left) + np.cos(new - right))
        dE = new_energy - old_energy
        if dE <= 0.0 or np.random.random() < np.exp(-beta * dE):
            spin = new % two_pi
            theta[i, j] = spin
    return theta


@njit(cache=True)
def ising_total_energy(spins, J):
    """Total Ising Hamiltonian :math:`E = -J \\sum_{\\langle i,j \\rangle} s_i s_j`.

    Each bond is counted once (only the right and bottom neighbor of every
    site), giving the physical extensive energy for a periodic :math:`L
    \\times L` lattice.
    """
    L = spins.shape[0]
    E = 0.0
    for i in range(L):
        for j in range(L):
            s = spins[i, j]
            right = spins[i, (j + 1) % L]
            bottom = spins[(i + 1) % L, j]
            E += -J * s * (right + bottom)
    return E


@njit(cache=True)
def ising_total_magnetization(spins):
    """Total magnetization :math:`M = \\sum_i s_i`."""
    return spins.sum()


@njit(cache=True)
def potts_total_energy(spins, J):
    """Total Potts Hamiltonian :math:`E = -J \\sum_{\\langle i,j \\rangle} \\delta(s_i, s_j)`."""
    L = spins.shape[0]
    E = 0.0
    for i in range(L):
        for j in range(L):
            s = spins[i, j]
            right = spins[i, (j + 1) % L]
            bottom = spins[(i + 1) % L, j]
            E += -J * ((s == right) + (s == bottom))
    return E


@njit(cache=True)
def xy_total_energy(theta, J):
    """Total XY Hamiltonian :math:`E = -J \\sum_{\\langle i,j \\rangle} \\cos(\\theta_i - \\theta_j)`."""
    L = theta.shape[0]
    E = 0.0
    for i in range(L):
        for j in range(L):
            t = theta[i, j]
            right = theta[i, (j + 1) % L]
            bottom = theta[(i + 1) % L, j]
            E += -J * (np.cos(t - right) + np.cos(t - bottom))
    return E


@njit(cache=True)
def xy_plaquette_vorticity(theta):
    """Topological charge of every elementary plaquette of an XY lattice.

    The vorticity of the plaquette with lower-left corner ``(i, j)`` is

    .. math::

        q_{ij} = \\frac{1}{2\\pi} \\oint \\nabla \\theta \\cdot d\\mathbf{l}
               = \\frac{1}{2\\pi} \\sum_{k} \\text{wrap}(\\theta_{k+1} - \\theta_k)

    summed counter-clockwise around the four bonds of the plaquette, where
    ``wrap`` maps angle differences into :math:`(-\\pi, \\pi]`. It is close to
    an integer (:math:`0` in the bulk, :math:`\\pm 1` at a vortex core).

    Parameters
    ----------
    theta : ndarray of shape (L, L), dtype float64
        Planar spin angles.

    Returns
    -------
    ndarray of shape (L, L), dtype float64
        Vorticity charge attached to each plaquette (periodic lattice, so
        every site indexes one plaquette).
    """
    L = theta.shape[0]
    q = np.zeros((L, L), dtype=np.float64)
    two_pi = 2.0 * np.pi
    for i in range(L):
        for j in range(L):
            t00 = theta[i, j]
            t10 = theta[i, (j + 1) % L]
            t11 = theta[(i + 1) % L, (j + 1) % L]
            t01 = theta[(i + 1) % L, j]
            total = 0.0
            for a, b in ((t00, t10), (t10, t11), (t11, t01), (t01, t00)):
                d = b - a
                d = d - two_pi * np.floor((d + np.pi) / two_pi)
                total += d
            q[i, j] = total / two_pi
    return q


@njit(cache=True)
def metropolis_sweep_spin_glass(spins, beta, J_right, J_down):
    """Perform one Metropolis-Hastings sweep of an Edwards-Anderson Ising spin glass, in place.

    Unlike :func:`metropolis_sweep_ising`, the bond coupling is a quenched,
    site-dependent random variable rather than a single constant: ``J_right[i,
    j]`` is the coupling between site ``(i, j)`` and its right neighbor
    ``(i, j+1)``, and ``J_down[i, j]`` is the coupling between ``(i, j)`` and
    its neighbor below, ``(i+1, j)`` (both periodic). A site's left and top
    bonds are read from its neighbor's ``J_right``/``J_down`` entries.

    Parameters
    ----------
    spins : ndarray of shape (L, L), dtype int64
        Spin configuration with values in ``{-1, +1}``. Modified in place.
    beta : float
        Inverse temperature.
    J_right : ndarray of shape (L, L), dtype float64
        Quenched rightward bond couplings.
    J_down : ndarray of shape (L, L), dtype float64
        Quenched downward bond couplings.

    Returns
    -------
    ndarray of shape (L, L)
        The same array passed in.
    """
    L = spins.shape[0]
    n = L * L
    for _ in range(n):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        s = spins[i, j]

        ip = (i + 1) % L
        im = (i - 1) % L
        jp = (j + 1) % L
        jm = (j - 1) % L

        dE = 2.0 * s * (J_right[i, j] * spins[i, jp] + J_right[i, jm] * spins[i, jm] + J_down[i, j] * spins[ip, j] + J_down[im, j] * spins[im, j])
        if dE <= 0.0 or np.random.random() < np.exp(-beta * dE):
            spins[i, j] = -s
    return spins


@njit(cache=True)
def spin_glass_total_energy(spins, J_right, J_down):
    """Total Edwards-Anderson Hamiltonian :math:`E = -\\sum_{\\langle i,j \\rangle} J_{ij} s_i s_j`.

    Each bond is counted once, using the same ``J_right``/``J_down``
    indexing convention as :func:`metropolis_sweep_spin_glass`.
    """
    L = spins.shape[0]
    E = 0.0
    for i in range(L):
        for j in range(L):
            s = spins[i, j]
            right = spins[i, (j + 1) % L]
            bottom = spins[(i + 1) % L, j]
            E += -J_right[i, j] * s * right - J_down[i, j] * s * bottom
    return E


@njit(cache=True)
def metropolis_sweep_sk(spins, beta, J):
    """Perform one Metropolis-Hastings sweep of a fully-connected Sherrington-Kirkpatrick spin glass, in place.

    Unlike the nearest-neighbor lattice models, every spin couples to every
    other spin: flipping site ``i`` requires the full local field
    :math:`\\sum_k J_{ik} s_k`, an :math:`O(N)` sum, so one sweep of
    :math:`N` attempted flips costs :math:`O(N^2)` -- the price of the
    model's infinite-range connectivity.

    Parameters
    ----------
    spins : ndarray of shape (N,), dtype int64
        Spin configuration with values in ``{-1, +1}``. Modified in place.
    beta : float
        Inverse temperature.
    J : ndarray of shape (N, N), dtype float64
        Quenched, symmetric coupling matrix with zero diagonal.

    Returns
    -------
    ndarray of shape (N,)
        The same array passed in.
    """
    N = spins.shape[0]
    for _ in range(N):
        i = np.random.randint(0, N)
        field = 0.0
        for k in range(N):
            field += J[i, k] * spins[k]
        dE = 2.0 * spins[i] * field
        if dE <= 0.0 or np.random.random() < np.exp(-beta * dE):
            spins[i] = -spins[i]
    return spins


@njit(cache=True)
def sk_total_energy(spins, J):
    """Total Sherrington-Kirkpatrick Hamiltonian :math:`E = -\\sum_{i<j} J_{ij} s_i s_j`.

    Parameters
    ----------
    spins : ndarray of shape (N,), dtype int64
        Spin configuration with values in ``{-1, +1}``.
    J : ndarray of shape (N, N), dtype float64
        Quenched, symmetric coupling matrix with zero diagonal.

    Returns
    -------
    float
    """
    N = spins.shape[0]
    total = 0.0
    for i in range(N):
        for k in range(N):
            total += J[i, k] * spins[i] * spins[k]
    return -0.5 * total
