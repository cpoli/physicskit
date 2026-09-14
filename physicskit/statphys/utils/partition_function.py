"""Exact partition-function enumeration and the Lee-Yang circle theorem.

Lee and Yang (1952) asked where a phase transition -- a genuine
discontinuity or divergence in the free energy -- could possibly come from,
given that a finite system's partition function is a finite sum of
manifestly analytic terms. Their answer: it comes from the zeros of the
partition function, analytically continued into the *complex* fugacity
plane. For a finite ferromagnetic Ising system in a field, every such zero
lies exactly on the unit circle (the "Lee-Yang circle theorem"); as the
system size grows, these zeros multiply and crowd toward the positive real
axis, and a phase transition occurs exactly where, in the thermodynamic
limit, they pinch that axis. No finite system has a true singularity, but
the zeros' approach to one is visible well before that limit.

This module computes the partition function of a small Ising system
*exactly*, by full enumeration of every spin configuration -- feasible up to roughly
:math:`N \\approx 20` (:math:`2^N` configurations are enumerated
explicitly) -- as a polynomial in the fugacity, and finds
that polynomial's roots directly.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ising_partition_polynomial", "yang_lee_zeros"]


def ising_partition_polynomial(N, beta, J=1.0, periodic=True):
    """Exact partition function of a small 1D Ising chain, as a polynomial in the fugacity.

    With an external field :math:`h`, the Hamiltonian is :math:`H =
    -J\\sum_{\\langle i,j\\rangle} s_i s_j - h \\sum_i s_i`. Writing the
    fugacity :math:`z = e^{2\\beta h}`, the partition function factors as

    .. math::

        Z(\\beta, h) = e^{-\\beta h N} \\sum_{k=0}^{N} g_k\\, z^k,

    where :math:`g_k = \\sum_{\\{s\\}:\\, n_\\uparrow(s) = k}
    e^{-\\beta H_0(s)}` sums the field-free Boltzmann weight over every
    configuration with exactly :math:`k` up-spins. The prefactor
    :math:`e^{-\\beta h N}` is nonzero and analytic in :math:`h`, so it
    never contributes a zero: the zeros of :math:`Z` as a function of
    :math:`z` are exactly the roots of this polynomial's coefficients
    ``g_k``, which is all :func:`yang_lee_zeros` needs.

    Every one of the :math:`2^N` configurations is enumerated directly, so
    this is only practical for modest :math:`N` (roughly 20 at most --
    ``2**30`` configurations will exhaust memory).

    Parameters
    ----------
    N : int
        Number of spins.
    beta : float
        Inverse temperature.
    J : float, default=1.0
        Nearest-neighbor coupling. ``J > 0`` (ferromagnetic) is the regime
        the Lee-Yang circle theorem applies to.
    periodic : bool, default=True
        Periodic (ring) vs. open chain boundary conditions.

    Returns
    -------
    ndarray of shape (N + 1,)
        Coefficients ``g_k``, ordered by increasing power of ``z`` (i.e.
        ``g[k]`` is the coefficient of ``z**k``).
    """
    configs = np.arange(2**N, dtype=np.int64)
    bits = (configs[:, None] >> np.arange(N)) & 1
    spins = bits.astype(np.float64) * 2 - 1  # shape (2**N, N), values +/-1

    if periodic:
        neighbor_product = spins * np.roll(spins, -1, axis=1)
    else:
        neighbor_product = spins[:, :-1] * spins[:, 1:]
    E0 = -J * np.sum(neighbor_product, axis=1)

    n_up = np.sum(bits, axis=1)
    weights = np.exp(-beta * E0)

    g = np.zeros(N + 1, dtype=np.float64)
    np.add.at(g, n_up, weights)
    return g


def yang_lee_zeros(coeffs):
    """Roots, in the complex fugacity plane, of the polynomial :math:`\\sum_k g_k z^k`.

    Lee and Yang's circle theorem predicts that for a ferromagnetic Ising
    system (``J > 0`` in :func:`ising_partition_polynomial`), every one of
    these roots lies exactly on the unit circle :math:`|z| = 1`, regardless
    of system size -- a striking, exact constraint on where a phase
    transition's mathematical signature is allowed to live.

    Parameters
    ----------
    coeffs : array_like
        Polynomial coefficients ``g_k``, ordered by increasing power of
        ``z`` (as returned by :func:`ising_partition_polynomial`).

    Returns
    -------
    ndarray of complex
        The polynomial's roots.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    return np.roots(coeffs[::-1])
