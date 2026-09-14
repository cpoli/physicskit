"""The Husimi (coherent-state) phase-space representation of a state on a torus.

Shared by :class:`~physicskit.chaos.quantum.maps.QuantumKickedRotor` and
:class:`~physicskit.chaos.quantum.maps.QuantumBakersMap`, whose Hilbert spaces are both
built on a position basis evenly spaced around a periodic ``[0, q_period)``
domain -- the Husimi construction below only depends on that shared structure,
not on which map produced the state.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def husimi_function(
    psi: NDArray[np.complexfloating],
    hbar: float,
    q_period: float,
    resolution: int = 80,
    n_wraps: int = 3,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Husimi (coherent-state overlap) phase-space distribution of a state.

    The Husimi function ``Q(q0, p0) = |<q0, p0|psi>|^2`` is the state's
    overlap with minimum-uncertainty coherent states centered at every point
    ``(q0, p0)`` of phase space -- the standard way to see a quantum state
    "on top of" classical phase space (e.g. a classical Poincare section or
    a classical map's orbits), since it is everywhere non-negative and
    smoothly spread out, unlike the (generally negative-valued) Wigner
    function.

    Both :class:`~physicskit.chaos.quantum.maps.QuantumKickedRotor` and
    :class:`~physicskit.chaos.quantum.maps.QuantumBakersMap` live on a position
    basis of `N` points evenly spaced around a periodic domain
    ``[0, q_period)`` (``q_period = 2*pi`` for the kicked rotor's angle,
    ``q_period = 1`` for the baker's map's unit interval); the coherent
    states used here are periodized Gaussians of width ``sqrt(hbar *
    q_period / (2*pi))`` on that same domain, so this one function serves
    both.

    Parameters
    ----------
    psi : ndarray of complex, shape (N,)
        State vector in the position representation (`N` basis points
        evenly spaced over ``[0, q_period)``); need not be normalized.
    hbar : float
        Effective Planck constant of the underlying quantum map.
    q_period : float
        Length of the (periodic) position and momentum domain, e.g.
        ``2*pi`` for the kicked rotor or ``1.0`` for the baker's map.
    resolution : int, default 80
        Number of grid points along each of the ``q`` and ``p`` axes.
    n_wraps : int, default 3
        Number of periodic images summed on each side to periodize the
        coherent state around the torus; 3 is already accurate to
        essentially machine precision for the Gaussian widths used here.

    Returns
    -------
    Q0, P0 : ndarray of float, shape (resolution, resolution)
        Phase-space grid (as from ``np.meshgrid(..., indexing="ij")``).
    husimi : ndarray of float, shape (resolution, resolution)
        The Husimi distribution, normalized to a peak value of 1 (this is a
        relative-intensity plotting convenience, not a probability density
        normalized to unit integral).
    """
    psi = np.asarray(psi, dtype=np.complex128)
    n = psi.shape[0]
    q_j = np.arange(n) * (q_period / n)
    sigma2 = hbar * q_period / (2.0 * np.pi)

    q0 = np.linspace(0.0, q_period, resolution, endpoint=False)
    p0 = np.linspace(0.0, q_period, resolution, endpoint=False)
    Q0, P0 = np.meshgrid(q0, p0, indexing="ij")

    overlap = np.zeros(Q0.shape, dtype=np.complex128)
    for m in range(-n_wraps, n_wraps + 1):
        dq = q_j[None, None, :] - Q0[:, :, None] - m * q_period
        coherent = np.exp(-(dq**2) / (2.0 * sigma2)) * np.exp(1j * P0[:, :, None] * dq / hbar)
        overlap += np.tensordot(np.conj(coherent), psi, axes=([2], [0]))

    husimi = np.abs(overlap) ** 2
    husimi /= husimi.max()
    return Q0, P0, husimi
