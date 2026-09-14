r"""Weyl semimetals (2011-2015): momentum-space monopoles and Fermi arcs.

Wan, Turner, Vishwanath, and Savrasov predicted that breaking either
inversion or time-reversal symmetry in a 3D Dirac material splits each
doubly-degenerate Dirac point into a pair of nondegenerate Weyl nodes of
opposite chirality -- momentum-space sources and sinks of Berry curvature,
each the direct 3D generalization of the 2D TKNN invariant
(:func:`physicskit.condensed.topology.compute_chern_number`) evaluated on a
small sphere surrounding it. Xu, Lv, and coworkers confirmed the prediction
by ARPES in TaAs in 2015, directly imaging both the bulk Weyl nodes and the
open, non-closed "Fermi arcs" of surface states connecting the surface
projections of opposite-chirality nodes -- a bulk-boundary signature with
no two-dimensional analogue.

This module implements the minimal two-band lattice model with this
physics, a cubic-lattice generalization of the same Dirac-matrix
construction used for the 3D topological insulator
(:mod:`physicskit.condensed.topological_insulator_3d`), reduced from four
bands to two since a genuine (as opposed to symmetry-protected) Weyl node
only needs a two-level crossing:

.. math::

   H(\mathbf{k}) = \sin k_x\,\sigma_x + \sin k_y\,\sigma_y +
   \left(m - t\sum_{i=x,y,z}\cos k_i\right)\sigma_z .

For :math:`1 < m/t < 3` the mass term vanishes at exactly two points on the
:math:`k_z` axis, :math:`\mathbf{k}=(0,0,\pm k_0)` with
:math:`\cos k_0 = m/t - 2` -- a pair of Weyl nodes of opposite chirality.
Between them, each fixed-:math:`k_z` slice is a 2D Chern insulator; outside
them, it is trivial. Opening the lattice along ``x`` exposes exactly the
surface signature of that alternation: a single chiral edge mode crossing
zero energy for :math:`|k_z| < k_0`, and a full gap for :math:`|k_z| > k_0`
-- the lattice-model fingerprint of the Fermi arc.

Uses the same natural-unit convention (:math:`\hbar=e=1`) as the rest of
:mod:`physicskit.condensed`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["weyl_semimetal_hamiltonian", "weyl_node_locations", "weyl_semimetal_slab_hamiltonian"]

_SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
_SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def weyl_semimetal_hamiltonian(kx: float, ky: float, kz: float, m: float, t: float = 1.0) -> np.ndarray:
    r"""Bloch Hamiltonian of the minimal two-band cubic-lattice Weyl semimetal.

    Parameters
    ----------
    kx, ky, kz : float
        Reduced crystal momenta, each periodic on :math:`[0, 2\pi)`.
    m : float
        Band mass. Two Weyl nodes exist on the :math:`k_z` axis for
        :math:`1 < m/t < 3` (see :func:`weyl_node_locations`); outside that
        window the model is fully gapped.
    t : float, default=1.0
        Hopping amplitude setting the bulk bandwidth.

    Returns
    -------
    ndarray, shape (2, 2)
        Hermitian Bloch Hamiltonian.

    See Also
    --------
    weyl_node_locations : Exact :math:`k_z` positions of the two Weyl nodes.
    weyl_semimetal_slab_hamiltonian : Open-boundary slab exposing Fermi-arc surface states.

    Examples
    --------
    At ``m/t = 2`` the two nodes sit at :math:`k_z = \pm\pi/2`, where the
    bulk gap closes exactly:

    >>> import numpy as np
    >>> H = weyl_semimetal_hamiltonian(0.0, 0.0, np.pi / 2, m=2.0, t=1.0)
    >>> np.round(np.linalg.eigvalsh(H), 8)
    array([0., 0.])
    >>> H_away = weyl_semimetal_hamiltonian(np.pi, np.pi, np.pi, m=2.0, t=1.0)
    >>> np.round(np.linalg.eigvalsh(H_away), 8)
    array([-5.,  5.])
    """
    mass = m - t * (np.cos(kx) + np.cos(ky) + np.cos(kz))
    return np.sin(kx) * _SIGMA_X + np.sin(ky) * _SIGMA_Y + mass * _SIGMA_Z


def weyl_node_locations(m: float, t: float = 1.0) -> tuple:
    r"""Exact :math:`k_z` positions of the two Weyl nodes on the :math:`k_z` axis.

    Solves :math:`m - t(2 + \cos k_z) = 0` -- the mass term of
    :func:`weyl_semimetal_hamiltonian` at :math:`k_x=k_y=0` -- for
    :math:`k_z`.

    Parameters
    ----------
    m : float
        Band mass.
    t : float, default=1.0
        Hopping amplitude.

    Returns
    -------
    tuple of float or None
        ``(-k0, +k0)`` with :math:`k_0 \in (0, \pi)`, or ``None`` if
        :math:`m/t` lies outside :math:`(1, 3)` (no nodes on this axis).

    Examples
    --------
    >>> import numpy as np
    >>> k0 = weyl_node_locations(m=2.0, t=1.0)
    >>> np.round(k0, 8)
    array([-1.57079633,  1.57079633])
    >>> weyl_node_locations(m=4.0, t=1.0) is None
    True
    """
    ratio = m / t - 2.0
    if not (-1.0 < ratio < 1.0):
        return None
    k0 = float(np.arccos(ratio))
    return np.array([-k0, k0])


def weyl_semimetal_slab_hamiltonian(ky: float, kz: float, n_layers: int, m: float, t: float = 1.0) -> np.ndarray:
    r"""Slab Hamiltonian: periodic in ``y`` and ``z``, open (finite) along ``x``.

    Truncating the ``x`` direction exposes two open surfaces whose surface
    Brillouin zone is the :math:`(k_y, k_z)` plane the two Weyl nodes
    project onto at :math:`(0, \pm k_0)`. Diagonalizing at fixed
    :math:`k_z` and scanning :math:`k_y` reveals, for
    :math:`|k_z| < k_0`, a single chiral mode crossing zero energy
    (the Fermi arc), and a full gap for :math:`|k_z| > k_0`.

    Parameters
    ----------
    ky, kz : float
        Reduced crystal momenta, periodic on :math:`[0, 2\pi)`.
    n_layers : int
        Number of layers stacked along ``x``.
    m : float
        Band mass (see :func:`weyl_semimetal_hamiltonian`).
    t : float, default=1.0
        Hopping amplitude.

    Returns
    -------
    ndarray, shape (2 * n_layers, 2 * n_layers)
        Hermitian, open-boundary slab Hamiltonian.

    Examples
    --------
    Between the two Weyl nodes (:math:`k_z=0`), a thick slab has a state
    pinned near zero energy at some :math:`k_y` -- the Fermi-arc surface
    mode -- absent outside the node range (:math:`k_z=\pi`):

    >>> import numpy as np
    >>> ky_grid = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    >>> gaps_inside = [np.min(np.abs(np.linalg.eigvalsh(weyl_semimetal_slab_hamiltonian(ky, 0.0, 40, m=2.0)))) for ky in ky_grid]
    >>> bool(min(gaps_inside) < 1e-6)
    True
    >>> gaps_outside = [np.min(np.abs(np.linalg.eigvalsh(weyl_semimetal_slab_hamiltonian(ky, np.pi, 40, m=2.0)))) for ky in ky_grid]
    >>> bool(min(gaps_outside) > 0.5)
    True
    """
    mass_yz = m - t * (np.cos(ky) + np.cos(kz))
    H0 = np.sin(ky) * _SIGMA_Y + mass_yz * _SIGMA_Z
    Tx = -0.5j * _SIGMA_X - 0.5 * t * _SIGMA_Z

    N = n_layers
    H = np.zeros((2 * N, 2 * N), dtype=complex)
    for layer in range(N):
        H[2 * layer : 2 * layer + 2, 2 * layer : 2 * layer + 2] = H0
    for layer in range(N - 1):
        H[2 * layer : 2 * layer + 2, 2 * (layer + 1) : 2 * (layer + 1) + 2] = Tx
        H[2 * (layer + 1) : 2 * (layer + 1) + 2, 2 * layer : 2 * layer + 2] = Tx.conj().T
    return H
