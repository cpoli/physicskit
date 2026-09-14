r"""The 3D strong topological insulator: from theory (2007) to Bi2Se3/Bi2Te3 (2008-2009).

Fu, Kane, and Mele extended the 2D :math:`\mathbb{Z}_2` classification of
:mod:`physicskit.condensed.models` (``kane_mele_hamiltonian``,
``bhz_hamiltonian``) to three dimensions, predicting a "strong" topological
insulator: a bulk band insulator whose *every* surface, regardless of
orientation, hosts a single, gapless, spin-momentum-locked Dirac cone
protected by time-reversal symmetry alone. Hasan and Cava's groups and
collaborators confirmed this directly in 2008-2009 by ARPES on
Bi\ :sub:`2`\ Se\ :sub:`3` and Bi\ :sub:`2`\ Te\ :sub:`3`, resolving exactly
one Dirac cone at the surface Brillouin zone center -- the 3D generalization
of the helical edge states Konig et al. had just observed at the 1D edge of
a 2D quantum spin Hall system.

This module implements the minimal 4-band lattice model with this physics
(Qi & Zhang, Rev. Mod. Phys. 83, 1057 (2011)):

.. math::

   H(\mathbf{k}) = \sin k_x\,\Gamma_1 + \sin k_y\,\Gamma_2 + \sin k_z\,\Gamma_3
   + \left(m + t\sum_{i=x,y,z}\cos k_i\right)\Gamma_0,

built from mutually anticommuting 4x4 Dirac matrices
:math:`\Gamma_0=\tau_z\otimes\sigma_0`, :math:`\Gamma_{1,2,3}=\tau_x\otimes\sigma_{x,y,z}`
(:math:`\tau` an orbital pseudospin, :math:`\sigma` the physical electron
spin) -- a cubic-lattice generalization of the BHZ model whose bulk gap
closes, and topological character changes, at every time-reversal-invariant
momentum where the mass term vanishes, :math:`m/t \in \{-3, -1, 1, 3\}`. Of
the resulting windows, :math:`1 < |m/t| < 3` is a strong topological
insulator (an open slab hosts a single, gapless surface Dirac cone at
:math:`\bar\Gamma=(0,0)` on each of its two open surfaces); :math:`|m/t| <
1` and :math:`|m/t| > 3` are trivial.

Uses the same natural-unit convention (:math:`\hbar=e=1`) as the rest of
:mod:`physicskit.condensed`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["topological_insulator_3d_hamiltonian", "topological_insulator_3d_slab_hamiltonian", "surface_dirac_hamiltonian"]

_sigma_0 = np.eye(2, dtype=complex)
_sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
_sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
_tau_x = _sigma_x
_tau_z = _sigma_z

_GAMMA0 = np.kron(_tau_z, _sigma_0)
_GAMMA1 = np.kron(_tau_x, _sigma_x)
_GAMMA2 = np.kron(_tau_x, _sigma_y)
_GAMMA3 = np.kron(_tau_x, _sigma_z)


def topological_insulator_3d_hamiltonian(kx: float, ky: float, kz: float, m: float, t: float = 1.0) -> np.ndarray:
    r"""Bloch Hamiltonian of the minimal cubic-lattice 3D topological insulator model.

    Parameters
    ----------
    kx, ky, kz : float
        Reduced crystal momenta, each periodic on :math:`[0, 2\pi)`.
    m : float
        Band mass. The model is a strong topological insulator for
        :math:`1 < |m/t| < 3`, and trivial for :math:`|m/t| < 1` or
        :math:`|m/t| > 3`.
    t : float, default=1.0
        Hopping amplitude setting the bulk bandwidth.

    Returns
    -------
    ndarray, shape (4, 4)
        Hermitian Bloch Hamiltonian.

    Examples
    --------
    >>> import numpy as np
    >>> H = topological_insulator_3d_hamiltonian(0.3, -0.7, 0.5, m=1.0)
    >>> np.allclose(H, H.conj().T)
    True
    >>> H0 = topological_insulator_3d_hamiltonian(0.0, 0.0, 0.0, m=1.0, t=1.0)
    >>> np.round(np.linalg.eigvalsh(H0), 8)
    array([-4., -4.,  4.,  4.])
    """
    mass = m + t * (np.cos(kx) + np.cos(ky) + np.cos(kz))
    return np.sin(kx) * _GAMMA1 + np.sin(ky) * _GAMMA2 + np.sin(kz) * _GAMMA3 + mass * _GAMMA0


def topological_insulator_3d_slab_hamiltonian(kx: float, ky: float, n_layers: int, m: float, t: float = 1.0) -> np.ndarray:
    r"""Slab Hamiltonian: periodic in ``x``, ``y``, open (finite) along ``z``.

    Truncating the ``z`` direction exposes the top and bottom (001)
    surfaces. Diagonalizing at each ``(kx, ky)`` and looking for
    mid-gap states localized at the outer layers reveals the surface
    Dirac cone directly, with no separate topological-invariant
    calculation needed.

    Parameters
    ----------
    kx, ky : float
        Reduced crystal momenta, periodic on :math:`[0, 2\pi)`.
    n_layers : int
        Number of atomic layers stacked along ``z``.
    m : float
        Band mass (see :func:`topological_insulator_3d_hamiltonian`).
    t : float, default=1.0
        Hopping amplitude.

    Returns
    -------
    ndarray, shape (4 * n_layers, 4 * n_layers)
        Hermitian, open-boundary slab Hamiltonian.

    Examples
    --------
    In the strong-topological-insulator regime (:math:`1 < |m/t| < 3`), a
    thick slab has a state pinned to exactly zero energy at the surface
    Brillouin zone center, localized entirely on one outer layer -- the
    surface Dirac point -- that a topologically trivial slab
    (:math:`|m/t| < 1`) lacks:

    >>> import numpy as np
    >>> gap = np.linalg.eigvalsh(topological_insulator_3d_slab_hamiltonian(0.0, 0.0, n_layers=40, m=-2.0))
    >>> bool(np.min(np.abs(gap)) < 1e-8)
    True
    >>> gap_trivial = np.linalg.eigvalsh(topological_insulator_3d_slab_hamiltonian(0.0, 0.0, n_layers=40, m=0.0))
    >>> bool(np.min(np.abs(gap_trivial)) > 0.5)
    True
    """
    mass_xy = m + t * (np.cos(kx) + np.cos(ky))
    H0 = np.sin(kx) * _GAMMA1 + np.sin(ky) * _GAMMA2 + mass_xy * _GAMMA0
    Tz = 0.5 * t * _GAMMA0 - 0.5j * _GAMMA3

    N = n_layers
    H = np.zeros((4 * N, 4 * N), dtype=complex)
    for layer in range(N):
        H[4 * layer : 4 * layer + 4, 4 * layer : 4 * layer + 4] = H0
    for layer in range(N - 1):
        H[4 * layer : 4 * layer + 4, 4 * (layer + 1) : 4 * (layer + 1) + 4] = Tz
        H[4 * (layer + 1) : 4 * (layer + 1) + 4, 4 * layer : 4 * layer + 4] = Tz.conj().T
    return H


def surface_dirac_hamiltonian(kx: float, ky: float, v_f: float = 1.0) -> np.ndarray:
    r"""Low-energy effective Hamiltonian of a single topological-insulator surface Dirac cone.

    .. math::

       H_{\text{surf}}(\mathbf{k}) = \hbar v_F\left(k_x\sigma_y - k_y\sigma_x\right),

    the massless, spin-momentum-locked Dirac fermion measured directly by
    ARPES on the Bi\ :sub:`2`\ Se\ :sub:`3` and Bi\ :sub:`2`\ Te\ :sub:`3`
    surfaces in 2008-2009: spin polarization locked perpendicular to
    momentum, with no Kramers-degenerate partner at the same energy and
    momentum (backscattering off nonmagnetic disorder is forbidden).

    Parameters
    ----------
    kx, ky : float
        Momentum measured from the surface Dirac point.
    v_f : float, default=1.0
        Surface Fermi velocity.

    Returns
    -------
    ndarray, shape (2, 2)
        Hermitian Hamiltonian with linear (Dirac) spectrum
        :math:`E_\pm(\mathbf{k}) = \pm v_F|\mathbf{k}|`.

    Examples
    --------
    >>> import numpy as np
    >>> H = surface_dirac_hamiltonian(0.3, -0.4, v_f=2.0)
    >>> np.round(np.linalg.eigvalsh(H), 8)
    array([-1.,  1.])
    """
    return v_f * (kx * _sigma_y - ky * _sigma_x)
