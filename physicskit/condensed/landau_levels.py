r"""Landau levels: the 1930 quantization of a charged particle in a uniform magnetic field.

Lev Landau showed that the classically continuous cyclotron motion of a
charged particle in a uniform magnetic field :math:`B` quantizes into
discrete, macroscopically degenerate levels

.. math::

   E_n = \hbar\omega_c\left(n+\tfrac12\right), \qquad \omega_c = \frac{eB}{m},
   \qquad n = 0, 1, 2, \dots,

exactly the harmonic-oscillator spectrum, with :math:`\omega_c` set by the
field rather than a spring constant. Each level's macroscopic degeneracy per
unit area, :math:`n_B = 1/(2\pi \ell_B^2)` with magnetic length
:math:`\ell_B = \sqrt{\hbar/(eB)}`, is the microscopic origin of orbital
(Landau) diamagnetism, and -- once a 2D electron gas and disorder-broadened
levels are added -- the direct ancestor of the integer quantum Hall effect
(:mod:`physicskit.condensed.topology`) fifty years later.

Uses the same natural-unit convention (:math:`\hbar=e=1`) as the rest of
:mod:`physicskit.condensed`; pass ``m``, ``e``, and ``hbar`` explicitly to
work in physical units instead. See
:func:`physicskit.condensed.tight_binding.apply_peierls_phase` for the
lattice (Peierls-substitution) route to the same physics.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "cyclotron_frequency",
    "magnetic_length",
    "landau_level_energies",
    "landau_degeneracy",
    "landau_density_of_states",
    "filling_factor",
]


def cyclotron_frequency(B: float, m: float = 1.0, e: float = 1.0) -> float:
    r"""Cyclotron frequency :math:`\omega_c = eB/m`.

    Parameters
    ----------
    B : float
        Magnetic field strength.
    m : float, default=1.0
        Particle mass.
    e : float, default=1.0
        Particle charge magnitude.

    Returns
    -------
    float

    Examples
    --------
    >>> cyclotron_frequency(B=2.0, m=0.5)
    4.0
    """
    return e * B / m


def magnetic_length(B: float, hbar: float = 1.0, e: float = 1.0) -> float:
    r"""Magnetic length :math:`\ell_B = \sqrt{\hbar/(eB)}`.

    The natural length scale of the lowest Landau level's cyclotron orbit.

    Parameters
    ----------
    B : float
        Magnetic field strength.
    hbar : float, default=1.0
        Reduced Planck constant.
    e : float, default=1.0
        Particle charge magnitude.

    Returns
    -------
    float

    Examples
    --------
    >>> magnetic_length(B=1.0)
    1.0
    >>> round(magnetic_length(B=4.0), 4)
    0.5
    """
    return float(np.sqrt(hbar / (e * B)))


def landau_level_energies(n_max: int, B: float, m: float = 1.0, e: float = 1.0, hbar: float = 1.0) -> np.ndarray:
    r"""Landau energy levels :math:`E_n = \hbar\omega_c(n+\tfrac12)` for :math:`n=0,\dots,n_{max}`.

    Parameters
    ----------
    n_max : int
        Highest Landau index to include.
    B : float
        Magnetic field strength.
    m : float, default=1.0
        Particle mass.
    e : float, default=1.0
        Particle charge magnitude.
    hbar : float, default=1.0
        Reduced Planck constant.

    Returns
    -------
    ndarray, shape (n_max + 1,)
        Equally spaced energies :math:`E_0 < E_1 < \dots < E_{n_{max}}`,
        spacing :math:`\hbar\omega_c`.

    See Also
    --------
    landau_degeneracy : Number of degenerate single-particle states per level.

    Examples
    --------
    >>> landau_level_energies(n_max=2, B=1.0)
    array([0.5, 1.5, 2.5])
    """
    omega_c = cyclotron_frequency(B, m, e)
    n = np.arange(n_max + 1)
    return hbar * omega_c * (n + 0.5)


def landau_degeneracy(area: float, B: float, hbar: float = 1.0, e: float = 1.0) -> float:
    r"""Number of degenerate single-particle states per Landau level.

    Each Landau level holds :math:`n_B \cdot \text{area}` states, where
    :math:`n_B = 1/(2\pi\ell_B^2) = eB/(2\pi\hbar) = B/\Phi_0` is one state
    per flux quantum :math:`\Phi_0 = 2\pi\hbar/e` threading the sample --
    the origin of the macroscopic degeneracy behind Landau diamagnetism.

    Parameters
    ----------
    area : float
        Real-space sample area.
    B : float
        Magnetic field strength.
    hbar : float, default=1.0
        Reduced Planck constant.
    e : float, default=1.0
        Particle charge magnitude.

    Returns
    -------
    float

    Examples
    --------
    >>> landau_degeneracy(area=2 * np.pi, B=1.0)
    1.0
    """
    l_B = magnetic_length(B, hbar, e)
    return float(area / (2 * np.pi * l_B**2))


def landau_density_of_states(
    energies,
    B: float,
    m: float = 1.0,
    e: float = 1.0,
    hbar: float = 1.0,
    n_max: int = 20,
    broadening: float = 0.05,
) -> np.ndarray:
    r"""Disorder-broadened density of states per unit area.

    Replaces each infinitely sharp level
    :math:`n_B\,\delta(E - E_n)` with a Gaussian of width ``broadening``,
    the standard phenomenological model for the disorder- or
    finite-lifetime-broadened Landau levels seen in a real 2D electron gas
    (and needed to have a finite conductivity between the exactly quantized
    Hall plateaus).

    Parameters
    ----------
    energies : array_like
        Energies at which to evaluate the density of states.
    B : float
        Magnetic field strength.
    m : float, default=1.0
        Particle mass.
    e : float, default=1.0
        Particle charge magnitude.
    hbar : float, default=1.0
        Reduced Planck constant.
    n_max : int, default=20
        Highest Landau index included in the sum.
    broadening : float, default=0.05
        Gaussian standard deviation of each broadened level.

    Returns
    -------
    ndarray
        Density of states per unit area, same shape as ``energies``.

    Examples
    --------
    >>> dos = landau_density_of_states([0.5], B=1.0, n_max=0, broadening=0.1)
    >>> round(float(dos[0]), 4)
    0.6349
    """
    energies = np.atleast_1d(np.asarray(energies, dtype=float))
    E_n = landau_level_energies(n_max, B, m, e, hbar)
    per_level = landau_degeneracy(1.0, B, hbar, e)
    norm = 1.0 / (broadening * np.sqrt(2 * np.pi))
    dos = np.zeros_like(energies)
    for En in E_n:
        dos += per_level * norm * np.exp(-0.5 * ((energies - En) / broadening) ** 2)
    return dos


def filling_factor(density: float, B: float, hbar: float = 1.0, e: float = 1.0) -> float:
    r"""Landau-level filling factor :math:`\nu = n_e / n_B = n_e h/(eB)`.

    The number of filled Landau levels (generally non-integer) for a 2D
    electron density ``density``. Integer :math:`\nu` is the condition for
    an incompressible quantum Hall plateau in
    :mod:`physicskit.condensed.topology`.

    Parameters
    ----------
    density : float
        2D electron number density (particles per unit area).
    B : float
        Magnetic field strength.
    hbar : float, default=1.0
        Reduced Planck constant.
    e : float, default=1.0
        Particle charge magnitude.

    Returns
    -------
    float

    Examples
    --------
    >>> round(filling_factor(density=2.0, B=1.0), 6)
    12.566371
    """
    l_B = magnetic_length(B, hbar, e)
    return float(density * 2 * np.pi * l_B**2)
