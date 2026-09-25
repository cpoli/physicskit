"""Physical constants shared across physicskit, in SI units.

Every subpackage in physicskit computes in whatever unit convention is
natural for its own domain -- ``astro`` uses gravitational units with
:math:`G = 1`, ``relativity`` uses geometrized units with :math:`G = c =
1`, the ``quantum`` family sets :math:`\\hbar = 1`, and ``statphys`` writes
entropy and free energy in units of :math:`k_B`. Each of those choices is
documented in the relevant subpackage and is the right call for a
self-contained chapter.

This module is the single place those subpackages (and calling code) can
get the *actual* SI value of a constant, so that "G" or "hbar" or "k_B"
means the same number everywhere in physicskit rather than being
independently retyped -- and potentially drifting -- in each subpackage.
Values are taken from :mod:`scipy.constants` (2019 SI redefinition /
CODATA), which is already a physicskit dependency.

A handful of conversion helpers are also provided, but only for
conversions that are well-defined without extra domain choices: energy
<-> temperature (via :math:`k_B`) and energy <-> electronvolts, plus a
helper for going from an :math:`(L_0, M_0)` length/mass scale choice to
the derived time and velocity scale of a gravitational (:math:`G = 1`)
unit system, e.g. as used in :mod:`physicskit.astro`. Converting a
:math:`\\hbar = 1` quantum-chapter result to SI additionally requires a
choice of mass or length scale specific to that chapter, so no generic
helper is provided for it -- see the chapter's own docstring instead.
"""

from __future__ import annotations

import scipy.constants as _sc

__all__ = [
    "C",
    "G",
    "H",
    "HBAR",
    "K_B",
    "ELEMENTARY_CHARGE",
    "ELECTRON_MASS",
    "PROTON_MASS",
    "NEUTRON_MASS",
    "AVOGADRO",
    "GAS_CONSTANT",
    "VACUUM_PERMITTIVITY",
    "VACUUM_PERMEABILITY",
    "STEFAN_BOLTZMANN",
    "ELECTRONVOLT",
    "SOLAR_MASS_KG",
    "ASTRONOMICAL_UNIT_M",
    "PARSEC_M",
    "energy_to_temperature",
    "temperature_to_energy",
    "joules_to_ev",
    "ev_to_joules",
    "gravitational_unit_system",
]

#: Speed of light in vacuum, in m/s (exact by SI definition).
C = _sc.c

#: Newton's gravitational constant, in m^3 kg^-1 s^-2.
G = _sc.G

#: Planck constant, in J*s (exact by SI definition).
H = _sc.h

#: Reduced Planck constant, in J*s.
HBAR = _sc.hbar

#: Boltzmann constant, in J/K (exact by SI definition).
K_B = _sc.k

#: Elementary charge, in C (exact by SI definition).
ELEMENTARY_CHARGE = _sc.e

#: Electron rest mass, in kg.
ELECTRON_MASS = _sc.m_e

#: Proton rest mass, in kg.
PROTON_MASS = _sc.m_p

#: Neutron rest mass, in kg.
NEUTRON_MASS = _sc.m_n

#: Avogadro constant, in mol^-1 (exact by SI definition).
AVOGADRO = _sc.N_A

#: Molar gas constant, in J mol^-1 K^-1 -- exactly ``AVOGADRO * K_B`` by SI
#: definition. Computed rather than taken from ``scipy.constants.R``, which
#: older SciPy releases store truncated (8.314462618).
GAS_CONSTANT = _sc.N_A * _sc.k

#: Vacuum electric permittivity, in F/m.
VACUUM_PERMITTIVITY = _sc.epsilon_0

#: Vacuum magnetic permeability, in N/A^2.
VACUUM_PERMEABILITY = _sc.mu_0

#: Stefan-Boltzmann constant, in W m^-2 K^-4.
STEFAN_BOLTZMANN = _sc.Stefan_Boltzmann

#: One electronvolt, in J.
ELECTRONVOLT = _sc.eV

#: Solar mass, in kg.
SOLAR_MASS_KG = 1.988_47e30

#: One astronomical unit, in m.
ASTRONOMICAL_UNIT_M = _sc.au

#: One parsec, in m.
PARSEC_M = _sc.parsec


def energy_to_temperature(energy_j):
    """Convert an energy (in J) to the temperature (in K) with :math:`E = k_B T`.

    Examples
    --------
    >>> round(float(energy_to_temperature(K_B)), 6)
    1.0
    """
    return energy_j / K_B


def temperature_to_energy(temperature_k):
    """Convert a temperature (in K) to an energy (in J) with :math:`E = k_B T`.

    Examples
    --------
    >>> temperature_to_energy(1.0) == K_B
    True
    """
    return temperature_k * K_B


def joules_to_ev(energy_j):
    """Convert an energy from joules to electronvolts."""
    return energy_j / ELECTRONVOLT


def ev_to_joules(energy_ev):
    """Convert an energy from electronvolts to joules."""
    return energy_ev * ELECTRONVOLT


def gravitational_unit_system(length_m, mass_kg):
    """Derive the time and velocity scale of a :math:`G = 1` unit system.

    Packages such as :mod:`physicskit.astro` compute with :math:`G = 1`,
    leaving the length and mass scales free. Fixing a length scale
    ``length_m`` (in meters, "1 length unit") and a mass scale
    ``mass_kg`` (in kg, "1 mass unit") determines the corresponding time
    and velocity units via :math:`G_{SI} = 1 \\times L^3/(M T^2)`, i.e.
    :math:`T = \\sqrt{L^3/(G_{SI} M)}`.

    Parameters
    ----------
    length_m : float
        The physical length, in meters, that corresponds to 1 length
        unit in the :math:`G = 1` system.
    mass_kg : float
        The physical mass, in kg, that corresponds to 1 mass unit in the
        :math:`G = 1` system.

    Returns
    -------
    dict
        A dict with keys ``"length_m"``, ``"mass_kg"``, ``"time_s"``, and
        ``"velocity_m_s"`` giving the physical size of one unit of each
        quantity in this unit system.

    Examples
    --------
    >>> units = gravitational_unit_system(length_m=PARSEC_M, mass_kg=SOLAR_MASS_KG)
    >>> units["time_s"] > 0
    True
    """
    time_s = (length_m**3 / (G * mass_kg)) ** 0.5
    return {
        "length_m": length_m,
        "mass_kg": mass_kg,
        "time_s": time_s,
        "velocity_m_s": length_m / time_s,
    }
