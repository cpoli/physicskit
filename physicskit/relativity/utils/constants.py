"""Physical constants and the geometrized unit system used throughout physicskit.relativity.

Every calculation in this package is done in **geometrized units**
(:math:`G = c = 1`), the standard convention in numerical relativity. In
these units, mass, length, and time all share the same dimension: a mass
:math:`M` (in kilograms) corresponds to a length
:math:`GM/c^2` and a time :math:`GM/c^3`. Concretely, a Schwarzschild black
hole of mass :math:`M` (in geometrized length units) has its horizon at
:math:`r = 2M`, its photon sphere at :math:`r = 3M`, and its innermost
stable circular orbit at :math:`r = 6M` -- clean numbers that would carry
awkward factors of :math:`G` and :math:`c` in SI units.

Use :func:`solar_masses_to_geometrized` to convert a physical mass (in solar
masses) into this geometrized length, and :func:`geometrized_to_seconds` /
:func:`geometrized_to_meters` to convert results back to physical units for
reporting.
"""

from __future__ import annotations

import warnings

import numpy as np

from physicskit import constants as _const

__all__ = [
    "C_SI",
    "G_SI",
    "PARSEC_M",
    "SOLAR_MASS_KG",
    "check_geometrized_distance",
    "check_geometrized_mass",
    "geometrized_to_meters",
    "geometrized_to_seconds",
    "geometrized_to_solar_masses",
    "seconds_to_geometrized",
    "solar_masses_to_geometrized",
]

#: Newton's gravitational constant, in m^3 kg^-1 s^-2.
G_SI = _const.G

#: Speed of light in vacuum, in m/s.
C_SI = _const.C

#: Solar mass, in kg.
SOLAR_MASS_KG = _const.SOLAR_MASS_KG

#: One parsec, in meters.
PARSEC_M = _const.PARSEC_M


def solar_masses_to_geometrized(mass_solar):
    """Convert a mass in solar masses to geometrized length units (meters).

    .. math::

        M_{\\text{geom}} = \\frac{G M_\\odot}{c^2} \\times \\left(\\frac{M}{M_\\odot}\\right)

    Parameters
    ----------
    mass_solar : float or array_like
        Mass in units of the Sun's mass.

    Returns
    -------
    float or ndarray
        Mass expressed as a geometrized length, in meters. For example,
        this is the horizon radius (up to the factor of 2) of a
        Schwarzschild black hole of that mass.

    Examples
    --------
    >>> round(float(solar_masses_to_geometrized(1.0)), 2)
    1476.67
    """
    return np.asarray(mass_solar) * G_SI * SOLAR_MASS_KG / C_SI**2


def geometrized_to_meters(length_geom):
    """Identity conversion: geometrized lengths are already in meters.

    Provided for symmetry and readability at call sites (e.g.
    ``geometrized_to_meters(r_isco)``); geometrized *length* units coincide
    with SI meters by construction (:math:`G = c = 1`).
    """
    return np.asarray(length_geom)


def geometrized_to_seconds(length_geom):
    """Convert a geometrized length (or time expressed as a length, :math:`ct`) to seconds.

    Parameters
    ----------
    length_geom : float or array_like
        A geometrized quantity with dimensions of length, e.g. :math:`c
        t` for a geometrized time.

    Returns
    -------
    float or ndarray
        Time in seconds.
    """
    return np.asarray(length_geom) / C_SI


def seconds_to_geometrized(time_s):
    """Convert a time in seconds to a geometrized length (:math:`ct`, in meters)."""
    return np.asarray(time_s) * C_SI


def geometrized_to_solar_masses(mass_geom):
    """Convert a geometrized mass (length, in meters) back to solar masses.

    Parameters
    ----------
    mass_geom : float or array_like
        Mass expressed as a geometrized length (meters).

    Returns
    -------
    float or ndarray
        Mass in units of the Sun's mass.

    Examples
    --------
    >>> round(float(geometrized_to_solar_masses(solar_masses_to_geometrized(30.0))), 6)
    30.0
    """
    return np.asarray(mass_geom) * C_SI**2 / (G_SI * SOLAR_MASS_KG)


#: Below this, a mass argument is almost certainly an unconverted physical
#: value rather than a geometrized length: it's lighter than any known or
#: theorized neutron star (~1 solar mass ~ 1477 m), let alone a black hole.
_MIN_PLAUSIBLE_COMPACT_OBJECT_MASS_M = 100.0

#: Below this, a distance argument is almost certainly an unconverted
#: physical value: it's many orders of magnitude short of even 1 AU
#: (~1.5e11 m), let alone an astrophysical source distance.
_MIN_PLAUSIBLE_ASTROPHYSICAL_DISTANCE_M = 1.0e9


def check_geometrized_mass(value, name="mass"):
    """Warn if `value` looks like an unconverted physical mass rather than a geometrized length.

    Intended for classes that require an *absolute* geometrized mass with
    no normalized (``M=1``) convention available -- :class:`~physicskit.relativity.chapters.gw_merger.BinaryMerger`
    is the motivating case: its radiated strain amplitude depends on the
    true physical scale, so silently passing a raw solar-mass count (e.g.
    ``36.0`` instead of ``solar_masses_to_geometrized(36.0)``) doesn't
    raise an error -- it just produces a "chirp" that never actually
    accumulates any phase, an easy mistake to miss since nothing crashes.

    This is a heuristic plausibility check, not a unit system: it only
    catches the common case of a compact-object mass that's wildly too
    small to be a geometrized length. It does not (and cannot, without
    real unit tracking) catch every possible mis-conversion.

    Parameters
    ----------
    value : float
        The mass argument to check, in whatever units the caller passed.
    name : str, default="mass"
        Parameter name to mention in the warning message.

    Examples
    --------
    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as w:
    ...     warnings.simplefilter("always")
    ...     check_geometrized_mass(36.0, name="m1")  # raw solar masses -- wrong
    ...     len(w)
    1
    >>> with warnings.catch_warnings(record=True) as w:
    ...     warnings.simplefilter("always")
    ...     check_geometrized_mass(solar_masses_to_geometrized(36.0), name="m1")  # correct
    ...     len(w)
    0
    """
    if 0.0 < value < _MIN_PLAUSIBLE_COMPACT_OBJECT_MASS_M:
        warnings.warn(
            f"{name}={value!r} is implausibly small for a geometrized-unit mass "
            f"(1 solar mass is ~{solar_masses_to_geometrized(1.0):.1f} m; even the "
            f"lightest known neutron star is a large fraction of that). Did you "
            f"forget to convert with solar_masses_to_geometrized()? If this value "
            f"is intentional (e.g. a sub-solar-mass primordial black hole), ignore "
            f"this warning.",
            UserWarning,
            stacklevel=3,
        )


def check_geometrized_distance(value, name="distance"):
    """Warn if `value` looks like an unconverted physical distance rather than a geometrized length.

    Same rationale as :func:`check_geometrized_mass`, for distance
    arguments: a raw parsec/Mpc count (e.g. ``410.0`` instead of
    ``410.0 * PARSEC_M * 1e6``) is many orders of magnitude too small to
    be a real geometrized distance, but doesn't raise an error on its own.

    Parameters
    ----------
    value : float
        The distance argument to check, in whatever units the caller passed.
    name : str, default="distance"
        Parameter name to mention in the warning message.

    Examples
    --------
    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as w:
    ...     warnings.simplefilter("always")
    ...     check_geometrized_distance(410.0, name="distance")  # raw Mpc -- wrong
    ...     len(w)
    1
    >>> with warnings.catch_warnings(record=True) as w:
    ...     warnings.simplefilter("always")
    ...     check_geometrized_distance(410.0e6 * PARSEC_M, name="distance")  # correct
    ...     len(w)
    0
    """
    if 0.0 < value < _MIN_PLAUSIBLE_ASTROPHYSICAL_DISTANCE_M:
        warnings.warn(
            f"{name}={value!r} is implausibly small for a geometrized-unit distance "
            f"(1 parsec is ~{PARSEC_M:.3e} m). Did you forget to convert, e.g. "
            f"distance_mpc * 1e6 * PARSEC_M? If this value is intentional (e.g. a "
            f"laboratory- or Earth-orbit-scale distance), ignore this warning.",
            UserWarning,
            stacklevel=3,
        )
