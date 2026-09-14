"""Relativistic clock corrections: why GPS needs General Relativity every day.

A GPS satellite clock experiences two competing relativistic effects
relative to a clock on the ground: it runs *faster* because it sits higher
in Earth's gravitational potential (gravitational blueshift), and *slower*
because it moves at orbital speed (special-relativistic time dilation). For
the GPS constellation's medium-Earth orbit, the gravitational effect wins by
a wide margin, and the net result -- about 38 microseconds per day -- is
large enough that, left uncorrected, GPS position errors would accumulate at
roughly 10 km per day. Every GPS satellite clock is deliberately built to
tick slightly slow on the ground so that, once in orbit, it keeps pace with
ground clocks -- General Relativity engineered directly into consumer
electronics.
"""

from __future__ import annotations

import numpy as np

from physicskit.relativity.utils.constants import C_SI, G_SI

__all__ = [
    "EARTH_MASS_KG",
    "EARTH_RADIUS_M",
    "EARTH_SIDEREAL_DAY_S",
    "GPS_ORBITAL_RADIUS_M",
    "clock_rate_factor",
    "earth_mass_geometrized",
    "gps_relativistic_offset_per_day",
]

#: Earth's mass, in kg.
EARTH_MASS_KG = 5.9722e24
#: Earth's mean equatorial radius, in meters.
EARTH_RADIUS_M = 6.371e6
#: Earth's sidereal rotation period, in seconds.
EARTH_SIDEREAL_DAY_S = 86164.0905
#: Typical GPS satellite orbital semi-major axis, in meters (~20,200 km altitude).
GPS_ORBITAL_RADIUS_M = 2.6560e7


def earth_mass_geometrized():
    """Earth's mass in geometrized length units, :math:`GM_\\oplus/c^2`.

    Returns
    -------
    float
    """
    return G_SI * EARTH_MASS_KG / C_SI**2


def clock_rate_factor(r, angular_velocity_si, M=None):
    """Proper-to-coordinate time rate :math:`d\\tau/dt` for a clock at radius ``r`` rotating at a given rate.

    .. math::

        \\frac{d\\tau}{dt} = \\sqrt{\\left(1 - \\frac{2M}{r}\\right) - r^2 \\Omega^2}

    combining gravitational time dilation and the special-relativistic time
    dilation of the clock's rotational motion in one formula (this reduces
    to the familiar circular-orbit result :math:`\\sqrt{1-3M/r}` when
    :math:`\\Omega` is the Keplerian orbital angular velocity, and to pure
    gravitational redshift :math:`\\sqrt{1-2M/r}` when :math:`\\Omega = 0`).

    Parameters
    ----------
    r : float
        Radial coordinate, in meters.
    angular_velocity_si : float
        The clock's angular velocity, in rad/s (SI).
    M : float, optional
        Central mass, in geometrized length units. Defaults to
        :func:`earth_mass_geometrized`.

    Returns
    -------
    float
        The dimensionless rate :math:`d\\tau/dt`.
    """
    if M is None:
        M = earth_mass_geometrized()
    omega_geometrized = angular_velocity_si / C_SI  # rad/s -> rad per geometrized length unit
    f = 1.0 - 2.0 * M / r
    return np.sqrt(f - r**2 * omega_geometrized**2)


def gps_relativistic_offset_per_day(r_satellite=None, r_ground=None, M=None):
    """Net relativistic clock rate offset of a GPS satellite relative to the ground, in microseconds/day.

    The satellite is treated as free-falling in a circular Keplerian orbit
    (angular velocity :math:`\\Omega = \\sqrt{M/r^3}`); the ground clock is
    treated as co-rotating with Earth at its sidereal rate. A positive
    result means the satellite clock runs fast relative to the ground.

    Parameters
    ----------
    r_satellite : float, optional
        Satellite orbital radius, in meters. Defaults to
        :data:`GPS_ORBITAL_RADIUS_M`.
    r_ground : float, optional
        Ground station radius, in meters. Defaults to
        :data:`EARTH_RADIUS_M`.
    M : float, optional
        Earth's mass, in geometrized length units. Defaults to
        :func:`earth_mass_geometrized`.

    Returns
    -------
    float
        Accumulated time offset, in microseconds per day (positive: the
        satellite clock runs fast).

    Examples
    --------
    >>> offset = gps_relativistic_offset_per_day()
    >>> bool(30.0 < offset < 45.0)  # the well-known ~38 microseconds/day GPS correction
    True
    """
    if r_satellite is None:
        r_satellite = GPS_ORBITAL_RADIUS_M
    if r_ground is None:
        r_ground = EARTH_RADIUS_M
    if M is None:
        M = earth_mass_geometrized()

    omega_satellite_si = np.sqrt(M / r_satellite**3) * C_SI  # geometrized Omega -> rad/s
    omega_ground_si = 2.0 * np.pi / EARTH_SIDEREAL_DAY_S

    rate_satellite = clock_rate_factor(r_satellite, omega_satellite_si, M=M)
    rate_ground = clock_rate_factor(r_ground, omega_ground_si, M=M)

    fractional_rate_diff = rate_satellite - rate_ground
    return fractional_rate_diff * 86400.0 * 1.0e6
