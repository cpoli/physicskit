r"""
GPS: General Relativity in your pocket
============================================

Every GPS satellite carries an atomic clock that runs measurably faster
than an identical clock on the ground -- about 38 microseconds per day, a
combination of gravitational blueshift (from sitting higher in Earth's
potential well) partially offset by special-relativistic time dilation
(from orbital speed). A clock at radius :math:`r` moving with angular
velocity :math:`\Omega` around a mass :math:`M` (Earth, here) has a
proper-to-coordinate time rate that combines both effects in one formula:

.. math::

    \frac{d\tau}{dt} = \sqrt{\left(1 - \frac{2M}{r}\right) - r^2 \Omega^2}

For a GPS satellite (:math:`\Omega` the Keplerian orbital rate
:math:`\sqrt{M/r^3}`) versus a ground station (:math:`\Omega` Earth's
sidereal rotation rate), this net rate difference amounts to about 38
microseconds per day. Left uncorrected, this drift would make GPS position
fixes wander by roughly 10 km per day. Every GPS satellite clock is built to
tick deliberately slow on the ground, specifically so that once in orbit it
matches ground clocks -- General Relativity, engineered into a system used
by billions of people daily.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.relativity.chapters.timekeeping import (
    EARTH_RADIUS_M,
    GPS_ORBITAL_RADIUS_M,
    clock_rate_factor,
    earth_mass_geometrized,
    gps_relativistic_offset_per_day,
)

# %%
# The net correction, and its two competing pieces
# ------------------------------------------------------------
M = earth_mass_geometrized()
net_offset = gps_relativistic_offset_per_day()

gravity_only_sat = clock_rate_factor(GPS_ORBITAL_RADIUS_M, 0.0, M=M)
gravity_only_ground = clock_rate_factor(EARTH_RADIUS_M, 0.0, M=M)
gravitational_offset = (gravity_only_sat - gravity_only_ground) * 86400.0 * 1.0e6
velocity_offset = net_offset - gravitational_offset

print(f"Gravitational blueshift (satellite higher up):  +{gravitational_offset:.2f} microseconds/day")
print(f"Velocity time dilation (satellite's orbital speed): {velocity_offset:.2f} microseconds/day")
print(f"Net relativistic correction: {net_offset:.2f} microseconds/day")
print(f"Uncorrected position drift: roughly {net_offset * 1e-6 * 3e8 / 1000.0:.1f} km/day")

# %%
# How the correction depends on orbital altitude
# ------------------------------------------------------------
altitudes_km = np.linspace(500.0, 40000.0, 200)
radii = EARTH_RADIUS_M + altitudes_km * 1000.0
offsets = [gps_relativistic_offset_per_day(r_satellite=r) for r in radii]

plt.figure(figsize=(7, 4.5))
plt.plot(altitudes_km, offsets)
plt.axvline(
    (GPS_ORBITAL_RADIUS_M - EARTH_RADIUS_M) / 1000.0,
    color="k",
    linestyle="--",
    linewidth=1,
    label="GPS altitude",
)
plt.axhline(0.0, color="gray", linewidth=0.5)
plt.xlabel("orbital altitude [km]")
plt.ylabel("net relativistic offset [microseconds/day]")
plt.title("GPS relativistic correction vs. orbital altitude")
plt.legend()
plt.tight_layout()

# %%
# The correction also depends (weakly) on the ground station's own altitude
# --------------------------------------------------------------------------------
# :func:`~physicskit.relativity.chapters.timekeeping.gps_relativistic_offset_per_day`
# takes both ``r_satellite`` and ``r_ground`` independently -- the curve above
# only varies the satellite's altitude at sea level. A ground station on a
# high plateau or mountaintop sits measurably higher in Earth's potential
# well too, running its own clock very slightly faster and so trimming the
# net satellite-minus-ground offset by a small but real amount.
sat_altitude_km = np.linspace(500.0, 40000.0, 120)
ground_altitude_km = np.linspace(0.0, 5.0, 80)  # sea level up to a high mountain
SAT_ALT, GROUND_ALT = np.meshgrid(sat_altitude_km, ground_altitude_km)
r_sat = EARTH_RADIUS_M + SAT_ALT * 1000.0
r_grd = EARTH_RADIUS_M + GROUND_ALT * 1000.0
offset_map = gps_relativistic_offset_per_day(r_satellite=r_sat, r_ground=r_grd)

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.pcolormesh(sat_altitude_km, ground_altitude_km, offset_map, shading="auto", cmap="coolwarm")
plt.colorbar(im, ax=ax, label="net relativistic offset [microseconds/day]")
ax.axvline(
    (GPS_ORBITAL_RADIUS_M - EARTH_RADIUS_M) / 1000.0,
    color="k",
    linestyle="--",
    linewidth=1,
    label="GPS altitude",
)
ax.set_xlabel("satellite orbital altitude [km]")
ax.set_ylabel("ground-station altitude [km]")
ax.set_title("GPS correction vs. satellite and ground-station altitude")
ax.legend()
plt.tight_layout()
plt.show()
