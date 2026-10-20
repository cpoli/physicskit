"""The worked example of the paper: one potential, -1/r, across four subpackages."""

import numpy as np
from numba import njit

import physicskit.relativity.utils.constants as const
from physicskit.astro.orbital_mechanics import orbital_period, state_from_orbital_elements
from physicskit.integrators import yoshida4_integrate
from physicskit.relativity.chapters.gw_merger import BinaryMerger
from physicskit.relativity.chapters.schwarzschild import SchwarzschildBlackHole
from physicskit.semiclassical.core.wkb import langer_corrected_wkb


# 1. Newton (astro + shared integrators): an ellipse, a = 1, e = 0.5, with the
#    symplectic Yoshida integrator. It closes after Kepler's period.
@njit
def gravity(r, t, p):
    return -p[0] * r / np.sum(r**2) ** 1.5


mu, a, e = 1.0, 1.0, 0.5
r0, v0 = state_from_orbital_elements(a, e, 0.0, 0.0, 0.0, 0.0, mu)
T = orbital_period(a, mu)
n = 20000
_, r, v = yoshida4_integrate(gravity, r0, v0, 0.0, T / n, n, np.array([mu]))
energy = 0.5 * np.sum(v**2, axis=1) - mu / np.linalg.norm(r, axis=1)
assert np.abs(r[-1] - r0).max() < 1e-9 and np.ptp(energy) < 1e-9

# 2. Einstein (relativity): a Schwarzschild geodesic no longer closes. Its
#    periapsis advances by Darwin's exact angle, well above the leading
#    post-Newtonian 6 pi M / p this close to the hole (M = 1, p = 2 r_a r_p / (r_a + r_p)).
bh = SchwarzschildBlackHole(M=1.0)
orbit = bh.integrate_geodesic(bh.eccentric_orbit_initial_state(r0=20.0, eccentricity_boost=0.15), dtau=0.02, n_steps=200000)
r_a, r_p = orbit["r"].max(), orbit["r"].min()
p, ecc = 2 * r_a * r_p / (r_a + r_p), (r_a - r_p) / (r_a + r_p)
chi = np.linspace(0, 2 * np.pi, 4001)
darwin = np.trapezoid(np.sqrt(p / (p - 6 - 2 * ecc * np.cos(chi))), chi) - 2 * np.pi
measured = np.mean(bh.perihelion_precession(orbit))
assert abs(measured - darwin) < np.radians(0.1) and measured > 6 * np.pi / p

# 3. Gravitational waves (relativity): the Hulse-Taylor pulsar's orbit decays
#    at the rate Peters' quadrupole formula predicts.
m1, m2 = (const.solar_masses_to_geometrized(m) for m in (1.4398, 1.3886))
pulsar = BinaryMerger(m1=m1, m2=m2, distance=1.0)
P = 27906.98  # orbital period (s)
a_pulsar = (pulsar.total_mass * (P * const.C_SI) ** 2 / (4 * np.pi**2)) ** (1 / 3)
dPdt = pulsar.period_decay_rate(a_pulsar, 0.6171338)
assert abs(dPdt / -2.4056e-12 - 1) < 0.005  # observed by radio timing

# 4. Quantum (semiclassical): the same -1/r, quantized by radial WKB with the
#    Langer correction, gives hydrogen's levels -1/2n^2 (atomic units).
for l in range(3):
    levels = langer_corrected_wkb(lambda x: -1.0 / x, l, m=1.0, r_max=400.0, n_max=4)
    assert np.allclose(levels, -0.5 / (np.arange(4) + l + 1) ** 2, rtol=1e-6)
print(f"precession {np.degrees(measured):.2f} deg/orbit (Darwin {np.degrees(darwin):.2f}), dP/dt = {dPdt:.4e}")
