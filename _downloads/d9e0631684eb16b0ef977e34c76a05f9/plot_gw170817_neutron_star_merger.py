r"""
GW170817: the first binary neutron star merger
====================================================

On 17 August 2017, LIGO and Virgo detected the inspiral of two neutron
stars -- far lighter than any binary black hole seen before, and
correspondingly slower-chirping: the signal remained in the detectors'
sensitive band for over a minute, instead of the fraction of a second
GW150914's ~30-solar-mass black holes took. The measured chirp mass,

.. math::

    \mathcal{M} = \frac{(m_1 m_2)^{3/5}}{(m_1+m_2)^{1/5}} \approx 1.186\,M_\odot,

was immediately recognizable as two neutron stars rather than two black
holes: far below any black hole binary LIGO had detected, and consistent
with the narrow mass range neutron stars are observed to occupy. This
example reproduces that chirp mass from representative component masses
and, using the same leading-order (Newtonian quadrupole) inspiral formula
as the GW150914 example, shows directly why such light compact objects
sweep through the detector band so much more slowly than a comparable
binary black hole.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.relativity.chapters.gw_merger import BinaryMerger
from physicskit.relativity.utils import constants as const

# %%
# The chirp mass pins down the source as two neutron stars, not black holes
# ---------------------------------------------------------------------------
# Component masses of roughly 1.46 and 1.27 solar masses -- squarely inside
# the observed neutron star mass range -- reproduce the measured chirp mass
# to within its reported uncertainty.
m1_Msun, m2_Msun = 1.46, 1.27
m1 = const.solar_masses_to_geometrized(m1_Msun)
m2 = const.solar_masses_to_geometrized(m2_Msun)
distance = const.PARSEC_M * 40.0e6  # GW170817: roughly 40 Mpc away
gw170817 = BinaryMerger(m1=m1, m2=m2, distance=distance, inclination=0.4)
chirp_mass_Msun = const.geometrized_to_solar_masses(gw170817.chirp_mass)
print(f"GW170817-like: M1={m1_Msun} Msun, M2={m2_Msun} Msun, chirp mass={chirp_mass_Msun:.3f} Msun (measured: ~1.186 Msun)")


# %%
# Why light neutron stars chirp for so much longer than heavy black holes
# ---------------------------------------------------------------------------
# Inverting the inspiral frequency formula
#
# .. math::
#
#     f(t) = \frac{1}{\pi}\left[\frac{5}{256\,(t_{\text{merger}}-t)}\right]^{3/8}
#            \mathcal{M}^{-5/8}
#
# for the time remaining before merger at a given frequency,
#
# .. math::
#
#     \tau(f) = \frac{5}{256\,(\pi f)^{8/3}\,\mathcal{M}^{5/3}},
#
# shows how strongly the chirp mass alone controls the inspiral's duration:
# a lighter chirp mass spends far longer sweeping through the same
# frequency band, since :math:`\tau \propto \mathcal{M}^{-5/3}`.
def time_to_merger_at_frequency(merger, f_gw_hz):
    """Time before merger at which the GW frequency first reaches ``f_gw_hz``."""
    f_geom = f_gw_hz / const.C_SI
    tau = 5.0 / (256.0 * (np.pi * f_geom) ** (8.0 / 3.0) * merger.chirp_mass ** (5.0 / 3.0))
    return const.geometrized_to_seconds(tau)


gw150914 = BinaryMerger(
    m1=const.solar_masses_to_geometrized(36.0),
    m2=const.solar_masses_to_geometrized(29.0),
    distance=const.PARSEC_M * 410.0e6,
)

f_low_hz = 24.0  # roughly where Advanced LIGO's sensitive band began for this event
tau_170817 = time_to_merger_at_frequency(gw170817, f_low_hz)
tau_150914 = time_to_merger_at_frequency(gw150914, f_low_hz)
print(f"Time from {f_low_hz:.0f} Hz to merger -- GW170817-like: {tau_170817:.1f} s, GW150914-like: {tau_150914:.2f} s")

t_170817_s = np.linspace(-tau_170817, -0.05, 2000)
t_150914_s = np.linspace(-tau_150914, -0.005, 2000)
f_170817_Hz = gw170817.inspiral_frequency(const.seconds_to_geometrized(t_170817_s), t_merger=0.0) * const.C_SI
f_150914_Hz = gw150914.inspiral_frequency(const.seconds_to_geometrized(t_150914_s), t_merger=0.0) * const.C_SI

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(t_170817_s, f_170817_Hz, label=f"GW170817-like (BNS): {tau_170817:.0f} s in band")
ax.plot(t_150914_s, f_150914_Hz, label=f"GW150914-like (BBH): {tau_150914:.2f} s in band")
ax.set_xlabel("time before merger [s]")
ax.set_ylabel("GW frequency [Hz]")
ax.set_yscale("log")
ax.set_title("Light neutron stars chirp far more slowly than heavy black holes")
ax.legend()
plt.tight_layout()
plt.show()
