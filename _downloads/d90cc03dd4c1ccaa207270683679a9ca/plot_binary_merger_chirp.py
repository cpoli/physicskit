r"""
GW150914: the first direct detection of gravitational waves
==================================================================

On September 14, 2015, LIGO detected a gravitational wave chirp from two
black holes (roughly 36 and 29 solar masses) spiraling together and merging
1.3 billion light-years away -- the first direct detection of gravitational
waves, a phenomenon predicted by Einstein in 1916 that took a century to
observe. This example generates a toy inspiral-merger-ringdown (IMR)
waveform for a comparable system using the leading-order (Newtonian
quadrupole) inspiral formula, in which the gravitational-wave frequency
sweeps upward as

.. math::

    f(t) = \frac{1}{\pi}\left[\frac{5}{256\,(t_{\text{merger}}-t)}\right]^{3/8}
           \mathcal{M}^{-5/8}, \qquad
    \mathcal{M} = \frac{(m_1 m_2)^{3/5}}{(m_1+m_2)^{1/5}}

where :math:`\mathcal{M}` is the chirp mass, the single mass combination
that controls the whole inspiral's rate of frequency sweep. Both the
frequency and the strain amplitude grow as the binary loses energy to
radiation and spirals in, right up to merger; after that, the newly formed
black hole rings down like a struck bell, radiating away its distortion as
a damped sinusoid at its dominant quasinormal-mode frequency.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.relativity.chapters.gw_merger import BinaryMerger
from physicskit.relativity.utils import constants as const
from physicskit.relativity.visualizers.wave_plots import animate_wave_ripple, plot_strain_waveform, plot_wave_ripple

# %%
# The chirp: frequency and amplitude both sweep upward toward merger
# ------------------------------------------------------------------------
# The plus and cross strain polarizations, at distance :math:`R` and
# inclination :math:`\iota` between the orbital angular momentum and the
# line of sight, are
#
# .. math::
#
#     h_+(t) = \frac{4}{R}\mathcal{M}^{5/3}(\pi f(t))^{2/3}
#              \frac{1+\cos^2\iota}{2}\cos\Phi(t), \qquad
#     h_\times(t) = \frac{4}{R}\mathcal{M}^{5/3}(\pi f(t))^{2/3}
#              \cos\iota\,\sin\Phi(t)
#
# with :math:`\Phi(t)` the accumulated orbital phase. Both polarizations
# grow in frequency and amplitude together as the phase advances faster and
# faster toward merger.
#
# :class:`~physicskit.relativity.chapters.gw_merger.BinaryMerger` works
# entirely in geometrized units (:math:`G=c=1`, so mass, length, and time
# all share the same "meters" unit) -- exactly like the rest of
# ``physicskit.relativity`` -- so physical masses, distances, and times must
# first be converted with :func:`~physicskit.relativity.utils.constants.solar_masses_to_geometrized`
# and :func:`~physicskit.relativity.utils.constants.seconds_to_geometrized`.
# Skipping that conversion (passing raw solar-mass and second values
# straight through) silently produces a "chirp" that never actually
# oscillates: with the real chirp mass expressed directly in meters
# (tens of thousands of them), a handful of raw "seconds" is such a
# vanishingly small fraction of a wave period that essentially no phase
# accumulates at all.
m1 = const.solar_masses_to_geometrized(36.0)
m2 = const.solar_masses_to_geometrized(29.0)
distance = const.PARSEC_M * 410.0e6  # GW150914: roughly 410 Mpc away
merger = BinaryMerger(m1=m1, m2=m2, distance=distance, inclination=0.5)
t_merger = 0.0
t = const.seconds_to_geometrized(np.linspace(-0.6, 0.05, 4000))
hp, hc = merger.full_waveform(t, t_merger)

plot_strain_waveform(t, hp, hc, t_merger=t_merger)
plt.title(
    f"GW150914-like chirp: M1={const.geometrized_to_solar_masses(merger.m1):.0f} Msun, "
    f"M2={const.geometrized_to_solar_masses(merger.m2):.0f} Msun, "
    f"chirp mass={const.geometrized_to_solar_masses(merger.chirp_mass):.1f} Msun"
)
plt.tight_layout()

# %%
# Frequency evolution: the classic chirp sweep
# ------------------------------------------------------------
# Isolating :math:`f(t) = \frac{1}{\pi}\left[5/(256(t_{\text{merger}}-t))\right]^{3/8}
# \mathcal{M}^{-5/8}` shows the characteristic runaway rise: as the binary
# loses energy to radiation it spirals closer, which raises the orbital
# (and hence gravitational-wave) frequency, which radiates even more
# strongly still.
t_inspiral_s = np.linspace(-0.6, -0.005, 1000)
f_Hz = merger.inspiral_frequency(const.seconds_to_geometrized(t_inspiral_s), t_merger) * const.C_SI
plt.figure(figsize=(7, 4))
plt.plot(t_inspiral_s, f_Hz)
plt.xlabel("time before merger [s]")
plt.ylabel("GW frequency [Hz]")
plt.title("Chirp: frequency sweeps upward as the binary spirals in")
plt.tight_layout()

# %%
# The ringdown: the remnant settles down like a struck bell
# ------------------------------------------------------------
# After merger the strain follows a damped sinusoid at the dominant
# (l=2, m=2) quasinormal-mode frequency :math:`f_{\text{QNM}}` and damping
# time :math:`\tau_{\text{damp}}` of the remnant black hole (mass
# :math:`M_f`, spin :math:`a_f`), using the Berti-Cardoso-Will (2006)
# fitting formulas
#
# .. math::
#
#     f_{\text{QNM}} = \frac{1}{2\pi M_f}\left[1.5251 - 1.1568(1-a_*)^{0.1292}\right],
#     \qquad a_* = a_f/M_f
#
# .. math::
#
#     Q = 0.7000 + 1.4187(1-a_*)^{-0.4990}, \qquad
#     \tau_{\text{damp}} = \frac{Q}{\pi f_{\text{QNM}}}
#
# so that :math:`h_+(t) \propto e^{-(t-t_{\text{merger}})/\tau_{\text{damp}}}
# \cos(2\pi f_{\text{QNM}}(t-t_{\text{merger}}))`.
f_qnm, tau_damp = merger.qnm_frequency_damping()
M_f, a_f = merger.remnant_estimate()
print(f"Remnant: M_f={const.geometrized_to_solar_masses(M_f):.1f} Msun, a_f/M_f={a_f / M_f:.3f}")
print(f"Ringdown quasinormal mode: f={f_qnm * const.C_SI:.1f} Hz, damping time tau={const.geometrized_to_seconds(tau_damp) * 1e3:.2f} ms")

# %%
# A snapshot of the spatial wave ripple pattern (schematic quadrupole lobes)
# ------------------------------------------------------------------------------
# The wave strain radiated outward is evaluated at each point's retarded
# time :math:`t_{\text{ret}} = t - r`, with :math:`1/r` amplitude falloff
# and the schematic quadrupolar :math:`\cos(2\varphi)` azimuthal pattern
# characteristic of the dominant (l=2, m=2) mode seen face-on -- the
# four-lobed pattern visible below. ``extent`` is a spatial half-width, in
# the same geometrized meters as ``m1``/``m2``/``distance``; a few times
# the wave's own wavelength near merger (a few thousand km here) keeps
# several rings on screen at once.
extent = 1.5e7  # meters
plot_wave_ripple(merger, t=const.seconds_to_geometrized(-0.1), t_merger=t_merger, extent=extent)
plt.tight_layout()
plt.show()

# %%
# Animation: the ripple expanding as the chirp accelerates
# ------------------------------------------------------------
# The same ripple pattern, played back frame by frame as the binary
# spirals through its final orbits and merges: the wavelength visibly
# shrinks and the amplitude climbs together, exactly the chirp underneath
# it accelerating.
t_frames = const.seconds_to_geometrized(np.linspace(-0.6, 0.05, 100))
anim = animate_wave_ripple(merger, t_merger, t_frames, grid_size=150, extent=extent)
plt.show()

# %%
# To save the animation to a file instead of (or in addition to) displaying
# it interactively, use e.g.::
#
#     anim.save("gw150914_ripple.gif", writer="pillow", fps=15)
