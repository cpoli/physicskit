r"""
The Eddington eclipse expedition: starlight bent by the Sun
==============================================================

General relativity predicts that a light ray passing a mass :math:`M` at
impact parameter :math:`b` is deflected by :math:`\delta = 4GM/(c^2b)`,
twice what Newtonian gravity gives for a particle moving at :math:`c`.
For a ray grazing the Sun that is 1.75 arcseconds. On 29 May 1919
expeditions to Sobral and Príncipe photographed the stars around the
eclipsed Sun and compared their positions with night-time plates of the
same field. The stars had shifted outward by the Einstein amount.

This example computes the deflection with
:meth:`~physicskit.relativity.chapters.schwarzschild.SchwarzschildBlackHole.light_deflection_angle`
in solar units, draws the displacement pattern of a field of stars
around the eclipsed Sun, and uses
:func:`~physicskit.relativity.chapters.lensing.exact_deflection_angle`
to show that the weak-field formula is essentially exact at the Sun but
fails near a black hole's photon sphere.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.relativity.chapters.lensing import exact_deflection_angle
from physicskit.relativity.chapters.schwarzschild import SchwarzschildBlackHole
from physicskit.relativity.utils import constants as const

ARCSEC = np.pi / 180 / 3600

# %%
# The grazing deflection
# --------------------------
# In geometrized units the Sun's mass is :math:`GM_\odot/c^2 = 1.477` km.
M_sun = const.solar_masses_to_geometrized(1.0)
R_sun = 6.957e8
sun = SchwarzschildBlackHole(M=M_sun)
delta_gr = sun.light_deflection_angle(R_sun) / ARCSEC
print(f"GM_sun / c^2 = {M_sun / 1e3:.3f} km")
print(f"deflection at the solar limb: Einstein {delta_gr:.3f} arcsec, Newtonian {delta_gr / 2:.3f} arcsec")
print("1919 results: Sobral 1.98 +/- 0.12 arcsec, Principe 1.61 +/- 0.30 arcsec")

# %%
# What the plates showed
# --------------------------
# Every star is pushed radially away from the Sun by
# :math:`\delta(b) = 1.75'' \times R_\odot/b`. The shifts, hugely
# exaggerated here, fall off as :math:`1/b`: only stars near the limb move
# measurably, which is why an eclipse was needed.
rng = np.random.default_rng(1919)
r_star = R_sun * rng.uniform(1.5, 6.0, 40)
phi = rng.uniform(0, 2 * np.pi, 40)
shift = np.array([sun.light_deflection_angle(b) for b in r_star]) / ARCSEC
arrow_per_arcsec = 0.6  # drawn arrow length, in solar radii, per arcsecond of shift
exaggeration = arrow_per_arcsec * 959.6  # the Sun's radius subtends 959.6 arcsec
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.add_patch(plt.Circle((0, 0), 1.0, color="k"))
ax1.add_patch(plt.Circle((0, 0), 1.6, color="gold", alpha=0.25, lw=0))
x0, y0 = r_star / R_sun * np.cos(phi), r_star / R_sun * np.sin(phi)
ax1.plot(x0, y0, "o", color="0.6", ms=4, label="night-time position")
ax1.quiver(
    x0,
    y0,
    np.cos(phi) * shift,
    np.sin(phi) * shift,
    angles="xy",
    scale_units="xy",
    scale=1 / arrow_per_arcsec,
    color="firebrick",
    width=0.004,
    label=f"eclipse shift (x{exaggeration:.0f})",
)
ax1.set_xlim(-6.5, 6.5)
ax1.set_ylim(-6.5, 6.5)
ax1.set_aspect("equal")
ax1.set_xlabel(r"x [$R_\odot$]")
ax1.set_ylabel(r"y [$R_\odot$]")
ax1.set_title("Star field around the eclipsed Sun")
ax1.legend(fontsize=8, loc="upper right")

b_over_R = np.linspace(1, 6, 100)
ax2.plot(b_over_R, 1.75 / b_over_R, color="firebrick", label=r"Einstein $4GM/c^2b$")
ax2.plot(b_over_R, 0.875 / b_over_R, "--", color="steelblue", label="Newtonian (half)")
ax2.errorbar([1.0, 1.0], [1.98, 1.61], yerr=[0.12, 0.30], fmt="s", color="k", capsize=3, label="1919, extrapolated to the limb")
ax2.set_xlabel(r"impact parameter $b / R_\odot$")
ax2.set_ylabel("deflection [arcsec]")
ax2.set_title("Deflection falls off as 1/b")
ax2.legend(fontsize=8)
fig.tight_layout()

# %%
# Where the weak-field formula holds
# --------------------------------------
# Integrating the exact null geodesic around a unit-mass black hole shows
# :math:`4M/b` is accurate for :math:`b \gg M`. The Sun's limb is at
# :math:`b \approx 470{,}000\,M`; near the photon sphere,
# :math:`b_c=3\sqrt3M\approx5.2M`, the exact deflection grows without
# bound.
bh = SchwarzschildBlackHole(M=1.0)
b_vals = np.array([6.0, 8.0, 12.0, 20.0, 50.0])
exact = np.array([exact_deflection_angle(bh, b) for b in b_vals])
for b, ex in zip(b_vals, exact):
    print(f"b = {b:6.1f} M: exact {ex:.5f} rad, weak-field {4 / b:.5f} rad, ratio {ex / (4 / b):.3f}")
print(f"the Sun's limb: b = {R_sun / M_sun:.3e} M, so the next-order correction is of relative size ~ M/b = {M_sun / R_sun:.1e}")

plt.show()
