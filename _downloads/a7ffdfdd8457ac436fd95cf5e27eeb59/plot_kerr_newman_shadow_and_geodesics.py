r"""
The Kerr-Newman black hole: geodesics and the shadow of a charged, rotating hole
================================================================================

In 1965 Ezra Newman and co-workers found the charged version of Kerr's
rotating black hole. It is the Kerr metric with

.. math::

    \Delta = r^2 - 2Mr + a^2 + Q^2,

the most general stationary black hole of Einstein-Maxwell theory, and it
contains the whole family: Schwarzschild (:math:`a = Q = 0`),
Reissner-Nordström (:math:`a = 0`) and Kerr (:math:`Q = 0`). Horizons exist
for :math:`a^2 + Q^2 < M^2`. Carter (1968) showed that its geodesics, even
those of charged particles, separate, with a fourth constant of motion.
Photons on *spherical* orbits, where :math:`R(r) = R'(r) = 0`, wind around
the hole at fixed radius, and their critical impact parameters draw the
edge of the shadow a distant observer sees. This example traces those
orbits, compares the shadows across the family, and checks the analytic
edge against a backward ray-traced image.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

from physicskit.relativity.chapters.kerr_newman import KerrNewmanBlackHole

fig = plt.figure(figsize=(17, 4.4))

# %%
# A spherical photon orbit
# ------------------------
bh = KerrNewmanBlackHole(a=0.7, Q=0.4)
r_pro, r_retro = bh.equatorial_photon_orbits()
r_sph = 0.5 * (r_pro + r_retro)
xi, eta = bh.photon_orbit_constants(r_sph)
orbit = bh.geodesic(r_sph, np.pi / 2, 1.0, xi, eta, n_steps=20000, step=0.005, record_every=4)
ax = fig.add_subplot(1, 4, 1, projection="3d")
ax.plot(orbit["x"], orbit["y"], orbit["z"], lw=0.4, color="darkorange")
lim = 1.1 * r_sph
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 15j]
rh = bh.outer_horizon_radius
ax.plot_surface(rh * np.cos(u) * np.sin(v), rh * np.sin(u) * np.sin(v), rh * np.cos(v), color="black", alpha=0.6)
ax.set_box_aspect((1, 1, 1))
ax.set_title(f"Spherical photon orbit, r = {r_sph:.3f} M\n(a = 0.7, Q = 0.4)", fontsize=9)
ax.set_axis_off()
print(f"photon orbit radius stays within {np.ptp(orbit['r']):.1e} of {r_sph:.4f}")

# %%
# Charged and neutral particles with the same constants of motion
# ---------------------------------------------------------------
# A particle's charge enters only through P = E(r^2 + a^2) - aL - eQr. With
# the same energy and angular momentum, a particle of charge opposite to the
# hole's is pulled in: it reaches closer and swings further out, and its
# orbit precesses faster.
bh_c = KerrNewmanBlackHole(a=0.5, Q=0.6)
ax = fig.add_subplot(1, 4, 2)
periapses = {}
for e, color, label in [(0.0, "navy", "neutral"), (-0.4, "crimson", "charge e = -0.4")]:
    g = bh_c.geodesic(12.0, np.pi / 2, 0.97, 3.8, 0.0, mu2=1.0, e=e, n_steps=40000, step=0.002, record_every=10)
    periapses[e] = g["r"].min()
    ax.plot(g["x"], g["y"], lw=0.5, color=color, label=f"{label}: periapsis {g['r'].min():.2f} M")
ax.add_patch(plt.Circle((0, 0), bh_c.outer_horizon_radius, color="black"))
ax.set_aspect("equal")
ax.set_xlim(-36, 36)
ax.set_ylim(-36, 36)
ax.set_title("Equatorial bound orbits, a = 0.5, Q = 0.6, E = 0.97, L = 3.8", fontsize=9)
ax.legend(fontsize=7, loc="lower left")

# %%
# The family of shadows
# ---------------------
ax = fig.add_subplot(1, 4, 3)
for a, Q, color, label in [
    (0.0, 0.0, "black", "Schwarzschild"),
    (0.0, 0.8, "green", "Reissner-Nordström, Q = 0.8"),
    (0.9, 0.0, "navy", "Kerr, a = 0.9"),
    (0.9, 0.43, "crimson", "Kerr-Newman, a = 0.9, Q = 0.43"),
]:
    al, be = KerrNewmanBlackHole(a=a, Q=Q).shadow_boundary()
    ax.plot(al, be, color=color, label=label)
ax.set_aspect("equal")
ax.set_xlabel(r"$\alpha / M$")
ax.set_ylabel(r"$\beta / M$")
ax.set_title("Shadow edges, edge-on observer", fontsize=9)
ax.legend(fontsize=7, loc="lower left")

# %%
# Ray-traced shadow against the analytic edge
# -------------------------------------------
# Every pixel's photon is integrated backward from r = 500 M; black pixels
# are captured by the horizon.
bh_rt = KerrNewmanBlackHole(a=0.9, Q=0.3)
pix = np.linspace(-7.0, 7.0, 81)
img = bh_rt.ray_traced_shadow(pix, pix)
al, be = bh_rt.shadow_boundary()
ax = fig.add_subplot(1, 4, 4)
ax.imshow(img == 1, origin="lower", extent=(pix[0], pix[-1], pix[0], pix[-1]), cmap="Greys")
ax.plot(al, be, color="crimson", lw=1, label="analytic edge")
ax.set_xlabel(r"$\alpha / M$")
ax.set_title("Ray-traced, a = 0.9, Q = 0.3", fontsize=9)
ax.legend(fontsize=7, loc="lower left")
plt.tight_layout()
plt.show()

X, Y = np.meshgrid(pix, pix)
inside = Path(np.column_stack((al, be))).contains_points(np.column_stack((X.ravel(), Y.ravel()))).reshape(X.shape)
mismatch = np.mean((img == 1) != inside)
print(f"pixels where ray tracing and the analytic edge disagree: {100 * mismatch:.2f}%")
r_rn = KerrNewmanBlackHole(Q=0.8).shadow_radius_nonrotating()
print(f"Reissner-Nordstrom Q = 0.8 shadow radius {r_rn:.4f} M (Schwarzschild {np.sqrt(27):.4f} M)")

# %%
# Check
# -----
assert np.ptp(orbit["r"]) < 1e-8
# turning points are the roots of R(r): 7.89 M (neutral) and 4.85 M (e = -0.4)
assert abs(periapses[0.0] - 7.89) < 0.02 and abs(periapses[-0.4] - 4.85) < 0.02
assert mismatch < 0.02
assert KerrNewmanBlackHole(a=0.9, Q=0.43).shadow_area() < KerrNewmanBlackHole(a=0.9).shadow_area()
