r"""
Laplace and Legendre: the multipole expansion of a far field
===============================================================

Legendre (1782) expanded :math:`1/|\mathbf r - \mathbf r'|` in powers of
:math:`r'/r` while computing the gravitational pull of a spheroid, and so
introduced the polynomials that bear his name. Laplace (1785) extended
the idea to the full angular dependence, the spherical harmonics. The
result, applied to charges, is the multipole expansion:

.. math::

    \phi(\mathbf x) = \frac{1}{4\pi\varepsilon_0}\left[\frac{q}{r}
    + \frac{\mathbf p\cdot\mathbf x}{r^3}
    + \frac{1}{2}\sum_{jk} Q_{jk}\frac{x_j x_k}{r^5} + \cdots\right].

Far from any compact charge cluster, the potential is set by a handful of
numbers: the total charge, the dipole moment, and the quadrupole tensor.
Each extra term makes the error fall off one power of :math:`r` faster.
:func:`~physicskit.fields.multipole_moments` computes the moments and
:func:`~physicskit.fields.multipole_potential` sums the truncated series.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.fields import coulomb_potential, multipole_moments, multipole_potential

# %%
# A random cluster of charges
# ---------------------------
# Twelve charges scattered within a unit ball, carrying a net 2.4 nC.

rng = np.random.default_rng(1782)
charges = rng.normal(size=12) * 1e-9
charges -= charges.mean() - 0.2e-9
u = rng.normal(size=(12, 3))
positions = u / np.linalg.norm(u, axis=1, keepdims=True) * rng.uniform(0.2, 1.0, (12, 1))
m = multipole_moments(charges, positions)
print(f"monopole q = {m.monopole:.3e} C")
print(f"dipole p = {np.round(m.dipole, 12)} C m")
print(f"quadrupole trace = {np.trace(m.quadrupole):.1e} (traceless by construction)")

# %%
# Error of each truncation versus distance
# ----------------------------------------
# Along a fixed direction, the relative error of the order-:math:`\ell`
# truncation falls as :math:`r^{-(\ell+1)}`: each term captures one more
# power of :math:`1/r`.

direction = np.array([0.3, -0.5, 0.81])
direction /= np.linalg.norm(direction)
r = np.logspace(0.5, 2.5, 40)
pts = r[:, None] * direction
exact = coulomb_potential(charges, positions, pts)
errors = {order: np.abs(multipole_potential(m, pts, order=order) / exact - 1) for order in (0, 1, 2)}
for order, err in errors.items():
    slope = np.polyfit(np.log(r[20:]), np.log(err[20:]), 1)[0]
    print(f"order {order}: relative error ~ r^{slope:.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
names = {0: "monopole", 1: "+ dipole", 2: "+ quadrupole"}
for order, err in errors.items():
    ax1.loglog(r, err, label=names[order])
ax1.set_xlabel("distance r")
ax1.set_ylabel("relative error of the truncated series")
ax1.set_title("Each multipole order gains a power of 1/r")
ax1.legend()

# %%
# The potential on a sphere around the cluster
# --------------------------------------------
# On a sphere of radius 3 the exact potential and the quadrupole-order
# expansion are already close to indistinguishable in angle.

theta = np.linspace(0, np.pi, 90)
phi_ang = np.linspace(0, 2 * np.pi, 180)
TH, PH = np.meshgrid(theta, phi_ang, indexing="ij")
sphere = 3.0 * np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], axis=-1)
exact_s = coulomb_potential(charges, positions, sphere)
approx_s = multipole_potential(m, sphere, order=2)
im = ax2.pcolormesh(np.degrees(PH), np.degrees(TH), exact_s, shading="auto", cmap="RdBu_r")
ax2.contour(np.degrees(PH), np.degrees(TH), approx_s, levels=10, colors="k", linewidths=0.7)
ax2.set_xlabel("azimuth (deg)")
ax2.set_ylabel("polar angle (deg)")
ax2.set_title("exact potential at r = 3 (color), quadrupole order (contours)")
fig.colorbar(im, ax=ax2, label="potential (V)")
fig.tight_layout()
print(f"max deviation on the r = 3 sphere, relative to max |phi|: {np.max(np.abs(approx_s - exact_s)) / np.max(np.abs(exact_s)):.3f}")
plt.show()
