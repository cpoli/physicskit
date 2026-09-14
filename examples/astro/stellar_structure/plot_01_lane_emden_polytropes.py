r"""
Lane-Emden polytropes: Lane, Ritter, and Emden's gas spheres
=================================================================

Lane (1870) first treated a star as a self-gravitating polytropic gas
sphere, Ritter (1878-1889) generalized the analysis to arbitrary
polytropic index :math:`n`, and Emden (1907) systematized the resulting
family of solutions to the dimensionless equation

.. math::

    \frac{1}{\xi^2}\frac{d}{d\xi}\left(\xi^2\frac{d\theta}{d\xi}\right)
    + \theta^n = 0, \qquad \theta(0)=1,\ \theta'(0)=0.

:func:`~physicskit.astro.stellar_structure.lane_emden` integrates
exactly this equation; :class:`~physicskit.astro.stellar_structure.PolytropicStar`
builds a physical star (radius, mass) from one solution. This example
compares the profile shape across several classic polytropic indices,
checks two of them against their known closed forms, and builds one
physical star.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.astro.stellar_structure import PolytropicStar, lane_emden
from physicskit.astro.visualizers import plot_lane_emden

# %%
# The Lane-Emden family, across classic polytropic indices
# ------------------------------------------------------------------
# n=0 (uniform density), n=1 (has a closed-form sinc solution), n=1.5
# (non-relativistic degenerate matter / fully convective stars), n=3
# (Eddington's "standard model" / relativistic degenerate matter,
# below), and n=4 (approaching the n=5 index at which the star has
# infinite radius). Larger n means more centrally concentrated mass.
indices = [0.0, 1.0, 1.5, 3.0, 4.0]
fig1, ax1 = plt.subplots(figsize=(6.5, 5))
for n in indices:
    xi, theta = lane_emden(n)
    ax1.plot(xi, theta, label=f"n={n} (surface at $\\xi_1$={xi[-1]:.4f})")
ax1.axhline(0.0, color="0.7", lw=0.8)
ax1.set_xlabel(r"$\xi$")
ax1.set_ylabel(r"$\theta(\xi)$")
ax1.set_title("The Lane-Emden family: larger n is more centrally concentrated")
ax1.legend(fontsize=8)
fig1.tight_layout()

# %%
# Checking two indices against their exact closed forms
# -----------------------------------------------------------------
# n=0 has the exact solution :math:`\theta=1-\xi^2/6`, with its first
# zero at :math:`\xi_1=\sqrt6`; n=1 has the exact solution
# :math:`\theta=\sin(\xi)/\xi`, with its first zero at
# :math:`\xi_1=\pi`.
xi0, theta0 = lane_emden(0.0)
xi1, theta1 = lane_emden(1.0)
print(f"n=0 surface: numeric xi_1={xi0[-1]:.6f}, exact sqrt(6)={np.sqrt(6):.6f}")
print(f"n=1 surface: numeric xi_1={xi1[-1]:.6f}, exact pi={np.pi:.6f}")
print(f"n=0 profile max error vs exact 1-xi^2/6: {np.max(np.abs(theta0 - (1 - xi0**2 / 6))):.2e}")
with np.errstate(divide="ignore", invalid="ignore"):
    exact_n1 = np.where(xi1 > 1e-8, np.sin(xi1) / xi1, 1.0)
print(f"n=1 profile max error vs exact sin(xi)/xi:  {np.max(np.abs(theta1 - exact_n1)):.2e}")

# %%
# One physical star
# ----------------------
# :class:`PolytropicStar` turns a single Lane-Emden solution into a
# physical star, given a polytropic constant ``K`` and central density
# ``rho_c`` (here in arbitrary :math:`G=1` units).
star = PolytropicStar(n=1.5, K=1.0, rho_c=1.0)
print(f"\nn=1.5 star: xi_1={star.xi1:.6f}, alpha={star.alpha:.6f}, radius={star.radius:.6f}, mass={star.mass:.6f}")

xi_star, theta_star = lane_emden(1.5)
fig2, ax2 = plot_lane_emden(xi_star, theta_star)
ax2.set_title(f"n=1.5 star (radius={star.radius:.3f}, mass={star.mass:.3f} in code units)")
fig2.tight_layout()

plt.show()
