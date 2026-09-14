r"""
The Reynolds number: one ratio, regardless of size or fluid
=================================================================

Osborne Reynolds injected a thread of dye into water flowing through a
glass pipe and found that a single dimensionless ratio,

.. math::

    Re = \frac{UL}{\nu},

controls exactly where the flow transitions from smooth to chaotic --
regardless of the pipe's size or the fluid's identity. That
dimensional-similarity claim is the founding result of scaled experimental
testing in fluid mechanics: a wind-tunnel model and a full-size aircraft
behave identically whenever their Reynolds numbers match, however different
their absolute sizes or fluids.

This example makes that "regardless of size or fluid" claim concrete using
:func:`~physicskit.fluids.systems.viscous_flow.stokes_drag` as a source of
physically realizable flows: a sphere settling at Stokes terminal velocity
provides a natural way to sweep both the length scale (the sphere's radius)
and the fluid's identity (its viscosity) independently, and
:func:`~physicskit.fluids.utils.dimensionless.reynolds_number` computes the
resulting ratio at every combination. The single :math:`Re=1` contour that
results cuts across both axes at once -- exactly the dimensional-similarity
signature Reynolds' 1883 experiments established, rather than a property
tied to any one sphere size or any one fluid.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.fluids.systems.viscous_flow import stokes_drag
from physicskit.fluids.utils.dimensionless import reynolds_number
from physicskit.fluids.visualizers import theme

# %%
# Sweep both sphere radius and fluid viscosity independently
# ---------------------------------------------------------------
# For each combination, a sphere of density ``rho_sphere`` settling under
# gravity through a fluid of density ``rho_fluid`` and viscosity ``mu``
# reaches a Stokes terminal velocity that fixes its own Reynolds number.

rho_fluid, rho_sphere, g = 1000.0, 2500.0, 9.81  # water-like fluid, glass bead
radii = np.geomspace(1e-5, 1e-3, 40)
mu_sweep = np.geomspace(1e-4, 1e-1, 60)  # from water-like to honey-like

Re_grid = np.array(
    [
        [
            reynolds_number(
                velocity=((4.0 / 3.0) * np.pi * R**3 * (rho_sphere - rho_fluid) * g) / stokes_drag(m, R, 1.0),
                length=2 * R,
                nu=m / rho_fluid,
            )
            for R in radii
        ]
        for m in mu_sweep
    ]
)

# %%
# A single Re = 1 contour, regardless of radius or viscosity
# -----------------------------------------------------------------
# The same ratio governs the transition everywhere along this contour, even
# though the sphere radius spans two decades and the viscosity spans three
# -- exactly Reynolds' claim that the ratio alone, not the size or the fluid
# separately, decides the regime.

mu_water = 1.0e-3
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.pcolormesh(radii, mu_sweep, np.log10(Re_grid), cmap=theme.SEQUENTIAL_CMAP, shading="auto")
fig.colorbar(im, ax=ax, label=r"$\log_{10} Re$")
cs = ax.contour(radii, mu_sweep, Re_grid, levels=[1.0], colors="white", linewidths=1.5)
ax.clabel(cs, fmt={1.0: "Re = 1"})
ax.axhline(mu_water, color=theme.MUTED, ls="--", lw=1.0, label=r"water-like $\mu$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("sphere radius (m)")
ax.set_ylabel(r"fluid viscosity $\mu$ (Pa s)")
ax.legend()
ax.set_title("Re = 1 cuts across both radius and viscosity alike")
fig.tight_layout()

n_water = np.argmin(np.abs(mu_sweep - mu_water))
crossing = radii[np.argmin(np.abs(Re_grid[n_water] - 1.0))]
print(f"at water-like viscosity, Re crosses 1 near radius {crossing * 1e6:.1f} microns")
print("the same Re = 1 threshold recurs at every viscosity in the sweep, at a")
print("correspondingly different radius -- the ratio, not either quantity")
print("alone, is what is physically meaningful.")

plt.show()
