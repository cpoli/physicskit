r"""
The Kolmogorov -5/3 cascade in decaying 2D turbulence
=========================================================

:class:`~physicskit.fluids.systems.navier_stokes.NavierStokes2D` integrates
the 2D incompressible vorticity-transport equation
:math:`\partial_t \omega + (\mathbf{u}\cdot\nabla)\omega = \nu \nabla^2
\omega` pseudo-spectrally on a doubly periodic domain, with no external
forcing: once set going, the flow's kinetic energy only decays, redistributed
across scales by the nonlinear advection term before viscosity ultimately
dissipates it. Kolmogorov's 1941 (K41) theory predicts that, in the
"inertial range" of scales between where the flow was initially seeded and
where viscosity dissipates it, the kinetic energy spectrum follows a
universal power law,

.. math::

    E(k) \propto k^{-5/3},

where :math:`E(k)` is defined so that :math:`\int E(k)\,dk` is the total
kinetic energy per unit mass. This example seeds the solver's initial
vorticity field with many randomly placed Gaussian vortex blobs of mixed
sign and random position, lets their nonlinear interactions cascade energy
across scales, and checks the resulting spectrum -- via
:func:`~physicskit.fluids.utils.spectral_analysis.energy_spectrum` -- against
that -5/3 law.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.fluids.systems.navier_stokes import NavierStokes2D
from physicskit.fluids.utils.spectral_analysis import energy_spectrum
from physicskit.fluids.visualizers.flow_fields import plot_vorticity_field
from physicskit.fluids.visualizers.spectra import plot_energy_spectrum

# %%
# Seed many random vortex blobs
# ---------------------------------

n, length = 128, 2 * np.pi
solver = NavierStokes2D(n=n, length=length, nu=2e-4)

rng = np.random.default_rng(0)
n_blobs, core = 40, 0.12
omega0 = np.zeros_like(solver.X)
for _ in range(n_blobs):
    xc, yc = rng.uniform(0, length, size=2)
    sign = rng.choice([-1.0, 1.0])
    dx = np.minimum(np.abs(solver.X - xc), length - np.abs(solver.X - xc))
    dy = np.minimum(np.abs(solver.Y - yc), length - np.abs(solver.Y - yc))
    omega0 += sign * np.exp(-(dx**2 + dy**2) / (2 * core**2))

# %%
# Let the nonlinear interactions cascade energy across scales
# -----------------------------------------------------------------

result = solver.simulate(omega0, dt=0.002, steps=800)

fig, ax = plot_vorticity_field(solver.X, solver.Y, result["omega"], result["u"], result["v"])
ax.set_title("Decaying 2D turbulence")
fig.tight_layout()

# %%
# Check the energy spectrum against Kolmogorov's -5/3 law
# -------------------------------------------------------------
# A genuine inertial range shows up as a stretch of the measured spectrum
# running parallel to the reference line on these log-log axes.

k, E = energy_spectrum(result["u"], result["v"], length)
fig, ax = plot_energy_spectrum(k, E)
fig.tight_layout()

plt.show()
