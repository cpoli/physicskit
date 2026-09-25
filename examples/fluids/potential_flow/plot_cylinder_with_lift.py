r"""
Flow past a lifting cylinder
==============================

Superposing a uniform stream of speed :math:`U_\infty` with a doublet of
strength :math:`\kappa = 2\pi U_\infty R^2` places a circular streamline of
radius :math:`R` exactly at the origin -- a solid cylinder, in
potential-flow terms. Adding a point vortex of circulation :math:`\Gamma` at
the same location keeps that circle intact (a vortex centered on the circle
induces purely tangential velocity there) while breaking its front-back
symmetry, generating lift with no viscosity anywhere in the calculation.
This example builds that flow with
:func:`~physicskit.fluids.systems.potential_flow.flow_past_cylinder`, plots
its surface pressure coefficient

.. math::

    C_p = 1 - \frac{u^2+v^2}{U_\infty^2},

and checks the resulting lift against the Kutta-Joukowski theorem,

.. math::

    L' = -\rho\,U_\infty\,\Gamma,

(:math:`\Gamma>0` counterclockwise; the clockwise :math:`\Gamma<0` used
here speeds up the flow over the top and lifts upward, like an airfoil)
(:func:`~physicskit.fluids.systems.potential_flow.kutta_joukowski_lift`), by
directly integrating that same pressure around the surface.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.fluids.systems.potential_flow import flow_past_cylinder, kutta_joukowski_lift
from physicskit.fluids.visualizers.flow_fields import plot_streamlines
from physicskit.fluids.visualizers.potential_flow import plot_pressure_coefficient

# %%
# Build the flow
# ---------------
# Circulation is chosen large enough to make the effect obvious; too much
# circulation would pull the (physically required) stagnation points off
# the cylinder surface entirely.

U_inf, R, rho = 1.0, 1.0, 1.0
Gamma = -3.0  # clockwise: faster flow over the top, upward lift
flow = flow_past_cylinder(U_inf=U_inf, radius=R, circulation=Gamma)

x = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, x, indexing="ij")
u, v = flow.velocity(X, Y)
inside = X**2 + Y**2 < R**2
u[inside] = np.nan
v[inside] = np.nan

fig, ax = plot_streamlines(X, Y, u, v)
theta_circle = np.linspace(0, 2 * np.pi, 200)
ax.plot(R * np.cos(theta_circle), R * np.sin(theta_circle), color="black", lw=1.5)
ax.set_title(f"Cylinder with circulation $\\Gamma$ = {Gamma}")
fig.tight_layout()

# %%
# Surface pressure coefficient
# ------------------------------
# Circulation shifts the two stagnation points (where :math:`C_p = 1`) away
# from the front/back symmetry point they'd sit at with no circulation, and makes
# the low-pressure suction peak on top stronger than the one on the bottom
# -- exactly the pressure asymmetry that produces lift.

theta = np.linspace(0, 2 * np.pi, 200)
x_surface, y_surface = R * np.cos(theta), R * np.sin(theta)
Cp = flow.pressure_coefficient(x_surface, y_surface)

fig, ax = plot_pressure_coefficient(theta, Cp)
fig.tight_layout()

# %%
# Cross-checking the Kutta-Joukowski theorem
# ---------------------------------------------
# Integrating the surface pressure directly should reproduce
# :func:`~physicskit.fluids.systems.potential_flow.kutta_joukowski_lift`'s
# closed-form answer, :math:`L' = -\rho U_\infty \Gamma`.

theta_fine = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
x_fine, y_fine = R * np.cos(theta_fine), R * np.sin(theta_fine)
Cp_fine = flow.pressure_coefficient(x_fine, y_fine)
p_fine = Cp_fine * (0.5 * rho * U_inf**2)
dtheta = theta_fine[1] - theta_fine[0]
lift_numeric = -np.sum(p_fine * np.sin(theta_fine) * R) * dtheta  # y-component of -oint p n dA
lift_theory = kutta_joukowski_lift(rho=rho, U_inf=U_inf, circulation=Gamma)

print(f"lift from pressure integral: {lift_numeric:.4f}")
print(f"lift from Kutta-Joukowski theorem (-rho * U_inf * Gamma): {lift_theory:.4f}")

plt.show()
