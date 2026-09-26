r"""
Poincaré and the three-body problem
======================================

Poincaré's 1890 memoir on the three-body problem studied its simplest
non-trivial case: the *restricted* problem, a massless body moving in the
field of two primaries on circular orbits. In the frame co-rotating with
the primaries (mass ratio :math:`\mu`, unit separation, unit angular
velocity),

.. math::

    \ddot x - 2\dot y = \partial_x\Omega, \qquad
    \ddot y + 2\dot x = \partial_y\Omega, \qquad
    \Omega = \frac{x^2+y^2}{2} + \frac{1-\mu}{r_1} + \frac{\mu}{r_2},

with one conserved quantity, the Jacobi constant
:math:`C = 2\Omega - \dot x^2 - \dot y^2`. With two degrees of freedom
and only one integral, the motion can't be solved in closed form. To
see it anyway, Poincaré recorded only where orbits cross a fixed slice,
the *surface of section*. Here that is :math:`y=0,\ \dot y>0`, computed
with :func:`~physicskit.classical.visualizers.phase_space.poincare_section`.
Regular orbits trace closed curves; the others fill a chaotic region,
where nearby orbits separate exponentially.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from physicskit.classical.visualizers.phase_space import poincare_section
from physicskit.integrators import rk4_integrate

MU = 0.01215  # Earth-Moon mass ratio


@njit
def r3bp_rhs(s, t, params):
    mu = params[0]
    x, y, vx, vy = s[0], s[1], s[2], s[3]
    r1 = np.sqrt((x + mu) ** 2 + y**2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y**2)
    ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    ay = -2 * vx + y - (1 - mu) * y / r1**3 - mu * y / r2**3
    return np.array([vx, vy, ax, ay])


def jacobi(s, mu=MU):
    x, y, vx, vy = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    r1 = np.sqrt((x + mu) ** 2 + y**2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y**2)
    return x**2 + y**2 + 2 * (1 - mu) / r1 + 2 * mu / r2 - vx**2 - vy**2


def launch(x0, C, mu=MU):
    """State on the y=0 line with vx=0 and vy>0 fixed by the Jacobi constant."""
    s = np.array([x0, 0.0, 0.0, 0.0])
    vy2 = jacobi(s, mu) - C
    return None if vy2 <= 0 else np.array([x0, 0.0, 0.0, np.sqrt(vy2)])


params = np.array([MU])
dt, n_steps = 2e-3, 250_000

# %%
# The surface of section
# --------------------------
# Orbits around the larger primary at a Jacobi constant just above the
# one that would let them escape through the inner Lagrange point
# (:math:`C_{L_1}\approx3.188` for Earth-Moon). Each colour is one orbit.
C = 3.20
fig1, ax1 = plt.subplots(figsize=(6.5, 5))
drift = 0.0
for i, x0 in enumerate(np.linspace(0.15, 0.62, 16)):
    s0 = launch(x0, C)
    if s0 is None:
        continue
    _, traj = rk4_integrate(r3bp_rhs, s0, 0.0, dt, n_steps, params)
    drift = max(drift, np.max(np.abs(jacobi(traj) - C)))
    cq, cp = poincare_section(traj[:, 0], traj[:, 2], traj[:, 1], traj[:, 3], value=0.0, direction=1)
    ax1.scatter(cq, cp, s=0.6, color=plt.cm.turbo(i / 15))
ax1.set_xlim(0.05, 0.85)
ax1.set_xlabel("x")
ax1.set_ylabel(r"$\dot x$")
ax1.set_title(f"Restricted three-body problem, section y=0, C={C}")
fig1.tight_layout()
print(f"max Jacobi-constant drift over all orbits: {drift:.1e}")

# %%
# Sensitive dependence on initial conditions
# ----------------------------------------------
# Two orbits launched :math:`10^{-9}` apart. On a regular orbit the gap
# grows only linearly in time; in the chaotic region it grows
# exponentially until it is as large as the orbit itself.
fig2, ax2 = plt.subplots(figsize=(6, 3.8))
for x0, label, color in [(0.50, "regular orbit", "steelblue"), (0.20, "chaotic orbit", "firebrick")]:
    s0 = launch(x0, C)
    t, a = rk4_integrate(r3bp_rhs, s0, 0.0, dt, 100_000, params)
    _, b = rk4_integrate(r3bp_rhs, s0 + np.array([1e-9, 0, 0, 0]), 0.0, dt, 100_000, params)
    sep = np.linalg.norm(a[:, :2] - b[:, :2], axis=1)
    ax2.semilogy(t, sep, color=color, label=f"{label}, x0={x0}")
    print(f"{label:14s}: separation after t={t[-1]:.0f} is {sep[-1]:.1e}")
ax2.set_xlabel(r"time (primaries orbit once every $2\pi$)")
ax2.set_ylabel("separation")
ax2.set_title(r"Nearby orbits, $10^{-9}$ apart")
ax2.legend(fontsize=8)
fig2.tight_layout()

plt.show()
