r"""
Rossler Attractor
=================

The Rossler system is three coupled nonlinear autonomous ODEs, deliberately
designed to be simpler than Lorenz while still chaotic:

.. math::

    \dot{x} &= -y - z \\
    \dot{y} &= x + a y \\
    \dot{z} &= b + z (x - c)

It produces a simpler-looking, ribbon-like strange attractor than Lorenz,
built from a single stretch-and-fold mechanism: trajectories spiral
outward in the :math:`(x, y)` plane (driven by the linear :math:`-y-z` and
:math:`x+ay` terms) until :math:`x` exceeds :math:`c`, at which point the
:math:`z (x - c)` term kicks the trajectory up and folds it back down onto
the spiral. This example integrates a trajectory with
:meth:`physicskit.chaos.systems.continuous.Rossler.trajectory` and plots it
in 3D, using the classic chaotic parameters :math:`a=0.2`, :math:`b=0.2`,
:math:`c=5.7`.
"""

import matplotlib.pyplot as plt

from physicskit.chaos.systems.continuous import Rossler
from physicskit.chaos.visualizers.section import plot_poincare_map

system = Rossler(a=0.2, b=0.2, c=5.7)

# %%
# Integrate
# ---------
t, states = system.trajectory(n_steps=40000, dt=0.02)
states = states[1000:]

# %%
# Plot
# ----
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(projection="3d")
ax.plot(states[:, 0], states[:, 1], states[:, 2], lw=0.4, color="teal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Rossler attractor")

plt.show()

# %%
# Poincare section: the stretch-and-fold mechanism, laid bare
# ------------------------------------------------------------------
# Sampling the flow only at the instants it crosses the plane :math:`y = 0`
# moving outward (:math:`\dot{y} > 0`) -- the natural section for a spiral
# that winds around the :math:`z`-axis -- turns the ribbon-like 3D attractor
# into a near-one-dimensional curve of successive crossings'
# :math:`(x, z)`. That the points collapse almost onto a single folded curve
# (rather than filling a 2D patch) is the direct visual signature of the
# single stretch-and-fold mechanism described above: each pass around the
# spiral maps an interval of :math:`x` values into a slightly different,
# folded-over interval, iteration after iteration.
fig2, ax2 = plot_poincare_map(
    system,
    system.initial_state(),
    coord=1,
    value=0.0,
    direction=1.0,
    plot_coords=(0, 2),
    t_max=400.0,
    dt=0.005,
    s=4,
)
ax2.set_xlabel("x")
ax2.set_ylabel("z")
ax2.set_title("Rossler Poincare section at y = 0 (ascending)")

plt.show()
