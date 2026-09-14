r"""
Trajectory Color-Coding
===========================

Coloring a trajectory by a per-point scalar -- rather than drawing it in a
single flat color -- can make a plot's structure much easier to read: which
parts of a billiard came early or late, or where a continuous trajectory is
moving fast versus slow. Two systems are colored below: the Bunimovich
stadium billiard (two straight edges joined by semicircular caps of radius
:math:`r`, with specular reflection :math:`\mathbf{v}' = \mathbf{v} -
2(\mathbf{v}\cdot\mathbf{n})\,\mathbf{n}` at the boundary), and the Lorenz
attractor,

.. math::

    \dot{x} = \sigma (y - x), \qquad \dot{y} = x (\rho - z) - y, \qquad
    \dot{z} = x y - \beta z.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.chaos.systems.billiards import BunimovichStadium
from physicskit.chaos.systems.continuous import Lorenz
from physicskit.chaos.visualizers.dynamic_plots import plot_colored_trajectory, plotly_3d_trajectory

# %%
# A billiard trajectory colored by bounce index
# --------------------------------------------------
# Coloring by "time" (here, which bounce a segment belongs to) shows how
# quickly -- or slowly -- a chaotic billiard trajectory explores the table:
# watch for whether early (dark) segments stay clustered while later
# (bright) segments have spread across the whole table.
billiard = BunimovichStadium(radius=1.0, straight_length=2.0)
path, _ = billiard.trajectory_segments(billiard.sample_interior_point(), vel=(0.5, 0.9), n_bounces=150)
bounce_index = np.arange(path.shape[0])

fig, ax = plot_colored_trajectory(path[:, 0], path[:, 1], bounce_index, colorbar_label="bounce index")
boundary = billiard.boundary_polyline()
ax.plot(boundary[:, 0], boundary[:, 1], color="black", lw=1.0)
ax.set_aspect("equal")
ax.set_title("Billiard trajectory colored by bounce index")

# %%
# An interactive 3D attractor colored by local speed
# ---------------------------------------------------------
# :func:`physicskit.chaos.visualizers.dynamic_plots.plotly_3d_trajectory` gives an
# orbit/pan/zoom-able 3D view; coloring by local speed highlights where the
# Lorenz attractor's flow accelerates (near the "wing crossings") versus
# where it lingers (spiraling within a wing).
system = Lorenz()
_, states = system.trajectory(n_steps=20000, dt=0.01)
states = states[500:]
speed = np.linalg.norm(np.diff(states, axis=0), axis=1) / 0.01

fig3d = plotly_3d_trajectory(states[1:], color_by=speed, colorbar_label="local speed", title="Lorenz attractor by speed")
fig3d.show()

plt.show()
