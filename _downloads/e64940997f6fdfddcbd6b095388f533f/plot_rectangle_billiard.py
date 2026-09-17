r"""
Rectangle Billiard (Integrable)
================================

A billiard particle moves in a straight line at constant speed inside a
closed boundary and undergoes specular reflection,
:math:`\mathbf{v}' = \mathbf{v} - 2(\mathbf{v}\cdot\mathbf{n})\,\mathbf{n}`,
whenever it strikes the boundary. Here the boundary is the rectangle
:math:`|x| \le w/2,\ |y| \le h/2` of width :math:`w` and height :math:`h`,
whose walls are all axis-aligned, so :math:`\mathbf{n}` is always
:math:`\pm\hat{x}` or :math:`\pm\hat{y}` -- reflections off the horizontal
and vertical walls therefore only ever flip the sign of one velocity
component. The rectangular billiard is integrable: each individual ray only
ever visits a handful of discrete :math:`\sin\phi` values (the sine of the
angle between the outgoing velocity and the local boundary tangent). This
example plots a single trajectory and a Poincare section built from several
rays launched at different angles.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.chaos.systems.billiards import RectangleBilliard
from physicskit.chaos.visualizers.dynamic_plots import animate_billiard_trajectory
from physicskit.chaos.visualizers.phase_space import plot_billiard_trajectory, plot_poincare_section

billiard = RectangleBilliard(width=2.0, height=1.0)

# %%
# Animation
# ---------
anim = animate_billiard_trajectory(billiard, pos=billiard.sample_interior_point(), vel=(1.0, 0.35), n_bounces=100, interval=50)

plt.show()

# %%
# To save the animation to a file instead of (or in addition to) displaying
# it interactively, use e.g.::
#
#     anim.save("rectangle_billiard_animation.gif", writer="pillow", fps=25)

# %%
# Trajectory
# ----------
fig, ax = plot_billiard_trajectory(billiard, pos=billiard.sample_interior_point(), vel=np.array([1.0, 0.35]), n_bounces=60)

# %%
# Poincare section
# ----------------
# Each ray only ever hits walls at two possible incidence angles (one for the
# horizontal walls, one for the vertical walls), so the section is made up of
# a discrete set of horizontal line segments rather than the dense chaotic
# sea seen in defocusing billiards.
fig, ax = plot_poincare_section(billiard, n_rays=25, n_bounces=150)

plt.show()
