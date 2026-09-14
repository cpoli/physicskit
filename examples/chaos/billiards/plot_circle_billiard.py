r"""
Circle Billiard (Integrable)
============================

A billiard is a point particle moving in a straight line at constant speed
inside a closed boundary, until it strikes the boundary and undergoes
specular reflection -- the outgoing velocity :math:`\mathbf{v}'` obtained
from the incoming velocity :math:`\mathbf{v}` and the local unit normal
:math:`\mathbf{n}` by

.. math::

    \mathbf{v}' = \mathbf{v} - 2(\mathbf{v}\cdot\mathbf{n})\,\mathbf{n}.

The circular billiard, whose boundary is :math:`x^2 + y^2 = r^2`, is the
simplest integrable 2D billiard: the angle of incidence at every bounce is
exactly conserved, so the boundary phase-space coordinate :math:`\sin\phi`
(the sine of the angle between the outgoing velocity and the local boundary
tangent) never changes along a trajectory. This example plots a single
trajectory and the resulting Poincare section.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.chaos.systems.billiards import CircleBilliard
from physicskit.chaos.visualizers.dynamic_plots import animate_billiard_trajectory
from physicskit.chaos.visualizers.phase_space import plot_billiard_trajectory, plot_poincare_section

billiard = CircleBilliard(radius=1.0)

# %%
# Launch off-center
# ------------------------
# The circle's default interior point (used elsewhere for e.g. sampling
# starting positions) is its exact center -- and a ray launched from dead
# center hits the boundary head-on at *every* bounce, reflecting straight
# back through the center each time: a degenerate back-and-forth line, not
# the star-like pattern integrability is famous for. Any off-center point
# avoids this.
pos = (0.3, 0.0)

# %%
# Animation
# ---------
# Watch the ray bounce around the disk, tracing out the star-like pattern
# live alongside its Poincare section filling in point by point.
anim = animate_billiard_trajectory(billiard, pos=pos, vel=(0.3, 1.0), n_bounces=60, interval=50)

plt.show()

# %%
# To save the animation to a file instead of (or in addition to) displaying
# it interactively, use e.g.::
#
#     anim.save("circle_billiard_animation.gif", writer="pillow", fps=25)

# %%
# Trajectory
# ----------
# A single ray, launched from an interior point, bounces around the disk
# forever tracing out a star-like pattern (unless its angle is a rational
# multiple of pi, in which case it eventually closes on itself).
fig, ax = plot_billiard_trajectory(billiard, pos=pos, vel=np.array([0.3, 1.0]), n_bounces=60)

# %%
# Poincare section
# ----------------
# Because the circle is integrable, ``sin(phi)`` is an exact invariant of the
# motion: each ray traces out its own horizontal line in the ``(s,
# sin(phi))`` section, at whatever ``sin(phi)`` its own impact parameter
# fixes, regardless of how many bounces are simulated -- together, many rays
# from the same off-center point foliate the section into a family of such
# lines (rays launched from dead center, by symmetry, would all share the
# same ``sin(phi) = 0`` and collapse onto a single line instead).
fig, ax = plot_poincare_section(billiard, n_rays=25, n_bounces=150, pos=pos)

plt.show()
