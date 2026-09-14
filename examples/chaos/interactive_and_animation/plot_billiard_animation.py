r"""
Animated Billiard Trajectory
=============================

The billiard animated here is the Bunimovich stadium: two straight edges,
:math:`y=\pm r` for :math:`|x|\le a`, joined by semicircular caps of radius
:math:`r` centered at :math:`(\pm a, 0)`. A billiard particle moves in a
straight line at constant speed and undergoes specular reflection off this
boundary,

.. math::

    \mathbf{v}' = \mathbf{v} - 2(\mathbf{v}\cdot\mathbf{n})\,\mathbf{n},

whenever it strikes it. :func:`physicskit.chaos.visualizers.dynamic_plots.animate_billiard_trajectory`
builds a Matplotlib ``FuncAnimation`` showing the ball bouncing inside the
stadium (left panel) alongside its Poincare section filling in live, point
by point, as each new bounce occurs (right panel).
"""

import matplotlib.pyplot as plt

from physicskit.chaos.systems.billiards import BunimovichStadium
from physicskit.chaos.visualizers.dynamic_plots import animate_billiard_trajectory

billiard = BunimovichStadium(radius=1.0, straight_length=2.0)

# %%
# Build the animation
# ---------------------
# Assign the animation to a variable to keep it alive; a reference must
# survive until it is displayed (``plt.show()``) or saved (``anim.save(...)``)
# or Matplotlib will silently drop it.
anim = animate_billiard_trajectory(billiard, pos=billiard.sample_interior_point(), vel=(0.5, 0.9), n_bounces=100, interval=50)

plt.show()

# %%
# To save the animation to a file instead of (or in addition to) displaying
# it interactively, use e.g.::
#
#     anim.save("stadium_animation.gif", writer="pillow", fps=30)
