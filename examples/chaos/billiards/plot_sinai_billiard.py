r"""
Sinai Billiard (Chaotic)
========================

A billiard particle moves in a straight line at constant speed inside a
closed boundary and undergoes specular reflection,
:math:`\mathbf{v}' = \mathbf{v} - 2(\mathbf{v}\cdot\mathbf{n})\,\mathbf{n}`,
whenever it strikes the boundary. The Sinai billiard's boundary is a square
cell, :math:`|x| \le L/2,\ |y| \le L/2`, with a circular scatterer of radius
:math:`r_s < L/2` removed from its center, :math:`x^2 + y^2 = r_s^2`. It is a
classic example of a *defocusing* chaotic billiard: because the scatterer is
convex as seen from inside the cell, it disperses nearby trajectories
exponentially fast (unlike the flat walls of a Rectangle billiard, which
neither focus nor defocus). This example plots a single trajectory and the
resulting Poincare section, which fills densely with points rather than
tracing out smooth curves.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.chaos.systems.billiards import SinaiBilliard
from physicskit.chaos.visualizers.dynamic_plots import animate_billiard_trajectory
from physicskit.chaos.visualizers.phase_space import plot_billiard_trajectory, plot_poincare_section

billiard = SinaiBilliard(cell_size=2.0, scatterer_radius=0.5)

# %%
# Animation
# ---------
# Watch the scatterer disperse the ray: notice how quickly it starts
# exploring the whole cell, and how the Poincare section fills in densely
# rather than tracing a smooth curve.
anim = animate_billiard_trajectory(billiard, pos=billiard.sample_interior_point(), vel=(0.4, 0.9), n_bounces=100, interval=50)

plt.show()

# %%
# To save the animation to a file instead of (or in addition to) displaying
# it interactively, use e.g.::
#
#     anim.save("sinai_billiard_animation.gif", writer="pillow", fps=25)

# %%
# Trajectory
# ----------
# Notice how quickly the ray explores the whole cell after just a few bounces
# off the central scatterer.
fig, ax = plot_billiard_trajectory(billiard, pos=billiard.sample_interior_point(), vel=np.array([0.4, 0.9]), n_bounces=100)

# %%
# Poincare section
# ----------------
# A single ray, given enough bounces, would already fill the section densely
# (this is the hallmark of ergodicity); here we pool several shorter rays for
# a quick illustration.
fig, ax = plot_poincare_section(billiard, n_rays=40, n_bounces=200)

plt.show()
