r"""
Bunimovich Stadium (Chaotic)
============================

A billiard particle moves in a straight line at constant speed inside a
closed boundary and undergoes specular reflection,
:math:`\mathbf{v}' = \mathbf{v} - 2(\mathbf{v}\cdot\mathbf{n})\,\mathbf{n}`,
whenever it strikes the boundary. The Bunimovich stadium's boundary is two
straight edges, :math:`y = \pm r` for :math:`|x| \le a`, joined by two
semicircular caps of radius :math:`r` centered at :math:`(\pm a, 0)`, where
:math:`2a` is the straight-edge length. It is chaotic despite having no
concave scatterer: the *focusing* semicircular caps defocus nearby
trajectories after they refocus and diverge past the caps' centers of
curvature (unlike a full circle, whose focusing never gets interrupted by a
flat stretch). This example plots a single trajectory and the resulting
Poincare section.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.chaos.systems.billiards import BunimovichStadium
from physicskit.chaos.visualizers.phase_space import plot_billiard_trajectory, plot_poincare_section

billiard = BunimovichStadium(radius=1.0, straight_length=2.0)

# %%
# Trajectory
# ----------
fig, ax = plot_billiard_trajectory(billiard, pos=billiard.sample_interior_point(), vel=np.array([0.5, 0.9]), n_bounces=100)

# %%
# Poincare section
# ----------------
# As with the Sinai billiard, the section fills in densely rather than
# tracing out smooth invariant curves.
fig, ax = plot_poincare_section(billiard, n_rays=40, n_bounces=200)

plt.show()
