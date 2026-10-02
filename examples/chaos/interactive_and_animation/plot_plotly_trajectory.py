r"""
Interactive Plotly Figures
============================

Alongside the Matplotlib helpers, :mod:`physicskit.chaos.visualizers.dynamic_plots`
provides interactive `Plotly <https://plotly.com/python/>`_ figures: a
trajectory you can pan and zoom, and an interactive Poincare section you can
hover over to inspect individual bounce points. The billiard used here is
the Sinai billiard: a square cell, :math:`|x|\le L/2,\ |y|\le L/2`, with a
circular scatterer of radius :math:`r_s` removed from its center,
:math:`x^2+y^2=r_s^2`, off of which the particle undergoes specular
reflection, :math:`\mathbf{v}' = \mathbf{v} - 2(\mathbf{v}\cdot\mathbf{n})
\,\mathbf{n}`, whenever it strikes the boundary.
"""

import numpy as np

from physicskit.chaos.systems.billiards import SinaiBilliard
from physicskit.chaos.visualizers.dynamic_plots import plotly_billiard_trajectory, plotly_poincare_section

billiard = SinaiBilliard(cell_size=2.0, scatterer_radius=0.5)

# %%
# Trajectory
# ----------
fig_traj = plotly_billiard_trajectory(billiard, pos=billiard.sample_interior_point(), vel=(0.4, 0.9), n_bounces=150)
fig_traj.show()

# %%
# Poincare section
# -----------------
fig_poincare = plotly_poincare_section(billiard, n_rays=40, n_bounces=200, seed=0)
fig_poincare.show()

# %%
# Check
# -----
# Every bounce of Sinai's billiard lands on the cell wall or on the
# scatterer, never inside it, and the speed is conserved.
run = billiard.simulate(billiard.sample_interior_point(), (0.4, 0.9), n_bounces=150)
assert np.hypot(run["x"], run["y"]).min() >= 0.5 - 1e-9
on_wall = np.isclose(np.abs(run["x"]), 1.0) | np.isclose(np.abs(run["y"]), 1.0)
on_disk = np.isclose(np.hypot(run["x"], run["y"]), 0.5)
assert np.all(on_wall | on_disk) and np.ptp(np.hypot(run["vx"], run["vy"])) < 1e-12
