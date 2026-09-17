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
