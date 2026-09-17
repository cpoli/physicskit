r"""
Interactive Plotly visualizations
======================================

Every other visualizer in physicskit.statphys is Matplotlib-based: static, ideal for
publication figures and this documentation's gallery.
:mod:`physicskit.statphys.visualizers.interactive` renders the same information as
pan/zoom/hover-enabled Plotly figures instead, convenient for
interactively exploring a parameter sweep or a dense vector field in a
Jupyter notebook. This example builds one such figure for each of three
lattice/particle models used throughout this gallery -- the 2D Ising
ferromagnet, the 2D XY model, and a Lennard-Jones gas -- and saves each as
a standalone HTML file.
"""

import numpy as np

from physicskit.statphys.chapters.ising_lattice import Ising2D, XYModel2D
from physicskit.statphys.chapters.molecular_dynamics import LennardJonesGas
from physicskit.statphys.visualizers.interactive import (
    interactive_particle_snapshot,
    interactive_temperature_sweep,
    interactive_vortex_field,
)

# %%
# An interactive temperature sweep
# ------------------------------------
# The 2D Ising ferromagnet, :math:`H = -J \sum_{\langle i,j \rangle} s_i
# s_j` with :math:`s_i = \pm 1` on a periodic :math:`L \times L` lattice, is
# swept in temperature across its critical point :math:`T_C` with the
# Wolff cluster algorithm; the resulting figure lets you hover over the
# energy, order parameter, specific heat, and susceptibility curves.
model = Ising2D(L=20, seed=0)
temperatures = np.linspace(model.T_C - 1.0, model.T_C + 1.0, 10)
result = model.run_temperature_sweep(temperatures, n_equil=100, n_measure=150, algorithm="wolff")
fig = interactive_temperature_sweep(result, T_c=model.T_C)
fig.write_html("ising_sweep_interactive.html")

# %%
# An interactive vortex field
# ------------------------------------
# The 2D XY model, :math:`H = -J \sum_{\langle i,j \rangle}
# \cos(\theta_i - \theta_j)` with continuous planar spin angles
# :math:`\theta_i`, is equilibrated exactly at its Kosterlitz-Thouless
# transition temperature :math:`T_{\text{KT}}`, where bound
# vortex-antivortex pairs are just beginning to unbind; the interactive
# field lets you pan and zoom into individual vortex cores.
xy_model = XYModel2D(L=20, seed=1)
xy_model.sweep(beta=1.0 / xy_model.T_KT, n_sweeps=300)
fig = interactive_vortex_field(xy_model.theta)
fig.write_html("xy_vortices_interactive.html")

# %%
# An interactive particle snapshot
# ------------------------------------
# A 2D gas of :math:`N=100` particles interacting through the
# Lennard-Jones potential :math:`V(r) = 4\epsilon[(\sigma/r)^{12} -
# (\sigma/r)^{6}]` is integrated forward in a periodic box; the
# interactive scatter plot colors each particle by its instantaneous
# speed, hoverable per-particle.
gas = LennardJonesGas(n_particles=100, box_size=15.0, seed=2)
gas.step(n_steps=500)
fig = interactive_particle_snapshot(gas)
fig.write_html("lj_gas_interactive.html")

print("Wrote 3 standalone interactive HTML files.")
