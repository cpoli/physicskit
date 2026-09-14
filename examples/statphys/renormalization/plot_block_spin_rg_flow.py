r"""
Kadanoff block-spin renormalization group flow
====================================================

Real-space renormalization group (RG) transformations reveal *why*
critical phenomena are universal. Starting from an Ising configuration
(:math:`H = -J \sum_{\langle i,j \rangle} s_i s_j`, :math:`s_i = \pm 1`)
equilibrated at temperature :math:`T`, Kadanoff's construction repeatedly
replaces each non-overlapping :math:`2\times2` block of spins by a single
effective spin via majority rule,

.. math::

    s'_{I} = \mathrm{sign}\!\left(\sum_{i \,\in\, \text{block } I} s_i\right),

with an exact tie broken by a fair coin flip -- mapping the lattice onto a
smaller one that behaves as if sampled at some effective temperature
:math:`T'`. Iterating this map drives almost any starting configuration to
one of three fixed points: a fully ordered lattice (:math:`T \ll T_C`), a
fully disordered one (:math:`T \gg T_C`), or -- exactly at :math:`T_C` --
a self-similar configuration that looks statistically the same at every
coarse-graining step, tracked here via the order parameter :math:`|\langle
s \rangle|`, the visual and quantitative signature of scale invariance at
a critical point.
"""

import matplotlib.pyplot as plt

from physicskit.statphys.chapters.ising_lattice import Ising2D
from physicskit.statphys.chapters.renormalization import BlockSpinRG
from physicskit.statphys.visualizers.rg_render import plot_rg_flow

T_C = Ising2D().T_C

# %%
# RG flow from three starting temperatures
# ----------------------------------------------
for T, label in [(0.5 * T_C, "T << T_C"), (T_C, "T = T_C"), (2.0 * T_C, "T >> T_C")]:
    rg = BlockSpinRG(L=64, T=T, J=1.0, n_equil_sweeps=400, seed=0)
    grids = rg.iterate(n_steps=4)
    orders = [rg.order_parameter(g) for g in grids]

    axes = plot_rg_flow(grids, titles=[f"L={g.shape[0]}\n|m|={m:.2f}" for g, m in zip(grids, orders)])
    plt.gcf().suptitle(f"Block-spin RG flow, {label} = {T:.2f}")
    plt.tight_layout()

plt.show()
