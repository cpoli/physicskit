r"""
Quantum Baker's Map
========================

The quantum baker's map is the quantization of physicskit.chaos's classical
:class:`~physicskit.chaos.systems.maps.BakersMap`,
:math:`T(x,y)=(x/\alpha,\ \alpha y)` for :math:`x<\alpha` and
:math:`T(x,y)=((x-\alpha)/(1-\alpha),\ \alpha+(1-\alpha)y)` for
:math:`x\ge\alpha`. It is built (following Balazs-Voros/Saraceno) as a
one-period unitary Floquet operator that applies the discrete Fourier
transform separately to the :math:`q < \alpha` and :math:`q \ge \alpha`
halves of the (finite, ``dim``-dimensional) position-basis Hilbert space,
then transforms the result back to the full position representation -- the
quantum-mechanical echo of the classical map's "stretch, cut, and stack"
mechanism. It is one of the simplest exactly-solvable models of quantum
chaos: the underlying classical map is uniformly hyperbolic everywhere (see
:doc:`/api/gallery/chaos/maps/plot_bakers_map`), with none of the mixed
regular/chaotic phase space the kicked rotor has.
"""

import matplotlib.pyplot as plt

from physicskit.chaos.quantum import QuantumBakersMap
from physicskit.chaos.visualizers import animate_husimi_evolution, plot_husimi, plot_quantum_spectrum

qbm = QuantumBakersMap(dim=120, alpha=0.5)

# %%
# The Floquet spectrum
# ------------------------
# As for the kicked rotor, one map iteration is a unitary Floquet operator
# whose eigenvalues live on the unit circle.
fig, ax = plot_quantum_spectrum(qbm.eigenphases())
ax.set_title(f"Quantum baker's map (dim={qbm.dim}) Floquet eigenphases")

plt.show()

# %%
# Animation: a wavepacket mixing across the unit square
# ------------------------------------------------------------
# Start a minimum-uncertainty wavepacket in a corner of phase space and
# iterate the map: the same stretch/cut/stack mechanism that mixes the
# classical baker's map (see the classical example's stripe-thinning figure)
# mixes the wavepacket's Husimi distribution, which spreads across the whole
# unit square within just a few iterations -- there is no room for it to
# settle onto a regular island, because the classical map has none.
psi0 = qbm.coherent_state(q0=0.15, p0=0.15)
states = qbm.evolve(psi0, n_steps=6)
anim = animate_husimi_evolution(qbm, states, resolution=90, interval=350, title=f"QuantumBakersMap (dim={qbm.dim})")

plt.show()

# %%
# To save the animation to a file instead of (or in addition to) displaying
# it interactively, use e.g.::
#
#     anim.save("quantum_bakers_map_animation.gif", writer="pillow", fps=3)

# %%
# A single iteration, up close
# ---------------------------------
# Before mixing takes over, a single iteration already shows the map's
# signature move: the coherent state's phase-space cell -- initially a small
# round blob -- gets stretched along ``q`` and squeezed along ``p``, then cut
# and stacked, in exact correspondence with the classical map's action.
fig2, axes = plt.subplots(1, 2, figsize=(12, 5.5))
q_grid, p_grid, husimi0 = qbm.husimi(states[0], resolution=100)
plot_husimi(q_grid, p_grid, husimi0, ax=axes[0], title="Before (iteration 0)")
q_grid, p_grid, husimi1 = qbm.husimi(states[1], resolution=100)
plot_husimi(q_grid, p_grid, husimi1, ax=axes[1], title="After (iteration 1)")
fig2.suptitle("Quantum baker's map: one iteration's stretch-cut-stack, in Husimi phase space")
fig2.tight_layout()

plt.show()
