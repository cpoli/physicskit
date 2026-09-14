r"""
A phase-colored tunneling animation
======================================

Propagates a wavepacket initially localized in the left well of a double
well :math:`V(x) = \lambda(x^2-a^2)^2` with the real FFT split-operator
solver (not the two-level analytic shortcut used in the double-well
example), and renders it with :func:`~physicskit.quantum.visualizers.wavefunctions.animate_density`'s
HSV phase coloring: hue tracks the local complex phase
:math:`\arg\psi(x,t)`, so the accumulating phase gradient as the packet
tunnels back and forth is directly visible, not just the probability
density.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.quantum.chapters.potentials import DoubleWellSimulator
from physicskit.quantum.core.solvers import SplitOperatorSolver1D
from physicskit.quantum.visualizers.wavefunctions import animate_density, plot_complex_wavefunction

dw = DoubleWellSimulator(lam=0.3, a=1.5)
result = dw.solve(n_states=2)
print(f"tunneling period: {result.tunneling_period:.3f}")

# Real (not two-level-analytic) propagation of a left-localized wavepacket,
# so the phase-colored animation reflects the full FFT dynamics.
psi0 = dw.localized_state(result, side="left").astype(complex)
solver = SplitOperatorSolver1D(dw.x, dw.V, dt=2e-4)
psi0 /= np.sqrt(solver.norm(psi0))

t_max = 1.2 * result.tunneling_period
n_steps = int(t_max / solver.dt)
save_every = max(n_steps // 80, 1)
frames, times = solver.propagate(psi0, n_steps, save_every=save_every)
print(f"norm conservation check: {solver.norm(frames[-1]):.8f}")

# %%
# The animation: hue sweeps across the whole wavefunction as its global
# phase accumulates while the density tunnels back and forth between wells.

anim = animate_density(dw.x, frames, times=times, interval=60, phase_colored=True)

# %%
# A static filmstrip (six phase-colored snapshots) for a quick,
# non-animated look at the same dynamics.

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
snapshot_idx = np.linspace(0, len(frames) - 1, 6).astype(int)
for ax, i in zip(axes.flat, snapshot_idx):
    plot_complex_wavefunction(dw.x, frames[i], ax=ax)
    ax.set_title(f"t = {times[i]:.2f}")
    ax.set_xlim(-4, 4)
fig.tight_layout()

# %%
# A continuous space-time density map of the same propagation
# --------------------------------------------------------------
#
# The six discrete snapshots above sample the same ``frames`` stack already
# produced by the FFT propagation; imaging :math:`|\psi(x,t)|^2` over the
# full (x,t) grid at once shows the tunneling back and forth as continuous
# diagonal-ish bands, rather than six isolated instants.

density_st = np.abs(frames) ** 2
fig2, ax2 = plt.subplots(figsize=(8, 5))
im = ax2.pcolormesh(dw.x, times, density_st, shading="auto", cmap="inferno")
ax2.set_xlim(-4, 4)
ax2.set_xlabel("x")
ax2.set_ylabel("t")
ax2.set_title(r"Space-time $|\psi(x,t)|^2$: tunneling back and forth between wells")
fig2.colorbar(im, ax=ax2, label=r"$|\psi(x,t)|^2$")
fig2.tight_layout()
