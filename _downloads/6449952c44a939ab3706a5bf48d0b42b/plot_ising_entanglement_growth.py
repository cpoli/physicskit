r"""
Dynamical entanglement generation
=====================================

Two qubits, each individually prepared in the unbiased state
:math:`\lvert+\rangle = (\lvert0\rangle+\lvert1\rangle)/\sqrt2` -- an
unentangled product state -- are coupled by an Ising interaction
:math:`\hat H = \hbar J\,\sigma_z^{(1)}\otimes\sigma_z^{(2)}` and become
genuinely entangled purely through that interaction's time evolution,
reaching a maximally entangled (Bell-equivalent) state at :math:`Jt=\pi/4`.
Unlike the static Bell/EPR correlations of
:doc:`/api/gallery/quantum/entanglement/plot_entanglement_and_topology`
(which start from an already-entangled state), this traces entanglement's
actual dynamical genesis.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.quantum.chapters.entanglement import IsingEntangler
from physicskit.quantum.visualizers.entanglement import animate_entanglement_growth

ising = IsingEntangler(J=1.0)
print(f"concurrence of the initial product state: {ising.concurrence(ising.initial_state()):.2e}")

t_max = np.pi / (2 * ising.J)  # spans the first entangling half-cycle: 0 -> pi/4 (max) -> pi/2 (back to 0)
times = np.linspace(0, t_max, 200)
concurrence = ising.concurrence_trajectory(times)
purity = np.array([ising.purity(ising.state(t)) for t in times])

# %%
# Concurrence grows from 0 (product state) to 1 (maximally entangled) at
# :math:`Jt=\pi/4`, while the reduced-state purity falls from 1 to 1/2 in
# lockstep -- the two faces of the same entangling process.
# --------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(times * ising.J, concurrence, label="concurrence C(t)")
ax.plot(times * ising.J, purity, label=r"purity $\mathrm{Tr}(\rho_1^2)$")
ax.axvline(np.pi / 4, color="gray", ls="--", lw=0.8, label="Jt = pi/4 (maximal entanglement)")
ax.set_xlabel("J t")
ax.set_title("Two qubits entangling under an Ising ZZ coupling")
ax.legend(fontsize=8)
fig.tight_layout()

print(f"max concurrence reached: {concurrence.max():.6f}")
print(f"purity at max concurrence: {purity[np.argmax(concurrence)]:.6f} (expect 0.5)")

# %%
# An animated view of the buildup
# ------------------------------------
#
# :func:`~physicskit.quantum.visualizers.entanglement.animate_entanglement_growth`
# traces the concurrence curve progressively, with a marker at the current
# time, as the Ising coupling acts.

anim = animate_entanglement_growth(times, concurrence, ylabel="concurrence")
# anim.save("entanglement_growth.gif", writer="pillow", fps=15)

# %%
# The reduced single-qubit density matrix itself, from pure to maximally
# mixed and back
# --------------------------------------------------------------------------
#
# Concurrence and purity above summarize the reduced state with a single
# number each; :meth:`~physicskit.quantum.chapters.entanglement.IsingEntangler.reduced_density_matrix`
# gives the full :math:`2\times2` matrix. Its magnitude visibly loses its
# off-diagonal coherence as entanglement grows -- the pure product state
# :math:`\lvert+\rangle\langle+\rvert` has :math:`\lvert\rho_1\rvert=0.5`
# everywhere, while the maximally entangled point leaves a purely diagonal,
# maximally mixed :math:`\rho_1=I/2` -- and regains it as the qubits
# disentangle again on the way back to :math:`Jt=\pi/2`.

times_rho = np.array([0.0, t_max / 6, t_max / 3, t_max / 2])
fig2, axes2 = plt.subplots(1, len(times_rho), figsize=(11, 3.3))
for ax, t_snap in zip(axes2, times_rho):
    psi_snap = ising.state(t_snap)
    rho1 = ising.reduced_density_matrix(psi_snap)
    im2 = ax.imshow(np.abs(rho1), vmin=0, vmax=0.5, cmap="viridis")
    ax.set_title(f"Jt={ising.J * t_snap:.2f}\nC={ising.concurrence(psi_snap):.2f}")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
fig2.colorbar(im2, ax=axes2, fraction=0.025, label=r"$|\rho_1|$")
fig2.suptitle(r"Reduced single-qubit $|\rho_1|$: pure $\to$ maximally mixed $\to$ pure")

print(f"|rho_1| at Jt=0 (pure product state): off-diagonal = {np.abs(ising.reduced_density_matrix(ising.state(0.0)))[0, 1]:.4f}")
print(f"|rho_1| at Jt=pi/4 (max entangled): off-diagonal = {np.abs(ising.reduced_density_matrix(ising.state(t_max / 2)))[0, 1]:.4f}")
