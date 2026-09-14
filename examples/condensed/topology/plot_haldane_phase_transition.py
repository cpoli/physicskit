r"""
The Haldane Model: A Chern Insulator with Zero Net Flux
==============================================================

Duncan Haldane showed that the integer quantum Hall effect does not
require a net magnetic field at all. Threading complex second-neighbor
hopping :math:`t_2 e^{i\phi}` through a honeycomb lattice, in a pattern
with zero *net* flux per unit cell but nonzero local (loop) curvature,
produces a Chern insulator with :math:`C=\pm1` -- the first quantum
anomalous Hall model, and the direct template for the
:math:`\mathbb{Z}_2` topological insulators discovered two decades later.
:func:`~physicskit.condensed.models.haldane_model` gives the Bloch
Hamiltonian used to track the bulk Chern number across the phase
transition, and :func:`~physicskit.condensed.models.haldane_lattice_hamiltonian`
builds the same model in real space to expose the chiral edge states that
bulk-boundary correspondence predicts.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.condensed.models import haldane_lattice_hamiltonian, haldane_model
from physicskit.condensed.tight_binding import build_ribbon
from physicskit.condensed.topology import compute_chern_number

# %%
# Crossing the topological phase transition
# ------------------------------------------------
# Scanning the sublattice (Semenoff) mass :math:`M` across the critical
# value :math:`M_c = 3\sqrt{3}\,t_2|\sin\phi|` drives the lower band's
# Chern number from :math:`1` to :math:`0` exactly where the bulk gap
# closes -- a topological invariant can only change value at a gap-closing
# transition.

t2, phi = 0.2, np.pi / 2
M_c = 3 * np.sqrt(3) * t2 * abs(np.sin(phi))
M_values = np.linspace(0, 2 * M_c, 15)
chern = [compute_chern_number(lambda k1, k2: haldane_model(k1, k2, t2=t2, phi=phi, M=M), grid_size=24)[0] for M in M_values]

# %%
# Bulk-boundary correspondence: chiral edge states in a ribbon
# --------------------------------------------------------------------
# A nonzero Chern number in the bulk (``M=0``) forces gapless states to
# cross the bulk gap on any boundary. Cutting the honeycomb lattice into a
# finite-width, open ribbon with :func:`build_ribbon` and plotting its
# spectrum vs. the remaining periodic momentum shows exactly this: bands
# that thread the bulk gap rather than staying confined to the bulk bands.

H_bulk = haldane_lattice_hamiltonian(t=1.0, t2=t2, phi=phi, M=0.0)
H_ribbon = build_ribbon(H_bulk, open_direction=1, n_cells=20)
k_parallel = np.linspace(0, 2 * np.pi, 200)
ribbon_bands = np.array([np.linalg.eigvalsh(H_ribbon([k])) for k in k_parallel])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(M_values / M_c, chern, "o-")
ax1.axvline(1.0, color="gray", ls="--", label=r"$M_c=3\sqrt{3}\,t_2|\sin\phi|$")
ax1.set_xlabel(r"$M / M_c$")
ax1.set_ylabel("Chern number (lower band)")
ax1.set_title("Topological phase transition")
ax1.legend()

ax2.plot(k_parallel, ribbon_bands, color="C0", lw=0.8)
ax2.set_xlabel(r"$k_\parallel$")
ax2.set_ylabel("Energy")
ax2.set_title("Ribbon spectrum: chiral edge states cross the bulk gap")

fig.suptitle("Haldane model: Chern insulator from zero net flux")
fig.tight_layout()

min_over_k = np.min(np.abs(ribbon_bands), axis=1)
print(f"bulk gap (away from edge crossings): ~{np.median(min_over_k[min_over_k > 0.5]):.3f}")
print(f"minimum |E| reached by an edge-crossing state: {min_over_k.min():.4f}")
