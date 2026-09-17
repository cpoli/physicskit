r"""
Weyl Semimetals: Momentum-Space Monopoles and Fermi Arcs
================================================================

Wan, Turner, Vishwanath, and Savrasov predicted that breaking inversion or
time-reversal symmetry in a 3D Dirac material splits each doubly-degenerate
Dirac point into a pair of nondegenerate Weyl nodes -- momentum-space
sources and sinks of Berry curvature carrying opposite chirality, the
direct 3D generalization of the 2D TKNN invariant. A surface hosts open,
non-closed "Fermi arcs" of surface states connecting the surface
projections of opposite-chirality bulk nodes, later confirmed by ARPES on
TaAs.
:func:`~physicskit.condensed.weyl.weyl_semimetal_hamiltonian` implements
the minimal two-band cubic-lattice model with this physics,

.. math::

    H(\mathbf{k}) = \sin k_x\,\sigma_x + \sin k_y\,\sigma_y +
    \left(m - t\sum_i \cos k_i\right)\sigma_z ,

whose mass term vanishes at exactly two points on the :math:`k_z` axis for
:math:`1 < m/t < 3` -- a pair of Weyl nodes of opposite chirality.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.condensed.weyl import weyl_node_locations, weyl_semimetal_hamiltonian, weyl_semimetal_slab_hamiltonian

m, t = 2.0, 1.0
k0 = weyl_node_locations(m, t)[1]
print(f"Weyl nodes at kz = +-{k0:.4f} (m/t = {m / t})")

# %%
# The bulk gap closes at exactly two points
# -----------------------------------------------
# Scanning the bulk gap along the kx = ky = 0 axis shows it dip to exactly
# zero at the two predicted node positions, and nowhere else.

kz_grid = np.linspace(-np.pi, np.pi, 400)
gap = [np.diff(np.linalg.eigvalsh(weyl_semimetal_hamiltonian(0.0, 0.0, kz, m, t)))[0] for kz in kz_grid]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
axes[0].plot(kz_grid, gap, color="steelblue")
axes[0].axvline(-k0, color="firebrick", lw=0.8, ls="--")
axes[0].axvline(k0, color="firebrick", lw=0.8, ls="--")
axes[0].set_xlabel(r"$k_z$  ($k_x=k_y=0$)")
axes[0].set_ylabel("bulk gap")
axes[0].set_title("Bulk gap closes only at the two Weyl nodes")

# %%
# A single chiral surface mode connects the two nodes
# -----------------------------------------------------------
# Opening the lattice along x exposes a surface Brillouin zone in
# (ky, kz). Between the two nodes' projections the slab hosts a single
# chiral mode crossing zero energy as a function of ky -- the lattice
# signature of a Fermi arc; outside the node range the slab is fully
# gapped, exactly like a trivial 2D insulator's edge.

ky_grid = np.linspace(0, 2 * np.pi, 300)
n_layers = 40
bands_inside = np.array([np.linalg.eigvalsh(weyl_semimetal_slab_hamiltonian(ky, 0.0, n_layers, m, t)) for ky in ky_grid])
bands_outside = np.array([np.linalg.eigvalsh(weyl_semimetal_slab_hamiltonian(ky, np.pi, n_layers, m, t)) for ky in ky_grid])

n_show = 6
mid = n_layers
for n in range(mid - n_show // 2, mid + n_show // 2):
    axes[1].plot(ky_grid, bands_inside[:, n], color="steelblue", lw=0.8)
    axes[1].plot(ky_grid, bands_outside[:, n], color="lightgray", lw=0.8)
axes[1].axhline(0, color="black", lw=0.5)
axes[1].plot([], [], color="steelblue", label=r"$k_z=0$ (between nodes): gapless")
axes[1].plot([], [], color="lightgray", label=r"$k_z=\pi$ (outside nodes): gapped")
axes[1].set_xlabel(r"$k_y$")
axes[1].set_ylabel("E")
axes[1].set_title("Slab spectrum near E=0")
axes[1].legend(fontsize=8)
fig.tight_layout()

# %%
# Tracing the ky where the slab spectrum crosses zero energy, as a function
# of kz swept across the whole node range, would trace out the Fermi arc
# itself; here the two representative cuts already make the qualitative
# point -- a single mode crossing zero between the nodes, a full gap
# outside them, exactly the surface signature ARPES resolved in TaAs.

plt.show()
