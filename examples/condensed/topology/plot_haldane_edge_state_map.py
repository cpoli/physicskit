r"""
The Haldane Model: a Chiral Edge State on an Arbitrary Boundary
========================================================================

:doc:`plot_haldane_phase_transition` exposes the Haldane model's chiral
edge states on a ribbon -- periodic in-plane, open along one
crystallographic direction. That is the easiest cut to compute, but it
leaves open whether the edge state depends on cutting *along* a lattice
direction the way graphene's zigzag zero modes do
(:doc:`plot_graphene_zigzag_flake`). It does not: a nonzero Chern number
guarantees a chiral state on *any* boundary of the sample, however it is
shaped. This example carves a circular disk out of the infinite lattice --
a boundary with no crystallographic meaning at all -- using
:func:`~physicskit.condensed.tight_binding.build_finite_cluster`'s ``keep``
predicate, and shows the in-gap states still hug that boundary in the
topological phase, and fail to in the trivial one.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.condensed.models import haldane_lattice_hamiltonian
from physicskit.condensed.tight_binding import Lattice, build_finite_cluster
from physicskit.condensed.visualizers import plot_lattice_structure

# %%
# Carving a disk out of the honeycomb lattice
# ----------------------------------------------
# ``build_finite_cluster``'s bounding box is a parallelogram of unit cells;
# the ``keep`` predicate then discards every site farther than ``radius``
# from the box's center, leaving a disk with an irregular, non-crystalline
# edge.

n_cells, radius = 14, 6.0
lat = Lattice.honeycomb()
center = np.array([n_cells / 2, n_cells / 2]) @ lat.lattice_vectors


def in_disk(cell, orbital, position):
    return np.linalg.norm(position - center) <= radius


# %%
# Topological vs. trivial: does an in-gap state hug this boundary?
# ----------------------------------------------------------------------
# The Haldane model is a Chern insulator for :math:`|M| < 3\sqrt3\,t_2|\sin\phi|`
# and trivial otherwise. In the topological phase, the handful of states
# nearest mid-gap should sit at a density-weighted mean radius close to the
# disk's edge; in the trivial phase, the states nearest mid-gap are just
# ordinary near-degenerate bulk states with no boundary preference.

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, (label, M) in zip(axes, [("topological (M=0)", 0.0), ("trivial (M=2.0)", 2.0)], strict=True):
    H_bulk = haldane_lattice_hamiltonian(t=1.0, t2=0.2, phi=np.pi / 2, M=M)
    H, positions, bonds = build_finite_cluster(H_bulk, n_cells=(n_cells, n_cells), keep=in_disk)
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    mid = len(eigenvalues) // 2
    in_gap = slice(mid - 3, mid + 3)
    density = np.sum(np.abs(eigenvectors[:, in_gap]) ** 2, axis=1)

    r = np.linalg.norm(positions - center, axis=1)
    mean_r_sites = r.mean()
    mean_r_weighted = np.sum(r * density) / density.sum()
    print(f"{label}: mean site radius = {mean_r_sites:.2f}, density-weighted radius of near-gap states = {mean_r_weighted:.2f}  (disk radius = {radius:.1f})")

    plot_lattice_structure(positions, bonds, weights=density, ax=ax)
    ax.set_title(f"{label}\nnear-gap density, E~[{eigenvalues[in_gap][0]:.2f}, {eigenvalues[in_gap][-1]:.2f}]")

fig.suptitle("Haldane model: a chiral edge state needs a Chern number, not a special edge")
fig.tight_layout()
