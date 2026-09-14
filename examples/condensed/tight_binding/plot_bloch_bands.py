r"""
Bloch's Theorem: From an Infinite Chain to H(k)
====================================================

Felix Bloch's theorem turns the Schrodinger equation for an electron in an
infinite periodic crystal into a finite eigenvalue problem :math:`H(k)
u_{nk} = E_n(k) u_{nk}` at each crystal momentum :math:`k`, periodic on the
Brillouin zone. :class:`~physicskit.condensed.tight_binding.Hamiltonian`
builds :math:`H(k)` by Bloch-summing real-space hoppings, and
:meth:`~physicskit.condensed.tight_binding.Hamiltonian.bands` diagonalizes
it -- reproducing the monatomic chain's exact dispersion
:math:`E(k) = -2t\cos k` with no continuum limit or approximation involved.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.condensed.tight_binding import Hamiltonian, Lattice, build_finite_cluster
from physicskit.condensed.visualizers import plot_lattice_structure

# %%
# A monatomic chain with a single real-space hopping
# ------------------------------------------------------
# One orbital per unit cell, nearest-neighbor hopping ``t=1`` -- the
# simplest possible crystal.

lat = Lattice.chain(a=1.0)
H = Hamiltonian(lat, onsite=[0.0])
H.add_hopping(0, 0, (1,), -1.0)

# %%
# Bloch-summing and diagonalizing at every k in the Brillouin zone
# ---------------------------------------------------------------------
# :meth:`Hamiltonian.bloch` assembles :math:`H(k) = -t(e^{ik} + e^{-ik}) =
# -2t\cos k` from the single real-space hopping added above;
# :meth:`Hamiltonian.bands` diagonalizes it. The result matches the
# textbook dispersion to machine precision -- there is no approximation
# between the real-space model and the Bloch-summed band.

k_grid = np.linspace(0, 2 * np.pi, 200)
bands = np.array([H.bands([k]) for k in k_grid])

# %%
# The real-space structure Bloch's theorem is summing over
# ---------------------------------------------------------------------
# :math:`H(k)` above is nothing but this same finite chain's single
# repeated hopping, re-summed with a phase :math:`e^{ikR}` per unit cell
# translation :math:`R`. :func:`build_finite_cluster` truncates that
# translational symmetry to a finite, open segment so the real-space
# structure Bloch's theorem abstracts away can be drawn directly.

H_finite, positions, bonds = build_finite_cluster(H, n_cells=10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(k_grid, bands[:, 0], lw=2.5, label="H.bands(k)")
ax1.plot(k_grid, -2 * np.cos(k_grid), "k--", lw=1.5, label=r"$E(k)=-2t\cos k$")
ax1.set_xlabel("k")
ax1.set_ylabel("Energy")
ax1.set_title("Bloch band (reciprocal space)")
ax1.legend()

plot_lattice_structure(positions, bonds, ax=ax2)
ax2.set_ylim(-0.5, 0.5)
ax2.set_title(f"The same chain, open and finite ({H_finite.shape[0]} sites)")

fig.suptitle("Tight-binding chain: Bloch sum reproduces the exact dispersion")
fig.tight_layout()

max_error = np.max(np.abs(bands[:, 0] - (-2 * np.cos(k_grid))))
print(f"max deviation from exact dispersion: {max_error:.2e}")
