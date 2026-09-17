r"""
The Hubbard Model: Mott Suppression of Itinerant Motion
==============================================================

The Hubbard model is the minimal model of interacting electrons on a
lattice: band electrons that hop with amplitude :math:`t`, penalized by an
onsite Coulomb repulsion :math:`U` whenever two opposite-spin electrons
share a site. As :math:`U/t \to \infty` at half filling, double occupancy
is suppressed entirely and the itinerant metal is driven into a Mott
insulator of localized moments.
:func:`~physicskit.condensed.correlated.hubbard_1d_exact_diagonalization`
builds and diagonalizes the full Fock-space Hamiltonian in a fixed
:math:`(n_\uparrow, n_\downarrow)` sector for small clusters, exposing this
crossover directly in the ground-state energy.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.condensed.correlated import hubbard_1d_exact_diagonalization

# %%
# Exact diagonalization of a half-filled chain across U/t
# -------------------------------------------------------------
# A 6-site chain at half filling (3 up, 3 down electrons), open boundary
# conditions, scanning the onsite repulsion from the noninteracting limit
# to the strongly correlated regime.

n_sites = 6
U_values = np.linspace(0, 12, 25)
E0 = [hubbard_1d_exact_diagonalization(n_sites, n_up=3, n_dn=3, t=1.0, U=U)["ground_state_energy"] for U in U_values]

# %%
# The itinerant-to-localized crossover
# ------------------------------------------
# At ``U=0`` the ground state is a free-fermion Fermi sea, gaining a large
# negative kinetic energy from delocalized hopping. As ``U/t`` grows, the
# energy curve flattens: double occupancy is squeezed out and the
# electrons freeze into singly occupied, localized moments -- the
# many-body signature of the Mott transition.

# %%
# The charge gap: the direct diagnostic of a Mott insulator
# ---------------------------------------------------------------------
# A flattening ground-state energy is suggestive, but the sharp,
# unambiguous signature of a Mott insulator is a nonzero *charge gap*:
# the energy cost to add or remove one electron from half filling,
# :math:`\Delta_c = E_0(N+1) + E_0(N-1) - 2E_0(N)`, evaluated by exact
# diagonalization in the neighboring particle-number sectors. In the
# thermodynamic limit a metal has :math:`\Delta_c=0`; a Mott insulator has
# :math:`\Delta_c>0` growing with :math:`U`. This 6-site chain is far from
# that limit, so :math:`\Delta_c(U=0)` is not actually zero -- it is the
# ordinary finite-size level spacing of a free-fermion "particle in a box,"
# which only vanishes as the chain grows. What is still a genuine,
# interaction-driven effect at this size is how much *further* the gap
# opens as :math:`U` turns on, well beyond that finite-size baseline --
# exactly the charge-transfer gap seen in real correlated
# insulators.

E_plus = [hubbard_1d_exact_diagonalization(n_sites, n_up=4, n_dn=3, t=1.0, U=U)["ground_state_energy"] for U in U_values]
E_minus = [hubbard_1d_exact_diagonalization(n_sites, n_up=2, n_dn=3, t=1.0, U=U)["ground_state_energy"] for U in U_values]
charge_gap = np.array(E_plus) + np.array(E_minus) - 2 * np.array(E0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(U_values, E0, "o-")
axes[0].set_xlabel("U/t")
axes[0].set_ylabel(r"Ground-state energy $E_0$")
axes[0].set_title("Mott suppression: itinerant metal -> localized moments")

axes[1].plot(U_values, charge_gap, "o-", color="C1")
axes[1].axhline(charge_gap[0], color="gray", ls="--", lw=1, label=f"U=0 finite-size baseline ({charge_gap[0]:.2f})")
axes[1].set_xlabel("U/t")
axes[1].set_ylabel(r"charge gap $\Delta_c = E_0(N{+}1)+E_0(N{-}1)-2E_0(N)$")
axes[1].set_title("Charge gap grows well past its finite-size baseline")
axes[1].legend(fontsize=8)

fig.suptitle("The Hubbard model at half filling: energy flattens, charge gap opens")
fig.tight_layout()

slope_start = (E0[1] - E0[0]) / (U_values[1] - U_values[0])
slope_end = (E0[-1] - E0[-2]) / (U_values[-1] - U_values[-2])
print(f"dE0/dU near U=0: {slope_start:.4f}   dE0/dU near U=12: {slope_end:.4f} (flattening as U grows)")
ratio = charge_gap[-1] / charge_gap[0]
print(f"charge gap at U=0: {charge_gap[0]:.4f} (finite-size baseline, not a Mott gap)   at U=12: {charge_gap[-1]:.4f} ({ratio:.1f}x above baseline)")
