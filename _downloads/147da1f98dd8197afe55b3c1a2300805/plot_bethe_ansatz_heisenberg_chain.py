r"""
Bethe's ansatz for the Heisenberg antiferromagnetic chain
=========================================================

Hans Bethe solved the spin-1/2 Heisenberg ring
:math:`H = J\sum_i \mathbf{S}_i\cdot\mathbf{S}_{i+1}` exactly in 1931 by
writing its eigenstates as superpositions of interacting magnons. Here the
ground-state energy from
:func:`~physicskit.condensed.spin_chains.bethe_ansatz_xxx_ground_energy`
is compared with Lanczos exact diagonalization of
:func:`~physicskit.condensed.spin_chains.xxz_hamiltonian` in the
:math:`S^z = 0` sector, and extrapolated to Hulthen's thermodynamic limit
:math:`E_0/N = J(1/4 - \ln 2)`.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.condensed.spin_chains import bethe_ansatz_xxx_ground_energy, lowest_eigenstates, xxz_hamiltonian

# %%
# Exact diagonalization vs. the Bethe equations
# ----------------------------------------------
#
# Restricting to the :math:`S^z = 0` sector cuts the Hilbert space from
# :math:`2^N` to :math:`\binom{N}{N/2}` states.

N_ed = np.arange(4, 19, 2)
E_ed, gap_ed = [], []
for N in N_ed:
    E0, _ = lowest_eigenstates(xxz_hamiltonian(N, n_up=N // 2)[0], k=1)
    E1, _ = lowest_eigenstates(xxz_hamiltonian(N, n_up=N // 2 + 1)[0], k=1)  # lowest S=1 state
    E_ed.append(E0[0])
    gap_ed.append(E1[0] - E0[0])
E_ed, gap_ed = np.array(E_ed), np.array(gap_ed)
E_bethe = np.array([bethe_ansatz_xxx_ground_energy(N) for N in N_ed])

for N, a, b in zip(N_ed, E_ed, E_bethe):
    print(f"N = {N:2d}: ED E0 = {a:.10f}   Bethe E0 = {b:.10f}   diff = {abs(a - b):.1e}")

# %%
# Approach to the thermodynamic limit
# ------------------------------------
#
# The Bethe equations stay cheap long after exact diagonalization becomes
# impossible. Conformal field theory predicts
# :math:`E_0/N = e_\infty - \pi^2 J / (12 N^2)` for this :math:`c = 1`
# critical chain.

N_bethe = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256])
e_bethe = np.array([bethe_ansatz_xxx_ground_energy(N) for N in N_bethe]) / N_bethe
e_inf = 0.25 - np.log(2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.plot(1 / N_bethe**2, e_bethe, "o-", label="Bethe ansatz")
ax.plot(1 / N_ed**2, E_ed / N_ed, "x", ms=10, mew=2, label="exact diagonalization")
x = np.linspace(0, 1 / 16, 50)
ax.plot(x, e_inf - np.pi**2 / 12 * x, "k--", lw=1, label=r"$e_\infty - \pi^2/(12N^2)$")
ax.axhline(e_inf, color="0.5", ls=":")
ax.set_xlabel(r"$1/N^2$")
ax.set_ylabel(r"$E_0 / (NJ)$")
ax.set_title(r"Ground-state energy per site $\to 1/4 - \ln 2$")
ax.legend()

# %%
# A gapless spectrum
# ------------------
#
# Unlike the integer-spin chain, the spin-1/2 Heisenberg chain has no gap:
# the singlet-triplet splitting closes as :math:`1/N`.

ax = axes[1]
ax.plot(1 / N_ed, gap_ed, "o-")
ax.set_xlabel(r"$1/N$")
ax.set_ylabel(r"$E_{S=1} - E_0$ ($J$)")
ax.set_title("Singlet-triplet gap closes linearly in 1/N")
ax.set_xlim(0, None)
ax.set_ylim(0, None)
fig.tight_layout()
