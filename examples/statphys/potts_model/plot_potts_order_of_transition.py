r"""
First- versus second-order transitions in the q-state Potts model
=====================================================================

The Potts model generalizes the Ising model (:math:`q=2`) to :math:`q`
discrete states per site: each site of a periodic :math:`L \times L`
lattice carries a state :math:`s_i \in \{0, \ldots, q-1\}`, and the
Hamiltonian rewards neighboring sites for matching,

.. math::

    H = -J \sum_{\langle i,j \rangle} \delta(s_i, s_j),

where :math:`\delta` is the Kronecker delta. On the square lattice, the
Baxter exact solution predicts that the transition, at the exact critical
temperature :math:`T_C = J / (k_B \ln(1+\sqrt{q}))`, stays second order
(continuous) for :math:`q \le 4` but becomes first order (discontinuous)
for :math:`q > 4` -- the order parameter jumps abruptly rather than
vanishing smoothly. This example compares :math:`q=3` (second order)
against :math:`q=8` (first order) by sweeping temperature through each
model's :math:`T_C`.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.statphys.chapters.ising_lattice import PottsModel2D

# %%
# Sweep both models across their respective critical points
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, q in zip(axes, [3, 8]):
    model = PottsModel2D(L=32, q=q, J=1.0, kB=1.0, seed=0)
    temperatures = np.linspace(model.T_C + 0.6, max(model.T_C - 0.6, 0.05), 20)
    result = model.run_temperature_sweep(temperatures, n_equil=150, n_measure=200)

    ax.plot(result["T"], result["m"], marker="o", ms=3)
    ax.axvline(model.T_C, color="k", linestyle="--", linewidth=1, alpha=0.6, label="$T_C$")
    ax.set_xlabel("Temperature")
    ax.set_title(f"q = {q} ({'2nd' if q <= 4 else '1st'} order)")
    ax.legend()

axes[0].set_ylabel("order parameter $m$")
plt.suptitle("Potts model order parameter: continuous vs. discontinuous onset")
plt.tight_layout()

# %%
# Energy histograms at T_C: the microscopic fingerprint of the order
# ------------------------------------------------------------------------
# A first-order transition means the ordered and disordered phases coexist
# at T_C with a nonzero latent heat between them. On a lattice this small a
# Metropolis run at T_C rarely crosses from one phase to the other, so start
# two runs at T_C, one from a disordered and one from a fully ordered
# configuration: for q = 8 each stays in its own phase, and the two energy
# histograms sit apart by the latent heat (Baxter: 0.486 J per site). A
# continuous transition has no latent heat and no coexistence, so for
# q = 3 both starts settle into one and the same peak.
mean_energy = {}
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, q in zip(axes, [3, 8]):
    for start, color in (("disordered", "steelblue"), ("ordered", "firebrick")):
        model = PottsModel2D(L=32, q=q, J=1.0, kB=1.0, seed=1)
        if start == "ordered":
            model.spins[:] = 0
        beta_c = 1.0 / model.T_C
        model.sweep(beta_c, n_sweeps=300)  # equilibrate at T_C
        energies = np.empty(400)
        for i in range(400):
            model.sweep(beta_c, n_sweeps=5)
            energies[i] = model.energy() / model.n_sites
        mean_energy[q, start] = energies.mean()
        ax.hist(energies, bins=30, range=(-1.9, -0.8), color=color, alpha=0.6, label=f"{start} start")
    ax.set_xlabel("energy per site")
    ax.set_title(f"q = {q} ({'2nd order' if q <= 4 else '1st order'}) at $T_C$")
    ax.legend(fontsize=8)
axes[0].set_ylabel("count")
plt.suptitle("Energy distribution at $T_C$: coexistence signature of a first-order transition")
plt.tight_layout()
plt.show()

# %%
# Check
# -----
# T_C = J / ln(1 + sqrt(q)). For q = 8 the two phases sit at Baxter's
# coexistence energies e = -(1 + 1/sqrt(q)) -+ L/2 (up to finite-size shifts
# of the ordered phase), with
# L = 2 (1 + 1/sqrt(q)) tanh(theta/2) prod_n tanh^2(n theta), cosh theta = sqrt(q)/2;
# for q = 3 the gap between the two starts is several times smaller.
for q in (3, 8):
    assert abs(PottsModel2D(L=8, q=q).T_C - 1 / np.log(1 + np.sqrt(q))) < 1e-12
theta = np.arccosh(np.sqrt(8) / 2)
latent = 2 * (1 + 1 / np.sqrt(8)) * np.tanh(theta / 2) * np.prod(np.tanh(np.arange(1, 50) * theta) ** 2)
assert abs(latent - 0.486) < 1e-3
assert abs(mean_energy[8, "disordered"] - (-(1 + 1 / np.sqrt(8)) + latent / 2)) < 0.05
assert abs(mean_energy[8, "ordered"] - (-(1 + 1 / np.sqrt(8)) - latent / 2)) < 0.1
gap = {q: mean_energy[q, "disordered"] - mean_energy[q, "ordered"] for q in (3, 8)}
assert gap[8] > 0.4 and gap[8] > 4 * gap[3]
