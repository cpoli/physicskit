r"""
Hahn, Strassmann, Meitner, and Frisch: nuclear fission
=========================================================

Late in 1938 Hahn and Strassmann found barium among the products of
uranium bombarded with neutrons, a nucleus about half uranium's size.
Meitner and Frisch explained it with the liquid-drop model. A uranium
nucleus is so highly charged that its Coulomb repulsion nearly cancels
the surface tension holding it round. A small kick from an absorbed
neutron can stretch it until it splits in two, and the two fragments fly
apart with about 200 MeV.

This example computes that energy three ways: from measured masses with
:func:`~physicskit.particle.nuclear.q_value`, from the liquid-drop
binding energies of
:func:`~physicskit.particle.nuclear.semf_binding_energy` over every way
of splitting the nucleus, and from Meitner and Frisch's own back-of-envelope
Coulomb estimate. It then shows why only the heaviest nuclei are close
to splitting.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.constants import AVOGADRO, ELECTRONVOLT
from physicskit.particle.nuclear import q_value, semf_binding_energy

U = 931.494

# %%
# Energy release from masses
# ------------------------------
# One observed channel: :math:`n+{}^{235}\mathrm U\to{}^{141}\mathrm{Ba}+
# {}^{92}\mathrm{Kr}+3n`.
m_n, m_U235, m_Ba141, m_Kr92 = 1.008665, 235.043930, 140.914411, 91.926156
Q_masses = q_value([m_U235, m_n], [m_Ba141, m_Kr92] + [m_n] * 3) * U
print(f"Q from measured masses (Ba-141 + Kr-92 + 3n): {Q_masses:.1f} MeV")

# %%
# Every possible split, from the liquid drop
# ----------------------------------------------
# The compound nucleus :sup:`236`\ U splits into fragments
# :math:`(Z_1,A_1)` and :math:`(92-Z_1, 236-A_1)`, each at the charge
# that keeps the fragments' charge-to-mass ratio equal to uranium's. The energy
# release is the gain in total binding energy. It is large and positive
# for any roughly even split.
B_parent = semf_binding_energy(92, 236)
A1 = np.arange(40, 197)
Q_split = []
for a in A1:
    z = int(round(92 * a / 236))
    Q_split.append(semf_binding_energy(z, a) + semf_binding_energy(92 - z, 236 - a) - B_parent)
Q_split = np.array(Q_split)
print(f"liquid-drop Q, symmetric split (A = 118 + 118): {Q_split[A1 == 118][0]:.0f} MeV")
print(f"liquid-drop Q, Ba/Kr-like split (A = 141 + 95): {Q_split[A1 == 141][0]:.0f} MeV")

# %%
# Meitner and Frisch's Coulomb estimate
# -----------------------------------------
# Two fragments of charge :math:`46e` just touching, radii
# :math:`1.2A^{1/3}` fm, repel with :math:`Z_1Z_2e^2/(4\pi\varepsilon_0 d)`.
# That electrostatic energy becomes the fragments' kinetic energy.
R_frag = 1.2 * 118 ** (1 / 3)
E_coulomb = 1.44 * 46 * 46 / (2 * R_frag)
print(f"Coulomb energy of two touching Pd-118 fragments: {E_coulomb:.0f} MeV  (order 200 MeV)")

energy_per_gram = AVOGADRO / 235.0 * Q_masses * 1e6 * ELECTRONVOLT
print(f"\nfissioning 1 g of U-235 releases {energy_per_gram:.1e} J, about {energy_per_gram / 4.2e9:.0f} tonnes of TNT")
print(f"per atom, about {Q_masses * 1e6 / 5:.0e} times a chemical bond energy (~5 eV)")

# %%
# The fissility parameter
# ---------------------------
# Stretching a liquid drop slightly raises its surface energy by
# :math:`\tfrac25\epsilon^2E_S` and lowers its Coulomb energy by
# :math:`\tfrac15\epsilon^2E_C`. The drop has no barrier at all once
# :math:`E_C > 2E_S`, i.e. when the fissility
# :math:`x=E_C/(2E_S)=a_CZ^2/(2a_SA)` reaches 1, at
# :math:`Z^2/A\approx50`. Uranium sits at :math:`x\approx0.7`: close enough
# for a captured neutron's energy to push it over.
a_C, a_S = 0.711, 17.8
nuclei = {"Fe-56": (26, 56), "Sn-120": (50, 120), "Pb-208": (82, 208), "U-236": (92, 236), "Pu-240": (94, 240), "Fm-256": (100, 256)}
x = {name: a_C * Z**2 / (2 * a_S * A) for name, (Z, A) in nuclei.items()}
for name, xv in x.items():
    print(f"  {name:7s} fissility x = {xv:.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
ax1.plot(A1, Q_split, color="firebrick")
ax1.axhline(Q_masses, color="steelblue", ls="--", label=f"measured Ba-141 + Kr-92 channel ({Q_masses:.0f} MeV)")
ax1.set_xlabel("mass number of one fragment $A_1$")
ax1.set_ylabel("energy released Q [MeV]")
ax1.set_title(r"Splitting $^{236}$U: liquid-drop Q for every split")
ax1.legend(fontsize=8)
ax2.bar(list(x), list(x.values()), color=["0.6", "0.6", "0.6", "firebrick", "firebrick", "firebrick"])
ax2.axhline(1.0, color="k", ls="--", label="x = 1: no barrier left")
ax2.set_ylabel(r"fissility $x = E_C / 2E_S$")
ax2.set_title("Only the heaviest drops are close to splitting")
ax2.legend(fontsize=8)
fig.tight_layout()

plt.show()
