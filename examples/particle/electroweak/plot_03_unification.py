r"""
Glashow, Weinberg, and Salam: electroweak unification
============================================================

Glashow (1961), Weinberg (1967), and Salam (1968) showed that the
electromagnetic and weak interactions are two faces of a single
:math:`SU(2)\times U(1)` gauge theory, spontaneously broken by the Higgs
mechanism into the massless photon (electromagnetism) and the massive
:math:`W^\pm,Z^0` bosons (the weak force) -- unifying two of nature's four
fundamental interactions. Gargamelle's 1973 discovery of weak neutral
currents, mediated by the previously unobserved :math:`Z^0`, was the
theory's first direct experimental confirmation. This example puts the
theory's two already-modeled pieces side by side:
:func:`~physicskit.particle.electroweak.qed_dsigma_domega_mumu`, the
electromagnetic sector reproduced at low energy, and
:func:`~physicskit.particle.electroweak.higgs_vev`, the symmetry-breaking
scale that gives the :math:`W`/:math:`Z` their mass while leaving the
photon massless -- exactly the two ingredients the unification joins.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.particle.electroweak import higgs_potential, higgs_vev, qed_total_cross_section_mumu

# %%
# Piece 1: the electromagnetic sector, reproduced at low energy
# --------------------------------------------------------------------
# Below the Z pole (sqrt(s) << M_Z ~ 91 GeV), the electroweak theory's
# prediction for e+e- -> mu+mu- reduces to pure photon exchange -- the
# same QED cross section already checked on its own terms in the
# renormalized-QED example.
sqrt_s_values = np.linspace(3.0, 80.0, 200)
sigma_qed = np.array([qed_total_cross_section_mumu(s) for s in sqrt_s_values])
M_Z = 91.19  # GeV, for reference -- not itself predicted by the pieces modeled here

fig1, ax1 = plt.subplots(figsize=(6.5, 4.5))
ax1.loglog(sqrt_s_values, sigma_qed, color="steelblue")
ax1.axvline(M_Z, color="firebrick", ls="--", label=f"$M_Z$={M_Z} GeV (where the full electroweak theory departs from pure QED)")
ax1.set_xlabel(r"$\sqrt{s}$ (GeV)")
ax1.set_ylabel(r"$\sigma$ (photon-exchange piece)")
ax1.set_title("The electromagnetic sector this theory reproduces at low energy")
ax1.legend(fontsize=8)
fig1.tight_layout()

# %%
# Piece 2: the symmetry-breaking scale setting the W/Z masses
# -------------------------------------------------------------------
# The same spontaneous-symmetry-breaking mechanism modeled in the Higgs
# example gives the W and Z bosons their mass (schematically,
# M_W, M_Z ~ g*v for the theory's gauge couplings g) while leaving the
# photon exactly massless -- a direct structural consequence of *which*
# combination of the SU(2)xU(1) generators remains unbroken.
a, b = 1.0, 1.0
v = higgs_vev(a, b)
print(f"symmetry-breaking scale v = {v:.6f} (schematic units)")
print("the W and Z acquire mass proportional to this scale (times the theory's gauge couplings);")
print("the photon corresponds to the one gauge direction left exactly unbroken -- massless by construction.")

phi_range = np.linspace(-1.3 * v, 1.3 * v, 400)
fig2, ax2 = plt.subplots(figsize=(6, 4.5))
ax2.plot(phi_range, higgs_potential(phi_range, a, b), color="gray")
ax2.plot([v, -v], [higgs_potential(v, a, b), higgs_potential(-v, a, b)], "o", color="firebrick", ms=8)
ax2.set_xlabel(r"$\phi$")
ax2.set_ylabel(r"$V(\phi)$")
ax2.set_title("The symmetry-breaking scale that sets M_W, M_Z (schematic)")
fig2.tight_layout()

# %%
# Gargamelle (1973): the first evidence for the Z boson this theory predicts
# ---------------------------------------------------------------------------------
# Gargamelle's bubble-chamber neutrino events showing a struck electron
# with no accompanying charged lepton -- a neutral current, mediated by
# exactly the Z0 boson this unification requires -- were the theory's
# first direct experimental confirmation, a decade before the W and Z
# were produced and detected directly (see the 1983 entry in this
# chronology). Nothing in the toy pieces modeled here computes a
# neutral-current cross section directly; the qualitative point is that
# both the QED piece and the symmetry-breaking piece above are
# ingredients of the single theory Gargamelle's neutral currents
# confirmed.
print("\nGargamelle (1973): observed neutrino-electron neutral-current scattering, mediated by the Z0")
print("this unification predicts -- confirming the theory a decade before the W/Z were produced directly.")

plt.show()
