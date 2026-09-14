r"""
BCS Theory: The Superconducting Quasiparticle Gap
========================================================

Bardeen, Cooper, and Schrieffer explained superconductivity as a
condensate of Cooper pairs bound by an effective attraction, however weak.
Pairing opens a gap :math:`\Delta` in the single-particle excitation
spectrum: unlike the normal-state band :math:`|\xi(k)|`, which touches zero
at the Fermi points, the BCS quasiparticle band :math:`E(k) =
\sqrt{\xi(k)^2 + |\Delta(k)|^2}` never does.
:func:`~physicskit.condensed.correlated.bdg_bcs_hamiltonian` builds the
Bogoliubov-de Gennes Hamiltonian in Nambu (particle-hole) space, and
:func:`~physicskit.condensed.correlated.bdg_spectrum` diagonalizes it
across the Brillouin zone.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.condensed.correlated import bdg_bcs_hamiltonian, bdg_spectrum

# %%
# The BdG Hamiltonian at a single momentum
# ---------------------------------------------
# :func:`bdg_bcs_hamiltonian` builds :math:`H(k) = \xi(k)\tau_z +
# \mathrm{Re}(\Delta)\tau_x - \mathrm{Im}(\Delta)\tau_y` in Nambu space;
# :func:`bdg_spectrum` sweeps it over the whole Brillouin zone. At the
# Fermi momentum (:math:`\xi(k)=0`), the two Nambu branches sit exactly at
# :math:`\pm|\Delta|`.

H_kF = bdg_bcs_hamiltonian(k=np.pi / 2, mu=0.0, t=1.0, delta=0.5)
print(f"quasiparticle energies at the Fermi momentum: {np.linalg.eigvalsh(H_kF)}")

# %%
# Normal state vs. the paired condensate
# ---------------------------------------------
# Setting the pairing amplitude ``delta=0`` recovers the ordinary
# normal-state band :math:`|\xi(k)|`, which is gapless at the Fermi
# points. Turning on ``delta=0.5`` opens a full gap.

k, E_normal = bdg_spectrum(mu=0.0, t=1.0, delta=0.0, n_k=400)
_, E_super = bdg_spectrum(mu=0.0, t=1.0, delta=0.5, n_k=400)

# %%
# The gap never closes
# ---------------------------
# The minimum of the BCS quasiparticle band equals exactly :math:`|\Delta|`,
# regardless of momentum -- the defining signature of a fully gapped
# superconductor.

# %%
# Watching the gap open continuously as pairing turns on
# ---------------------------------------------------------------------
# The two curves above are just two slices of a whole family: sweeping
# :math:`|\Delta|` from zero and re-running :func:`bdg_spectrum` at every
# value traces out the full :math:`E(k, |\Delta|)` surface. At
# :math:`\Delta=0` the normal-state V-shaped touching at the Fermi
# momentum is exact; turning on pairing lifts that V into a hyperbola with
# minimum exactly :math:`|\Delta|`, for every :math:`\Delta` at once.

delta_values = np.linspace(0.0, 1.0, 60)
E_surface = np.array([bdg_spectrum(mu=0.0, t=1.0, delta=d, n_k=400)[1] for d in delta_values])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(k, E_normal, label=r"normal state $|\xi(k)|$")
ax1.plot(k, E_super, label="BCS quasiparticle $E(k)$")
ax1.axhline(0.5, color="gray", ls="--", label=r"gap $|\Delta|=0.5$")
ax1.set_xlabel("k")
ax1.set_ylabel("Energy")
ax1.set_title(r"Two slices: $\Delta=0$ vs $\Delta=0.5$")
ax1.legend(fontsize=8)

pm = ax2.pcolormesh(k, delta_values, E_surface, cmap="magma", shading="auto")
ax2.set_xlabel("k")
ax2.set_ylabel(r"$|\Delta|$")
ax2.set_title(r"$E(k,|\Delta|)$: the gap opening continuously")
fig.colorbar(pm, ax=ax2, label="quasiparticle energy")

fig.suptitle("BCS pairing opens a gap in the excitation spectrum")
fig.tight_layout()

print(f"normal-state minimum energy: {E_normal.min():.4f} (gapless)")
print(f"BCS quasiparticle minimum energy: {E_super.min():.4f} (== |Delta| = 0.5)")
print(f"min(E) vs Delta matches |Delta| to: {np.max(np.abs(E_surface.min(axis=1) - delta_values)):.2e}")
