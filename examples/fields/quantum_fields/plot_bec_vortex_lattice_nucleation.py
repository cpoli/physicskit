r"""
The critical rotation frequency for BEC vortex nucleation
================================================================

Following the first dilute-gas Bose-Einstein condensates in 1995 (Cornell,
Wieman, Ketterle), experimentalists set condensates rotating and watched
them nucleate quantized vortices above a critical rotation frequency,
arranging themselves into a triangular Abrikosov-like lattice -- exactly
as Gross-Pitaevskii theory, by then over three decades old, had predicted.
Both the vortex-free and single-vortex states compared here are relaxed
(at zero rotation) via imaginary-time propagation of the Gross-Pitaevskii
equation for a harmonically trapped condensate,

.. math::

    i\partial_t\psi = \Big[-\tfrac12\nabla^2 + V(\mathbf{r}) + g|\psi|^2\Big]\psi,
    \qquad V(\mathbf{r}) = \tfrac12(x^2+y^2) \quad (\hbar=m=1),

by :func:`~physicskit.fields.quantum_fields.gpe_relax`; the vortex state
starts from :func:`~physicskit.fields.quantum_fields.gpe_imprint_vortex`
multiplying in a single :math:`2\pi` phase winding before relaxation.
Comparing the two relaxed states' lab-frame energy :math:`E` and angular
momentum :math:`L_z` then gives the standard *energetic* nucleation
criterion,

.. math::

    \Omega_c = \frac{\Delta E}{\Delta L_z},

the rotation frequency above which a vortex genuinely lowers the
rotating-frame energy :math:`E - \Omega L_z`, reproduced here from first
principles without ever simulating the rotating trap itself, and
:func:`~physicskit.fields.quantum_fields.count_vortices` confirms the
imprinted state carries exactly one quantum of circulation -- the
single-vortex "unit cell" of the triangular lattices seen experimentally
above that threshold (worked through further in
:doc:`/tutorials/bec_vortex_lattice_creation`).
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.fields import count_vortices, gpe_energy, gpe_imprint_vortex, gpe_relax, harmonic_trap_grid, plot_bec_density, plot_bec_phase

# %%
# Relax the vortex-free ground state and a state seeded with one vortex
# --------------------------------------------------------------------------

n, length, g = 64, 12.0, 4.0
X, Y, KX, KY, K2 = harmonic_trap_grid(n, length)
V = 0.5 * (X**2 + Y**2)
psi_seed = np.exp(-0.5 * (X**2 + Y**2)).astype(complex)
psi_vortex_free = gpe_relax(psi_seed, V, g, dtau=5e-4, steps=1500, X=X, Y=Y, K2=K2)
psi_vortex = gpe_relax(gpe_imprint_vortex(psi_seed, X, Y, [(0.0, 0.0)]), V, g, dtau=5e-4, steps=1500, X=X, Y=Y, K2=K2)

# %%
# Compare their lab-frame energy and angular momentum
# -----------------------------------------------------------

E0 = gpe_energy(psi_vortex_free, V, g, X, Y, K2)
E1 = gpe_energy(psi_vortex, V, g, X, Y, K2)

# %%
# Above :math:`\Omega_c = \Delta E/\Delta L_z`, nucleating a vortex lowers
# the rotating-frame energy -- the textbook criterion, reproduced here
# from first principles, and the single-vortex "unit cell" of the
# triangular lattices seen experimentally above that threshold.

Omega_c = (E1["total"] - E0["total"]) / (E1["angular_momentum"] - E0["angular_momentum"])
winding = count_vortices(psi_vortex)

fig, ax = plot_bec_density(X, Y, psi_vortex)
ax.set_title(f"relaxed single-vortex state, Omega_c = {Omega_c:.3f}")
fig.tight_layout()

print(f"critical rotation frequency Omega_c = {Omega_c:.4f}")
print(f"quantized circulation in the relaxed vortex state: {np.sum(np.abs(winding))} (expected: 1)")

# %%
# The vortex-free ground state and the vortex's phase singularity
# ---------------------------------------------------------------------
# The single density panel above only shows the relaxed vortex state; the
# vortex-free ground state relaxed alongside it (used to compute
# :math:`\Omega_c`) is a smooth, hole-free cloud by contrast, and the
# vortex state's phase winds by exactly :math:`2\pi` around the density
# zero visible above -- the two complementary views of the same single
# quantized vortex, side by side.

fig2, (ax_density0, ax_phase) = plt.subplots(1, 2, figsize=(10, 4))
plot_bec_density(X, Y, psi_vortex_free, ax=ax_density0)
ax_density0.set_title("vortex-free ground state (no density hole)")
plot_bec_phase(X, Y, psi_vortex, ax=ax_phase)
ax_phase.set_title(f"vortex state phase, winding = {np.sum(np.abs(winding))}")
fig2.tight_layout()
