r"""
Second-harmonic generation and phase matching
=============================================

In 1961, a year after the first laser, Franken and co-workers focused a
ruby laser into quartz and recorded a faint spot of ultraviolet light at
twice its frequency: the first nonlinear optical effect. A second-order
polarization :math:`P^{(2\omega)} \propto \chi^{(2)}E^2` radiates the
harmonic, but the fundamental and the harmonic travel at different speeds,
so the harmonic generated at different depths gets out of step. With
phase mismatch :math:`\Delta k = k_{2\omega} - 2k_\omega`, an undepleted
pump gives

.. math::

    P_{2\omega}(L) \propto L^2\,\mathrm{sinc}^2(\Delta k L/2),

which only oscillates, with period twice the coherence length
:math:`L_c = \pi/\Delta k`. Giordmaine and Maker et al. (1962) removed the
mismatch with the birefringence of the crystal, choosing the angle at
which :math:`n_e^{2\omega}(\theta) = n_o^{\omega}`. Armstrong, Bloembergen
et al. (1962) proposed instead flipping the sign of :math:`\chi^{(2)}` every
coherence length (quasi-phase matching). This example integrates the
coupled-amplitude equations for each case.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.optics.nonlinear import (
    extraordinary_index,
    shg_coupled_amplitudes,
    shg_phase_matched_efficiency,
    shg_undepleted_power,
    type_i_phase_matching_angle,
)

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

# %%
# The phase-matching curve
# ------------------------
L = 1.0
dkL = np.linspace(-4 * np.pi, 4 * np.pi, 1001)
axes[0].plot(dkL / np.pi, shg_undepleted_power(L, dkL / L) / shg_undepleted_power(L, 0.0), color="navy")
axes[0].set_xlabel(r"$\Delta k L / \pi$")
axes[0].set_ylabel("relative SH power")
axes[0].set_title(r"$\mathrm{sinc}^2(\Delta k L/2)$")

# %%
# Growth along the crystal: mismatched, quasi-phase-matched, phase-matched
# --------------------------------------------------------------------------
kappa = 2e-3  # weak conversion: the undepleted regime
dk = 2 * np.pi  # coherence length L_c = 0.5
z = np.linspace(0.0, 4.0, 801)
mismatched = shg_coupled_amplitudes(z, dk, kappa)
qpm = shg_coupled_amplitudes(z, dk, kappa, qpm_period=2 * np.pi / dk)
matched = shg_coupled_amplitudes(z, 0.0, kappa)
scale = kappa**2
axes[1].plot(z, np.abs(matched["A2"]) ** 2 / scale, color="crimson", label=r"phase matched, $\propto z^2$")
axes[1].plot(z, np.abs(qpm["A2"]) ** 2 / scale, color="darkorange", label=r"quasi-phase matched, $(2/\pi)^2$ slower")
axes[1].plot(z, np.abs(mismatched["A2"]) ** 2 / scale, color="navy", label=r"mismatched, period $2L_c$")
axes[1].plot(z, (2 / np.pi) ** 2 * z**2, "k:", lw=1)
axes[1].set_xlabel("z")
axes[1].set_ylabel(r"$|A_2|^2/\kappa^2$")
axes[1].set_title(r"$\Delta k = 2\pi$, $L_c = 0.5$")
axes[1].legend(fontsize=7)

# %%
# Pump depletion at perfect phase matching
# ----------------------------------------
zd = np.linspace(0.0, 3.0, 61)
strong = shg_coupled_amplitudes(zd, 0.0, kappa=1.0)
axes[2].plot(zd, np.abs(strong["A1"]) ** 2, color="firebrick", label=r"fundamental $|A_1|^2$")
axes[2].plot(zd, np.abs(strong["A2"]) ** 2, "o", ms=3, color="green", label=r"harmonic $|A_2|^2$")
axes[2].plot(zd, shg_phase_matched_efficiency(zd), "k-", lw=1, label=r"$\tanh^2(\kappa A_0 z)$")
axes[2].set_xlabel(r"$\kappa A_0 z$")
axes[2].set_title("Full conversion with depletion")
axes[2].legend(fontsize=8)

# %%
# Birefringent phase matching in KDP at 1064 nm
# ---------------------------------------------
n_o_w, n_e_w, n_o_2w, n_e_2w = 1.4942, 1.4603, 1.5129, 1.4709
theta = np.radians(np.linspace(0, 90, 901))
theta_m = type_i_phase_matching_angle(n_o_w, n_o_2w, n_e_2w)
axes[3].plot(np.degrees(theta), np.full_like(theta, n_o_w), color="firebrick", label=r"$n_o(\omega)$")
axes[3].plot(np.degrees(theta), extraordinary_index(theta, n_o_2w, n_e_2w), color="green", label=r"$n_e(2\omega, \theta)$")
axes[3].axvline(np.degrees(theta_m), color="gray", ls="--", label=rf"$\theta_m = {np.degrees(theta_m):.1f}^\circ$")
axes[3].set_xlabel(r"angle to optic axis $\theta$ (deg)")
axes[3].set_ylabel("refractive index")
axes[3].set_title("Type-I phase matching, KDP")
axes[3].legend(fontsize=8)
plt.tight_layout()
plt.show()
print(f"KDP type-I phase-matching angle: {np.degrees(theta_m):.2f} deg")

# %%
# Check
# -----
np.testing.assert_allclose(np.abs(mismatched["A2"]) ** 2, shg_undepleted_power(z, dk, kappa), rtol=1e-3, atol=1e-12)
assert abs(np.abs(qpm["A2"][-1]) / np.abs(matched["A2"][-1]) - 2 / np.pi) < 2e-3
np.testing.assert_allclose(np.abs(strong["A2"]) ** 2, shg_phase_matched_efficiency(zd), atol=1e-7)
assert abs(np.degrees(theta_m) - 41.2) < 0.3
