r"""
The Fabry-Pérot interferometer: multiple-beam interference and finesse
======================================================================

Two parallel mirrors of reflectance :math:`R` a distance :math:`L` apart
send light back and forth many times, and the transmitted beams add with
a round-trip phase :math:`\delta = 4\pi nL\cos\theta/\lambda`. Summing the
geometric series gives the Airy function (Fabry and Pérot, 1899)

.. math::

    T = \frac{1}{1 + F\sin^2(\delta/2)}, \qquad F = \frac{4R}{(1-R)^2}.

Every resonance transmits fully, however reflective the mirrors are, and
the peaks sharpen as :math:`R \to 1`: the ratio of their spacing (the free
spectral range) to their width is the finesse
:math:`\mathcal F = \pi\sqrt R/(1 - R)`. This made the interferometer the
first instrument to resolve the fine structure of spectral lines. The last
panel builds an all-dielectric cavity, a half-wave spacer between two
Bragg mirrors, layer by layer with the characteristic-matrix method, and
follows its resonance as the mirrors get more reflective.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.optics.thin_films import (
    airy_transmission,
    coefficient_of_finesse,
    finesse,
    multilayer_response,
    quarter_wave_stack,
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# %%
# The Airy function for increasing mirror reflectance
# ---------------------------------------------------
delta = np.linspace(-np.pi, 3 * np.pi, 4000)
for R, color in [(0.3, "#9ecae1"), (0.7, "#4292c6"), (0.95, "#08306b")]:
    axes[0].plot(delta / (2 * np.pi), airy_transmission(delta, R), color=color, label=f"R = {R}, finesse {finesse(R):.1f}")
axes[0].set_xlabel(r"round-trip phase $\delta / 2\pi$")
axes[0].set_ylabel("transmission")
axes[0].set_title("Airy function")
axes[0].legend(fontsize=8)

# %%
# Finesse: free spectral range over linewidth
# -------------------------------------------
R_values = np.linspace(0.2, 0.99, 30)
fine = np.linspace(-np.pi, np.pi, 400001)
measured = [2 * np.pi / np.ptp(fine[airy_transmission(fine, R) >= 0.5]) for R in R_values]
exact = np.pi / (2 * np.arcsin(1 / np.sqrt(coefficient_of_finesse(R_values))))
axes[1].semilogy(R_values, exact, "k-", label=r"$\pi / 2\arcsin(F^{-1/2})$")
axes[1].semilogy(R_values, finesse(R_values), "k--", lw=1, label=r"$\pi\sqrt{R}/(1 - R)$, high-R limit")
axes[1].semilogy(R_values, measured, "o", ms=4, color="crimson", label="FSR / FWHM of the Airy peak")
axes[1].set_xlabel("mirror reflectance R")
axes[1].set_ylabel("finesse")
axes[1].legend()
axes[1].set_title("Sharper fringes as R approaches 1")

# %%
# An all-dielectric Fabry-Pérot cavity
# ------------------------------------
# Quarter-wave pairs of TiO2/MgF2-like layers on each side of a half-wave
# spacer. Each extra pair multiplies the mirror transmission :math:`1 - R`,
# and so the cavity linewidth, by :math:`(n_L/n_H)^2` once :math:`R` is
# close to 1.
nH, nL, lam0 = 2.3, 1.38, 600.0
lam = np.linspace(590.0, 610.0, 200001)
fwhm = {}
for pairs, color in [(3, "#9ecae1"), (4, "#4292c6"), (5, "#08306b")]:
    n, d = quarter_wave_stack(nH, nL, pairs, lam0, cavity=True)
    T = multilayer_response(n, d, lam, 1.0, 1.0)["T"]
    fwhm[pairs] = np.ptp(lam[T >= 0.5])
    axes[2].plot(lam, T, color=color, label=f"{pairs} pairs per mirror: FWHM {fwhm[pairs]:.3f} nm")
    if pairs == 5:
        T_peak = T[np.argmin(np.abs(lam - lam0))]
axes[2].set_xlim(596, 604)
axes[2].set_xlabel("wavelength (nm)")
axes[2].set_ylabel("transmission")
axes[2].set_title("Dielectric cavity, half-wave spacer")
axes[2].legend(fontsize=7, loc="upper left")
plt.tight_layout()
plt.show()
ratios = [fwhm[3] / fwhm[4], fwhm[4] / fwhm[5]]
print(f"linewidth ratio per added pair: {ratios[0]:.3f}, {ratios[1]:.3f} (expected (nH/nL)^2 = {(nH / nL) ** 2:.3f})")

# %%
# Check
# -----
np.testing.assert_allclose(measured, exact, rtol=3e-3)
assert T_peak > 0.999
# (nH/nL)^2 is the large-R limit, approached as pairs are added
assert abs(ratios[1] / (nH / nL) ** 2 - 1) < 0.02 and ratios[1] < ratios[0]
