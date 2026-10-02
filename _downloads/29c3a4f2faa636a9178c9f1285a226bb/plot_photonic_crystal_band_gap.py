r"""
Photonic band gaps in a 1D photonic crystal
===========================================

A periodic stack of two dielectrics is to light what a crystal lattice is
to electrons. Writing one period's characteristic matrix
:math:`M_{\text{period}}`, Bloch's theorem gives the modes of the infinite
stack,

.. math::

    \cos(K\Lambda) = \tfrac12\operatorname{tr}M_{\text{period}}
    = \cos\delta_1\cos\delta_2 - \tfrac12\left(\frac{n_1}{n_2} + \frac{n_2}{n_1}\right)\sin\delta_1\sin\delta_2,

and where the right side exceeds 1 in magnitude no real Bloch wavenumber
:math:`K` exists: light in that band of frequencies cannot propagate and
is reflected. Rayleigh found this for periodic layers in 1887; Yablonovitch
and John (1987) generalized it to 3D photonic crystals that forbid light
in every direction. For a quarter-wave stack the first gap is centred on
:math:`\omega_0` with width

.. math::

    \frac{\Delta\omega}{\omega_0} = \frac{4}{\pi}\arcsin\frac{n_H - n_L}{n_H + n_L}.

This example draws the band structure, measures the gap, and shows the
reflectance of finite stacks approaching that of the infinite crystal.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.optics.thin_films import bloch_wavenumber, multilayer_response, quarter_wave_band_gap, quarter_wave_stack

nH, nL, lam0 = 2.3, 1.38, 600.0
d = [lam0 / (4 * nH), lam0 / (4 * nL)]
Lambda = sum(d)
w = np.linspace(0.01, 3.2, 20000)  # omega / omega_0
KL = bloch_wavenumber([nH, nL], d, lam0 / w)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

# %%
# Band structure
# --------------
in_gap = KL.imag > 0
axes[0].plot(KL.real[~in_gap] / np.pi, w[~in_gap], ".", ms=1, color="navy")
axes[0].plot(-KL.real[~in_gap] / np.pi, w[~in_gap], ".", ms=1, color="navy")
n_avg = (nH * d[0] + nL * d[1]) / Lambda
K_light = np.linspace(-1, 1, 3)
axes[0].plot(K_light, np.abs(K_light) * np.pi / (2 * np.pi * n_avg * Lambda / lam0), "k:", lw=1, label="uniform medium, average index")
gap_edges = []
for start, stop in zip(np.nonzero(np.diff(in_gap.astype(int)) == 1)[0], np.nonzero(np.diff(in_gap.astype(int)) == -1)[0]):
    axes[0].axhspan(w[start + 1], w[stop], color="gold", alpha=0.4)
    gap_edges.append((w[start + 1], w[stop]))
axes[0].set_xlabel(r"Bloch wavenumber $K\Lambda/\pi$")
axes[0].set_ylabel(r"$\omega/\omega_0$")
axes[0].set_title("Photonic bands and gaps")
axes[0].set_xlim(-1, 1)
axes[0].legend(fontsize=7, loc="lower left")

# %%
# Decay inside the gap
# --------------------
axes[1].plot(w, KL.imag, color="crimson")
axes[1].set_xlabel(r"$\omega/\omega_0$")
axes[1].set_ylabel(r"Im$(K\Lambda)$: decay per period")
axes[1].set_title("Evanescent Bloch waves in the gaps")
first_gap = gap_edges[0]
width = first_gap[1] - first_gap[0]
print(f"first gap: {first_gap[0]:.4f} to {first_gap[1]:.4f}, width {width:.4f} (closed form {quarter_wave_band_gap(nH, nL):.4f})")

# %%
# Finite stacks: reflectance approaches the infinite crystal
# ----------------------------------------------------------
lam = lam0 / np.linspace(0.6, 1.4, 3000)
for pairs, color in [(2, "#9ecae1"), (5, "#4292c6"), (10, "#08306b")]:
    n, dd = quarter_wave_stack(nH, nL, pairs, lam0)
    R = multilayer_response(n, dd, lam, 1.0, 1.52)["R"]
    axes[2].plot(lam0 / lam, R, color=color, label=f"{pairs} pairs")
    if pairs == 10:
        R10 = R
axes[2].axvspan(*first_gap, color="gold", alpha=0.4, label="gap of the infinite crystal")
axes[2].set_xlabel(r"$\omega/\omega_0$")
axes[2].set_ylabel("reflectance")
axes[2].set_title("Bragg mirror")
axes[2].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %%
# Check
# -----
# The gap matches the closed form, is centred on omega_0, and a 10-pair
# stack reflects almost everything inside it.
assert abs(width - quarter_wave_band_gap(nH, nL)) < 1e-3
assert abs(0.5 * (first_gap[0] + first_gap[1]) - 1.0) < 1e-3
inside = (lam0 / lam > first_gap[0] + 0.05) & (lam0 / lam < first_gap[1] - 0.05)
assert R10[inside].min() > 0.99
