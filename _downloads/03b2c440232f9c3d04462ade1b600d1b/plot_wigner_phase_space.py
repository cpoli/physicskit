r"""
Wigner phase-space quasi-probability
========================================

The Wigner function represents a quantum state as a quasi-probability
distribution over classical phase space :math:`(x,p)`,

.. math::

    W(x,p) = \frac{1}{\pi\hbar} \int_{-\infty}^{\infty}
        \psi^*(x+y)\,\psi(x-y)\, e^{2ipy/\hbar}\, dy,

built here from the harmonic-oscillator Fock eigenstates
:math:`\psi=\phi_n(x)`. Unlike a genuine probability density, :math:`W`
can go negative; its :math:`x`- and :math:`p`-marginals still reproduce the
ordinary quantum probabilities, :math:`\int W\,dp = \lvert\psi(x)\rvert^2`.
The ground state (:math:`n=0`) stays a positive Gaussian, but the first
excited state (:math:`n=1`) develops genuinely negative regions -- a
hallmark of non-classicality with no counterpart in any classical
phase-space distribution.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.quantum.chapters.harmonic_spin import HarmonicOscillator
from physicskit.quantum.visualizers.phase_space import WignerVisualizer

ho = HarmonicOscillator()
x = np.linspace(-6, 6, 220)
wv = WignerVisualizer(n_p=180)

# %%
# The n=0 and n=1 Fock-state Wigner functions
# ------------------------------------------------

fig = plt.figure(figsize=(12, 5))

for i, n in enumerate([0, 1]):
    psi_n = ho.eigenfunction(n, x)
    xg, p, W = wv.compute(x, psi_n, p_max=6.0)

    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    wv.plot_surface(xg, p, W, ax=ax)
    ax.set_title(f"Fock state n={n}  (min W = {W.min():.3f})")

fig.tight_layout()

# %%
# Sanity check: integrating W over p should recover :math:`\lvert\psi(x)\rvert^2`.

psi1 = ho.eigenfunction(1, x)
xg, p, W1 = wv.compute(x, psi1)
marginal = wv.position_marginal(p, W1)
print("max |marginal - |psi|^2| =", np.max(np.abs(marginal - psi1**2)))

# %%
# Phase-space interference fringes in a Fock-state superposition
# --------------------------------------------------------------------
#
# A single Fock state's Wigner function is rotationally symmetric in phase
# space, but a coherent *superposition* of two Fock states develops genuine
# interference fringes between them -- oscillatory, sign-alternating ripples
# with no classical counterpart -- built here with
# :meth:`~physicskit.quantum.chapters.harmonic_spin.HarmonicOscillator.superposition_wavefunction`
# and rendered with the (previously unused)
# :meth:`~physicskit.quantum.visualizers.phase_space.WignerVisualizer.plot_contour`.

psi_super = ho.superposition_wavefunction([0, 2], [1.0, 1.0], x, t=0.0)
xg_s, p_s, W_super = wv.compute(x, psi_super, p_max=6.0)

fig2, ax_super = plt.subplots(figsize=(6.5, 5.5))
wv.plot_contour(xg_s, p_s, W_super, ax=ax_super)
ax_super.set_title(r"Wigner function of $(|0\rangle+|2\rangle)/\sqrt{2}$" "\n(interference fringes between the two lobes)")
fig2.tight_layout()

print(f"min W of the superposition (non-classicality): {W_super.min():.4f}")
