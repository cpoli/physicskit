r"""
The WKB approximation: connection formulas on the quantum bouncer
=====================================================================

Wentzel, Kramers and Brillouin (1926) wrote the wavefunction in a slowly
varying potential as an amplitude times a fast phase,

.. math::

    \psi(x) \approx \frac{C}{\sqrt{p(x)}}\,\sin\!\left(\frac1\hbar\int_0^x p\,dx' + \phi_0\right),
    \qquad p(x)=\sqrt{2m(E-V(x))}.

The approximation fails where :math:`p\to0`, at the classical turning
points. Kramers's contribution was to match it through each turning
point to the exact local solution, an Airy function. This fixes the
phase lost at a soft turning point at :math:`\pi/4`, against
:math:`\pi/2` at a hard wall. Those phases are the difference between
the old rule :math:`\oint p\,dq = nh` and the correct spectrum.

The quantum bouncer (a particle above a hard floor in uniform gravity,
:math:`V=mgx`) has one of each kind of turning point, and exact
solutions given by Airy functions. This example compares the exact
levels with WKB using the Kramers phases, built from
:func:`~physicskit.semiclassical.core.wkb.classical_momentum` and
:func:`~physicskit.semiclassical.core.wkb.wkb_action`, and with the
uncorrected integer rule. Units: :math:`\hbar = m = g = 1`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ai_zeros, airy

from physicskit.semiclassical.core.wkb import classical_momentum, wkb_action

V = lambda x: x  # noqa: E731  (floor at x = 0)

# %%
# Exact levels
# ----------------
# :math:`-\tfrac12\psi'' + x\psi = E\psi` with :math:`\psi(0)=0` is solved
# by :math:`\mathrm{Ai}(2^{1/3}(x-E))`, so the energies are the Airy
# zeros :math:`a_n` rescaled: :math:`E_n = -a_n/2^{1/3}`.
n_levels = 10
E_exact = -ai_zeros(n_levels)[0] / 2 ** (1 / 3)

# %%
# WKB levels
# --------------
# Between the floor and the turning point :math:`x=E`, the action is
# :math:`\int_0^E p\,dx = \tfrac{2\sqrt2}{3}E^{3/2}`. With a hard wall
# (:math:`\pi/2`) and a soft turning point (:math:`\pi/4`), the
# quantization condition is :math:`\int_0^E p\,dx = (n + \tfrac34)\pi`.
# The uncorrected rule, ignoring both phases, would be :math:`(n+1)\pi`.
n = np.arange(n_levels)
E_wkb = (3 * np.pi * (n + 0.75) / (2 * np.sqrt(2))) ** (2 / 3)
E_naive = (3 * np.pi * (n + 1.0) / (2 * np.sqrt(2))) ** (2 / 3)

check = wkb_action(E_wkb[3], V, 1.0, 0.0, E_wkb[3]) / np.pi
print(f"numerical action / pi at the n=3 WKB level: {check:.4f}  (n + 3/4 = 3.75)")
print(f"{'n':>2s} {'exact':>8s} {'WKB n+3/4':>10s} {'rel. err':>9s} {'naive n+1':>10s} {'rel. err':>9s}")
for k in range(n_levels):
    print(f"{k:2d} {E_exact[k]:8.4f} {E_wkb[k]:10.4f} {abs(E_wkb[k] / E_exact[k] - 1):9.1e} {E_naive[k]:10.4f} {abs(E_naive[k] / E_exact[k] - 1):9.1e}")

# %%
# The wavefunction and the turning point
# ------------------------------------------
# In the allowed region the WKB wave matches the exact Airy function almost
# perfectly, but its :math:`1/\sqrt p` amplitude blows up at
# :math:`x = E`. Beyond it, the WKB form is the decaying exponential
# :math:`e^{-\int\kappa\,dx}/\sqrt\kappa`, again good away from the
# turning point. The Airy function joins the two smoothly; matching both
# WKB forms to it is what fixes the :math:`\pi/4`.
level = 4
E = E_wkb[level]
x = np.linspace(1e-4, E + 3.0, 2000)
psi_exact = airy(2 ** (1 / 3) * (x - E_exact[level]))[0]
inside, outside = x < E, x > E
p = classical_momentum(E, V, x[inside], m=1.0)
phase = np.array([wkb_action(E, V, 1.0, 0.0, xi) for xi in x[inside]])
psi_in = np.sin(phase) / np.sqrt(p)
kappa = np.sqrt(2 * (x[outside] - E))
decay = (2 * np.sqrt(2) / 3) * (x[outside] - E) ** 1.5
psi_out = 0.5 * np.exp(-decay) / np.sqrt(kappa) * (-1) ** level

scale = np.max(np.abs(psi_exact[x < 0.6 * E])) / np.max(np.abs(psi_in[x[inside] < 0.6 * E]))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(x, psi_exact, color="k", lw=2.5, alpha=0.4, label="exact (Airy)")
ax1.plot(x[inside], scale * psi_in, color="steelblue", label=r"WKB, allowed region")
ax1.plot(x[outside], scale * psi_out, color="firebrick", label=r"WKB, forbidden region")
ax1.axvline(E, color="0.5", ls=":", label="turning point x = E")
ax1.set_ylim(-1.2, 1.2)
ax1.set_xlabel("height x")
ax1.set_ylabel(r"$\psi(x)$")
ax1.set_title(f"Level n = {level}: WKB fails only at the turning point")
ax1.legend(fontsize=8)
ax2.semilogy(n, np.abs(E_wkb / E_exact - 1), "o-", color="steelblue", label=r"WKB with Kramers phases, $n+3/4$")
ax2.semilogy(n, np.abs(E_naive / E_exact - 1), "s-", color="firebrick", label=r"uncorrected, $n+1$")
ax2.set_xlabel("level n")
ax2.set_ylabel("relative energy error")
ax2.set_title("The connection phases fix the spectrum")
ax2.legend(fontsize=8)
fig.tight_layout()

plt.show()
