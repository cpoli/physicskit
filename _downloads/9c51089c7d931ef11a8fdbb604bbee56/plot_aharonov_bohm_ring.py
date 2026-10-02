r"""
The Aharonov-Bohm effect
============================

A charged particle confined to a 1D ring of radius :math:`R` threads a
magnetic flux :math:`\Phi` through its center; even though the field
:math:`B=0` everywhere the particle can actually be, the eigenspectrum

.. math::

    E_n(\Phi) = \frac{\hbar^2}{2mR^2}\left(n - \frac{\Phi}{\Phi_0}\right)^2,
    \qquad \Phi_0 = \frac{2\pi\hbar}{q},

still shifts periodically with :math:`\Phi/\Phi_0` -- the Aharonov-Bohm
effect, a purely topological/boundary-condition consequence of the vector
potential.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.quantum.chapters.entanglement import AharonovBohmRing

# %%
# The ring's flux-periodic energy spectrum
# --------------------------------------------------------------------------
# The ring's energy levels :math:`E_n(\Phi)` for angular-momentum quantum
# numbers :math:`n=-2,\dots,2`, plotted against the enclosed flux in units
# of the flux quantum :math:`\Phi_0` -- each parabola is centered on the
# :math:`n` that minimizes the shifted quantum number, giving the
# sawtooth-like periodic ground-state energy.

ring = AharonovBohmRing(R=1.0)
Phi_over_Phi0 = np.linspace(-2, 2, 400)

fig, ax2 = plt.subplots(figsize=(7, 4.5))
for n in range(-2, 3):
    E_n = [ring.energy(n, f * ring.flux_quantum) for f in Phi_over_Phi0]
    ax2.plot(Phi_over_Phi0, E_n, label=f"n={n}")
ax2.set_xlabel(r"$\Phi / \Phi_0$")
ax2.set_ylabel("E_n(Phi)")
ax2.set_title("Aharonov-Bohm ring: flux-periodic energy spectrum")
ax2.legend(fontsize=7, ncol=2)
fig.tight_layout()

print(f"ground-state energy at Phi=0: {ring.energy(0, 0.0):.6f}")
print(f"ground-state energy at Phi=Phi0/2: {min(ring.energy(n, 0.5 * ring.flux_quantum) for n in range(-2, 3)):.6f}")

# %%
# Check
# -----
# E_n(Phi) = (n - Phi/Phi0)^2 / 2 m R^2: the spectrum is periodic in Phi0,
# and at Phi0/2 the ground state is degenerate with energy 1/8.
assert ring.energy(0, 0.0) == 0
assert np.isclose(min(ring.energy(n, 0.5 * ring.flux_quantum) for n in range(-2, 3)), 0.125)
assert np.isclose(ring.energy(0, 0.5 * ring.flux_quantum), ring.energy(1, 0.5 * ring.flux_quantum))
for f in (0.2, 0.7):
    E_f = sorted(ring.energy(n, f * ring.flux_quantum) for n in range(-3, 4))
    E_f1 = sorted(ring.energy(n, (f + 1) * ring.flux_quantum) for n in range(-2, 5))
    np.testing.assert_allclose(E_f, E_f1, atol=1e-12)
