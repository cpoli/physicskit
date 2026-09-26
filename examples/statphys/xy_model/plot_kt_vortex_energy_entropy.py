r"""
Kosterlitz and Thouless: the vortex energy-entropy argument
===============================================================

Kosterlitz and Thouless (1973) located a phase transition in the 2D XY
model with a one-line free-energy estimate. A single vortex in a system
of size :math:`L` with lattice spacing :math:`a` costs

.. math::

    E_{\rm vortex} = \pi J \ln(L/a),

but it can sit on any of :math:`(L/a)^2` sites, so its entropy is
:math:`S = 2k_B\ln(L/a)`. Both grow as :math:`\ln L`, so the free energy

.. math::

    F = E - TS = (\pi J - 2k_BT)\ln(L/a)

changes sign at :math:`k_BT = \pi J/2`. Below that, free vortices are
suppressed and only bound pairs, whose energy does not grow with
:math:`L`, survive. This example measures :math:`E_{\rm vortex}` on
lattices of increasing size, shows that a vortex-antivortex pair's
energy depends on its separation but not on :math:`L`, and compares
the estimate with the Monte Carlo transition temperature
:attr:`~physicskit.statphys.chapters.ising_lattice.XYModel2D.T_KT`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.statphys.chapters.ising_lattice import XYModel2D

J = 1.0


def excess_energy(theta):
    """XY energy above the aligned state, open boundaries: J * sum(1 - cos(dtheta))."""
    return J * (np.sum(1.0 - np.cos(np.diff(theta, axis=0))) + np.sum(1.0 - np.cos(np.diff(theta, axis=1))))


def vortex_field(L, centres):
    """Superpose winding fields atan2 around each (x, y, charge), centred between sites."""
    y, x = np.mgrid[0:L, 0:L].astype(float)
    theta = np.zeros((L, L))
    for xc, yc, q in centres:
        theta += q * np.arctan2(y - yc, x - xc)
    return theta


# %%
# A single vortex costs :math:`\pi J\ln L`
# --------------------------------------------
sizes = np.array([8, 16, 32, 64, 128, 256])
E_single = np.array([excess_energy(vortex_field(L, [(L / 2 - 0.5, L / 2 - 0.5, 1)])) for L in sizes])
slope, intercept = np.polyfit(np.log(sizes[2:]), E_single[2:], 1)
print(f"E_vortex(L) = {slope:.3f} ln L + {intercept:.2f}   (theory: slope pi J = {np.pi * J:.3f})")

# %%
# A bound pair's energy is independent of :math:`L`
# -----------------------------------------------------
# A vortex and an antivortex a distance :math:`r` apart: far away, their
# windings cancel, so the energy depends only on :math:`r`, growing as
# :math:`2\pi J\ln r`. :meth:`~physicskit.statphys.chapters.ising_lattice.XYModel2D.vorticity`
# confirms the two plaquette charges.
separations = np.array([2, 4, 8, 16, 32])
print("\npair energy E(r) for two lattice sizes:")
E_pair = {}
for L in (128, 256):
    E_pair[L] = np.array([excess_energy(vortex_field(L, [(L / 2 - 0.5 - r / 2, L / 2 - 0.5, 1), (L / 2 - 0.5 + r / 2, L / 2 - 0.5, -1)])) for r in separations])
    print(f"  L = {L:3d}: " + ", ".join(f"r={r}: {e:.2f}" for r, e in zip(separations, E_pair[L])))
pair_slope = np.polyfit(np.log(separations[1:]), E_pair[256][1:], 1)[0]
print(f"pair slope dE/d ln r = {pair_slope:.3f}   (theory 2 pi J = {2 * np.pi * J:.3f})")

model = XYModel2D(L=32, J=J)
model.theta = vortex_field(32, [(11.5, 15.5, 1), (19.5, 15.5, -1)]) % (2 * np.pi)
charges = model.vorticity()
print(f"plaquette charges found by XYModel2D.vorticity(): {sorted(np.rint(charges[np.abs(charges) > 0.5]).astype(int).tolist())}")

# %%
# Free energy and the transition temperature
# ----------------------------------------------
# :math:`F(L) = E_{\rm vortex}(L) - 2k_BT\ln L`. Its slope in
# :math:`\ln L` flips sign at :math:`T = \pi J/2 \approx 1.57`. The true
# transition, found by Monte Carlo, is lower (:math:`\approx0.893\,J`):
# bound pairs screen the vortex interaction and soften the stiffness,
# which Kosterlitz's 1974 renormalization-group treatment accounts for.
T_estimate = np.pi * J / 2
print(f"\nenergy-entropy estimate k_B T = pi J / 2 = {T_estimate:.3f}")
print(f"Monte Carlo T_KT (XYModel2D.T_KT)       = {model.T_KT:.3f}")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
ax1.semilogx(sizes, E_single, "o", color="steelblue", label="measured")
ax1.semilogx(sizes, slope * np.log(sizes) + intercept, "--", color="orange", label=rf"${slope:.2f}\,\ln L$ + const")
ax1.set_xlabel("system size L")
ax1.set_ylabel(r"$E_{\rm vortex}$ [J]")
ax1.set_title("One free vortex: grows as ln L")
ax1.legend(fontsize=8)

for L, marker in ((128, "o"), (256, "s")):
    ax2.semilogx(separations, E_pair[L], marker, mfc="none", ms=8, label=f"L = {L}")
ax2.set_xlabel("pair separation r")
ax2.set_ylabel(r"$E_{\rm pair}$ [J]")
ax2.set_title("Bound pair: depends on r, not L")
ax2.legend(fontsize=8)

for T, color in zip([0.8, T_estimate, 2.2], ["steelblue", "0.3", "firebrick"]):
    ax3.semilogx(sizes, E_single - 2 * T * np.log(sizes), "o-", color=color, label=f"$k_BT$ = {T:.2f}")
ax3.set_xlabel("system size L")
ax3.set_ylabel("F = E - TS [J]")
ax3.set_title(r"Free vortices win above $\pi J/2$")
ax3.legend(fontsize=8)
fig.tight_layout()

plt.show()
