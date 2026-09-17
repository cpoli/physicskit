r"""
Oort and Lindblad's differential rotation: the Oort constants
==================================================================

Lindblad argued the Milky Way could not rotate rigidly; Oort (1927)
supplied the observational test: if the Galaxy rotates differentially,
nearby stars' radial velocities and proper motions vary with Galactic
longitude :math:`l` in a double-sine pattern set by just two numbers,

.. math::

    A = -\frac{1}{2}R_0\left(\frac{d\Omega}{dR}\right)_{R_0}, \qquad
    B = A - \Omega(R_0),

fixed by the local value and slope of the rotation curve
:math:`\Omega(R)=v_c(R)/R`. This example builds a rotation curve from
:func:`~physicskit.astro.galactic_dynamics.circular_velocity` and
:func:`~physicskit.astro.galactic_dynamics.nfw_enclosed_mass`, computes
:math:`A` and :math:`B` numerically at the Sun's Galactocentric radius,
and reproduces the double-sine streaming pattern in radial velocity and
proper motion that Oort actually measured.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.astro.galactic_dynamics import circular_velocity, nfw_enclosed_mass

# %%
# A rotation curve, and the Sun's place on it
# ---------------------------------------------------
# An NFW-halo-dominated rotation curve, in :math:`G=1` units, standing
# in for the Milky Way's; :math:`R_0` marks the Sun's Galactocentric
# radius.
rho_s, r_s = 1.0, 8.0
R0 = 8.0


def v_c(R):
    return circular_velocity(R, lambda r: nfw_enclosed_mass(r, rho_s, r_s))


R_values = np.linspace(0.5, 30.0, 400)
v_values = v_c(R_values)

# %%
# Oort's constants from the local rotation curve
# ----------------------------------------------------
# :math:`\Omega(R)=v_c(R)/R`; its value and slope at :math:`R_0` fix
# :math:`A` and :math:`B` directly, via a simple centered finite
# difference on a fine local grid.
h = 1e-4
Omega_R0 = v_c(R0) / R0
dOmega_dR = (v_c(R0 + h) / (R0 + h) - v_c(R0 - h) / (R0 - h)) / (2 * h)

A = -0.5 * R0 * dOmega_dR
B = A - Omega_R0
print(f"Omega(R0)   = {Omega_R0:.6f}")
print(f"dOmega/dR   = {dOmega_dR:.6f}")
print(f"Oort A = {A:.6f}")
print(f"Oort B = {B:.6f}")
print(f"(A - B = Omega(R0) exactly, by construction: {A - B:.6f} vs {Omega_R0:.6f})")

fig1, ax1 = plt.subplots(figsize=(6, 4.5))
ax1.plot(R_values, v_values, color="steelblue")
ax1.axvline(R0, color="firebrick", ls="--", label=f"$R_0$={R0}")
ax1.set_xlabel("R")
ax1.set_ylabel(r"$v_c(R)$")
ax1.set_title("Rotation curve and the Sun's Galactocentric radius")
ax1.legend()
fig1.tight_layout()

# %%
# The double-sine streaming pattern
# ---------------------------------------
# To leading order in :math:`(R-R_0)/R_0` for stars near the Sun, the
# line-of-sight (radial) velocity and the proper motion vary with
# Galactic longitude :math:`l` as
# :math:`v_r(l)=Ad\sin(2l)` and :math:`\mu(l)=A\cos(2l)+B`, for a star
# at fixed distance :math:`d` -- exactly the pattern Oort found in
# existing stellar radial velocities, confirming differential rotation.
l_values = np.linspace(0.0, 2.0 * np.pi, 400)
d = 1.0  # a fixed, small distance from the Sun (local approximation)
v_r = A * d * np.sin(2 * l_values)
mu = A * np.cos(2 * l_values) + B

fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax2.plot(np.degrees(l_values), v_r, color="darkorange")
ax2.set_xlabel("Galactic longitude $l$ (degrees)")
ax2.set_ylabel(r"$v_r(l) = A\,d\sin(2l)$")
ax2.set_title("Radial-velocity streaming")

ax3.plot(np.degrees(l_values), mu, color="seagreen")
ax3.set_xlabel("Galactic longitude $l$ (degrees)")
ax3.set_ylabel(r"$\mu(l) = A\cos(2l) + B$")
ax3.set_title("Proper-motion streaming")
fig2.tight_layout()

plt.show()
