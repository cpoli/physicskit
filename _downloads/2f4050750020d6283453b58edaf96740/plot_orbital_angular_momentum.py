r"""
Allen et al.: orbital angular momentum of a Laguerre-Gaussian vortex
=========================================================================

Les Allen and coworkers showed that Laguerre-Gaussian light modes carrying
a helical (spiral) phase front :math:`e^{il\phi}` possess a well-defined
*orbital* angular momentum of :math:`l\hbar` per photon -- an intrinsic,
mechanical property of light entirely separate from the spin angular
momentum carried by polarization. The discovery opened the field of
structured light and optical vortices.
:func:`~physicskit.optics.gaussian.laguerre_gaussian_mode` constructs the
:math:`\mathrm{LG}_p^l` mode amplitude

.. math::

    u_{lp}(r, \phi, z) = \frac{w_0}{w(z)}
        \left(\frac{r\sqrt2}{w(z)}\right)^{|l|}
        L_p^{|l|}\!\left(\frac{2r^2}{w(z)^2}\right)
        \exp\!\left[-\frac{r^2}{w(z)^2}\right]
        \exp\!\left[-\frac{ikr^2}{2R(z)}\right]
        \exp(il\phi)
        \exp\!\left[i(|l|+2p+1)\zeta(z)\right] \exp(-ikz)

directly, with :math:`w(z)` the beam radius, :math:`R(z)` the wavefront
radius of curvature, :math:`\zeta(z)` the Gouy phase, :math:`L_p^{|l|}`
the associated Laguerre polynomial, azimuthal index :math:`l`, and radial
index :math:`p`. The radial factor :math:`(r\sqrt2/w(z))^{|l|}` vanishes
on-axis for any :math:`l\ne0`, producing the doughnut-shaped, on-axis-dark
intensity profile, while the :math:`e^{il\phi}` term winds the phase
through :math:`2\pi l` once around the axis -- the helical wavefront that
carries the orbital angular momentum. This is compared below against the
ordinary (:math:`l=0,\ p=0`) Gaussian mode's single bright spot and flat
on-axis phase.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.optics.gaussian import GaussianBeam, laguerre_gaussian_mode

# %%
# The l=0 fundamental mode vs. an l=1 optical vortex, same beam parameters
# -------------------------------------------------------------------------

beam = GaussianBeam(wavelength=0.5e-3, w0=0.5, z0=0.0)
coords = np.linspace(-2, 2, 300)
X, Y = np.meshgrid(coords, coords)
R, Phi = np.hypot(X, Y), np.arctan2(Y, X)

u_gaussian = laguerre_gaussian_mode(R, Phi, 0.0, beam, l=0, p=0)
u_vortex = laguerre_gaussian_mode(R, Phi, 0.0, beam, l=1, p=0)

# %%
# The vortex's doughnut intensity (dark core on-axis, since the LG
# amplitude carries a factor r^|l| that vanishes at r=0) and its helical
# phase (a 2*pi phase winding around the axis, absent for l=0).

fig, axes = plt.subplots(2, 2, figsize=(7, 7))
axes[0, 0].imshow(np.abs(u_gaussian) ** 2, extent=[-2, 2, -2, 2])
axes[0, 0].set_title("l=0 intensity")
axes[0, 1].imshow(np.angle(u_gaussian), extent=[-2, 2, -2, 2], cmap="twilight")
axes[0, 1].set_title("l=0 phase")
axes[1, 0].imshow(np.abs(u_vortex) ** 2, extent=[-2, 2, -2, 2])
axes[1, 0].set_title("l=1 intensity (donut, dark core)")
axes[1, 1].imshow(np.angle(u_vortex), extent=[-2, 2, -2, 2], cmap="twilight")
axes[1, 1].set_title("l=1 phase (helical, 2*pi winding)")
fig.tight_layout()

center = len(coords) // 2
I_gaussian_axis = np.abs(u_gaussian[center, center]) ** 2
I_vortex_axis = np.abs(u_vortex[center, center]) ** 2

# Phase sampled around a small circle of radius r around the axis: for l=1
# this should wind through 2*pi once; for l=0 it should stay flat.
theta_ring = np.linspace(0, 2 * np.pi, 400, endpoint=False)
r_ring = 0.5
u_ring_vortex = laguerre_gaussian_mode(r_ring, theta_ring, 0.0, beam, l=1, p=0)
phase_unwrapped = np.unwrap(np.angle(u_ring_vortex))
winding = (phase_unwrapped[-1] - phase_unwrapped[0]) / (2 * np.pi)

print(f"on-axis intensity, l=0 (single bright spot): {I_gaussian_axis:.6f}")
print(f"on-axis intensity, l=1 (dark core):           {I_vortex_axis:.2e}")
print(f"phase winding around the l=1 vortex axis: {winding:.3f} x 2*pi (predicted: 1.0)")
print("orbital angular momentum per photon: L_z = l*hbar -> nonzero only for l != 0")
