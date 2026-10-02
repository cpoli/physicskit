r"""
Grey radiative transfer: the Eddington approximation and limb darkening
=======================================================================

The Sun's disc is darker at the edge than at the centre. Looking toward
the limb, the line of sight reaches optical depth one higher up, in cooler
gas. Schwarzschild (1906) and Eddington (1926) explained this with the
transfer equation for a plane-parallel atmosphere,

.. math::

    \mu\frac{dI}{d\tau} = I - S,

in radiative equilibrium with a grey (frequency-independent) opacity, so
that :math:`S = J` and the flux :math:`F = \sigma T_{\rm eff}^4` is constant.
The temperature is then :math:`T^4 = \tfrac34 T_{\rm eff}^4[\tau + q(\tau)]`.
Eddington's closure :math:`K = J/3` gives :math:`q = 2/3`, so the surface
sits at :math:`T = 2^{-1/4}T_{\rm eff}` and :math:`T = T_{\rm eff}` at
:math:`\tau = 2/3`, with limb darkening

.. math::

    \frac{I(0, \mu)}{I(0, 1)} = \frac{2 + 3\mu}{5}.

The exact solution (Hopf 1930, Chandrasekhar 1944) has :math:`q` rising
from :math:`1/\sqrt3` to :math:`0.7104`. This example computes it by
Chandrasekhar's discrete-ordinates method and compares both with the
approximate linear limb darkening of the visible Sun.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.astro.radiative_transfer import GreyAtmosphere, eddington_limb_darkening, eddington_temperature

atm = GreyAtmosphere(n_streams=32)
tau = np.geomspace(1e-3, 10.0, 400)
mu = np.linspace(0.0, 1.0, 201)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# %%
# The Hopf function
# -----------------
axes[0].semilogx(tau, atm.hopf(tau), color="crimson", label="exact grey (Hopf)")
axes[0].axhline(2 / 3, color="k", ls="--", label="Eddington, 2/3")
axes[0].axhline(1 / np.sqrt(3), color="gray", ls=":", lw=1)
axes[0].axhline(atm.Q, color="gray", ls=":", lw=1)
axes[0].text(1.5e-3, 1 / np.sqrt(3) + 0.003, r"$1/\sqrt{3}$", fontsize=8)
axes[0].text(1.5e-3, atm.Q + 0.003, f"{atm.Q:.4f}", fontsize=8)
axes[0].set_xlabel(r"optical depth $\tau$")
axes[0].set_ylabel(r"$q(\tau)$")
axes[0].set_title("Hopf function")
axes[0].legend(fontsize=8, loc="center right")

# %%
# Temperature structure
# ---------------------
axes[1].semilogx(tau, atm.temperature(tau), color="crimson", label="exact grey")
axes[1].semilogx(tau, eddington_temperature(tau), "k--", label="Eddington")
axes[1].axhline(1.0, color="gray", ls=":", lw=1)
axes[1].axvline(2 / 3, color="gray", ls=":", lw=1)
axes[1].set_xlabel(r"$\tau$")
axes[1].set_ylabel(r"$T / T_{\rm eff}$")
axes[1].set_title(r"$T = T_{\rm eff}$ near $\tau = 2/3$")
axes[1].legend(fontsize=8)

# %%
# Limb darkening
# --------------
# For comparison, the visible solar disc roughly follows the linear law
# I(mu)/I(1) = 1 - u(1 - mu) with u of about 0.6, close to Eddington's
# u = 3/5; the real Sun is not grey, so its darkening varies with
# wavelength.
u_sun = 0.6
sun = 1 - u_sun * (1 - mu)
axes[2].plot(mu, atm.limb_darkening(mu), color="crimson", label="exact grey")
axes[2].plot(mu, eddington_limb_darkening(mu), "k--", label=r"Eddington $(2 + 3\mu)/5$")
axes[2].plot(mu, sun, color="goldenrod", ls=":", label=r"Sun, visible, $u \approx 0.6$")
axes[2].set_xlabel(r"$\mu = \cos\theta$ (1 at disc centre)")
axes[2].set_ylabel(r"$I(0, \mu)/I(0, 1)$")
axes[2].set_title("Limb darkening")
axes[2].legend(fontsize=8)
plt.tight_layout()
plt.show()

flux = 2 * np.trapezoid(atm.emergent_intensity(mu) * mu, mu)
print(f"q(0) = {float(atm.hopf(0.0)):.6f}, q(inf) = {atm.Q:.6f}")
print(f"limb/centre intensity: exact grey {float(atm.limb_darkening(0.0)):.4f}, Eddington 0.4000")
print(f"emergent flux / sigma T_eff^4 = {flux:.5f}")

# %%
# Check
# -----
assert abs(float(atm.hopf(0.0)) - 1 / np.sqrt(3)) < 1e-10
assert abs(atm.Q - 0.710446) < 1e-4
assert abs(float(atm.limb_darkening(0.0)) - 0.3439) < 1e-3
assert abs(flux - 1.0) < 1e-3
