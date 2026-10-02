r"""
Thin-film interference: the antireflection coating
==================================================

Light reflected from the top and bottom of a thin film interferes, which is
why soap bubbles and oil slicks show colours. Around 1935 Alexander Smakula
at Zeiss turned this into the antireflection coating. A film of index
:math:`n_1` and optical thickness :math:`\lambda_0/4` on glass of index
:math:`n_s` sends back two reflections half a wave apart, and their
amplitudes cancel when :math:`n_1 = \sqrt{n_0 n_s}`:

.. math::

    R(\lambda_0) = \left(\frac{n_0 n_s - n_1^2}{n_0 n_s + n_1^2}\right)^2.

Abelès (1950) put any stack of films into one product of :math:`2\times 2`
characteristic matrices, which is how coatings are designed today. This
example computes the colours of a soap film as it thins, the
reflectance of glass with no coating, with a single MgF\ :sub:`2` layer,
and with a two-layer quarter-quarter coating, and the angle dependence of
the single layer.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.optics.thin_films import multilayer_response

lam = np.linspace(400.0, 750.0, 351)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# %%
# Soap-film colours
# -----------------
# Reflectance of a free-standing water film (n = 1.33) against thickness
# and wavelength. As the film drains, the bands sweep through the spectrum
# until, thinner than about lambda/4n, it turns black.
thickness = np.linspace(0.0, 800.0, 400)
R_soap = np.array([multilayer_response([1.33], [t], lam, 1.0, 1.0)["R"] for t in thickness])
axes[0].pcolormesh(lam, thickness, R_soap, cmap="magma", shading="auto")
axes[0].set_xlabel("wavelength (nm)")
axes[0].set_ylabel("film thickness (nm)")
axes[0].set_title("Soap film reflectance")

# %%
# Antireflection coatings on glass
# --------------------------------
n_glass, n_mgf2, n_high = 1.52, 1.38, 1.70
lam0 = 550.0
bare = multilayer_response([], [], lam, 1.0, n_glass)["R"]
single = multilayer_response([n_mgf2], [lam0 / (4 * n_mgf2)], lam, 1.0, n_glass)["R"]
ideal = multilayer_response([np.sqrt(n_glass)], [lam0 / (4 * np.sqrt(n_glass))], lam, 1.0, n_glass)["R"]
# quarter-quarter (MgF2 on a higher-index layer): zero reflectance when n2/n1 = sqrt(ns/n0)
double = multilayer_response([n_mgf2, n_high], [lam0 / (4 * n_mgf2), lam0 / (4 * n_high)], lam, 1.0, n_glass)["R"]
axes[1].plot(lam, 100 * bare, "k-", label="bare glass")
axes[1].plot(lam, 100 * single, color="steelblue", label=r"$\lambda/4$ MgF$_2$")
axes[1].plot(lam, 100 * ideal, color="steelblue", ls="--", label=r"$\lambda/4$, $n = \sqrt{n_s}$")
axes[1].plot(lam, 100 * double, color="crimson", label=r"$\lambda/4$ MgF$_2$ + $\lambda/4$ n=1.70")
axes[1].set_xlabel("wavelength (nm)")
axes[1].set_ylabel("reflectance (%)")
axes[1].set_title("Antireflection coatings")
axes[1].legend(fontsize=8)
R_single_exact = ((n_glass - n_mgf2**2) / (n_glass + n_mgf2**2)) ** 2
i0 = np.argmin(np.abs(lam - lam0))
print(f"bare glass R = {bare[i0]:.4f}; MgF2 at 550 nm: R = {single[i0]:.5f} (closed form {R_single_exact:.5f})")

# %%
# Angle of incidence
# ------------------
angles = np.radians(np.linspace(0, 80, 81))
for pol, ls in [("s", "-"), ("p", "--")]:
    Rb = [multilayer_response([], [], lam0, 1.0, n_glass, a, pol)["R"] for a in angles]
    Rc = [multilayer_response([n_mgf2], [lam0 / (4 * n_mgf2)], lam0, 1.0, n_glass, a, pol)["R"] for a in angles]
    axes[2].plot(np.degrees(angles), 100 * np.array(Rb), "k", ls=ls, label=f"bare, {pol}")
    axes[2].plot(np.degrees(angles), 100 * np.array(Rc), color="steelblue", ls=ls, label=f"MgF$_2$, {pol}")
axes[2].set_xlabel("angle of incidence (deg)")
axes[2].set_ylabel("reflectance at 550 nm (%)")
axes[2].set_ylim(0, 40)
axes[2].set_title("The coating works up to about 40 degrees")
axes[2].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %%
# Check
# -----
assert abs(single[i0] - R_single_exact) < 1e-12
assert ideal[i0] < 1e-12 and double[i0] < single[i0]
assert np.all(single < bare)
