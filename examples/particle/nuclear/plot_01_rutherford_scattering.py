r"""
Rutherford scattering and the discovery of the nucleus
============================================================

Geiger and Marsden's 1909 observation that some alpha particles fired at
gold foil bounced back at large angles was inexplicable under Thomson's
diffuse "plum pudding" atom. Rutherford (1911) showed that treating the
alpha particle and nucleus as point charges interacting via the Coulomb
force predicts exactly this large-angle scattering, via the differential
cross section

.. math::

    \frac{d\sigma}{d\Omega} = \left(\frac{Z_1Z_2\alpha}{4E_{\rm kin}}\right)^2
    \frac{1}{\sin^4(\theta/2)}.

This example plots :func:`~physicskit.particle.scattering.rutherford_dsigma_domega`'s
steep small-angle divergence and large-angle tail directly, and uses
:func:`~physicskit.particle.scattering.impact_parameter` to show that
the large-angle scattering Geiger and Marsden actually detected requires
the incoming alpha particle to pass implausibly close to the nucleus --
exactly the geometric argument that revealed how small and concentrated
the nucleus must be.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.particle.scattering import impact_parameter, rutherford_dsigma_domega
from physicskit.particle.visualizers import plot_differential_cross_section

# %%
# The differential cross section: alpha particles on gold
# ----------------------------------------------------------------
Z_alpha, Z_gold = 2, 79
E_kin = 5.0  # MeV, a typical natural alpha-source energy

theta = np.linspace(0.02, np.pi - 0.02, 400)
dsigma = rutherford_dsigma_domega(theta, Z_alpha, Z_gold, E_kin)

fig1, ax1 = plot_differential_cross_section(theta, dsigma)
ax1.set_title(r"Rutherford scattering: $\alpha$ on gold, steep $1/\sin^4(\theta/2)$ falloff")
fig1.tight_layout()

# Verify the exact scaling analytically underlying the plot: the
# quantity vals*sin^4(theta/2) should be perfectly flat.
flat_check = dsigma * np.sin(theta / 2.0) ** 4
print(f"d(sigma)*sin^4(theta/2): min={flat_check.min():.6e}, max={flat_check.max():.6e} (should be constant)")

# %%
# The geometric argument: impact parameter for large-angle scattering
# ------------------------------------------------------------------------------
# A scattering angle near 180 degrees -- the "utterly incredible" result
# Rutherford compared to a shell bouncing off tissue paper -- requires an
# alpha particle to pass almost dead-center through the nucleus. The
# impact parameter for a 90-degree deflection already sets the scale of
# how close an approach is needed.
theta_probe = np.radians([1.0, 10.0, 90.0, 179.0])
b_values = impact_parameter(theta_probe, Z_alpha, Z_gold, E_kin)
print("\nscattering angle -> impact parameter needed (in natural units, hbar*c/MeV):")
for th_deg, b in zip([1.0, 10.0, 90.0, 179.0], b_values):
    print(f"  theta={th_deg:6.1f} deg  ->  b = {b:.4e}")
print(f"\nb(179 deg)/b(1 deg) = {b_values[-1] / b_values[0]:.2e}: only a vanishingly rare, nearly head-on")
print("approach produces the large-angle scattering actually observed -- exactly the argument that")
print("located essentially all of the atom's charge and mass in a minuscule central nucleus.")

fig2, ax2 = plt.subplots(figsize=(6, 4.5))
theta_fine = np.linspace(1.0, 179.0, 300)
ax2.semilogy(theta_fine, impact_parameter(np.radians(theta_fine), Z_alpha, Z_gold, E_kin), color="darkorange")
ax2.set_xlabel(r"scattering angle $\theta$ (degrees)")
ax2.set_ylabel("impact parameter b (natural units)")
ax2.set_title("Larger deflection requires a closer approach")
fig2.tight_layout()

plt.show()
