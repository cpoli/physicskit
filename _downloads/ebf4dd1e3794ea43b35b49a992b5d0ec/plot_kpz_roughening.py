r"""
Universal interface roughening: the Kardar-Parisi-Zhang universality class
=================================================================================

Kardar, Parisi, and Zhang's 1986 equation for a growing interface,

.. math::

    \frac{\partial h}{\partial t} = \nu \nabla^2 h
    + \frac{\lambda}{2}(\nabla h)^2 + \eta(x, t),

predicts universal scaling for the interface width :math:`w(t) =
\sqrt{\langle (h - \bar h)^2 \rangle}`: early-time growth :math:`w \sim
t^\beta` with :math:`\beta = 1/3`, crossing over to a size-limited
saturation :math:`w_{\text{sat}} \sim L^\alpha` with :math:`\alpha = 1/2`
once :math:`t` exceeds the correlation time :math:`\tau \sim L^z`
(:math:`z = 3/2`). Rather than integrate this delicate nonlinear SPDE
directly, this example uses the Restricted Solid-On-Solid (RSOS) growth
automaton -- a purely local, parameter-free deposition rule long known to
lie in the same universality class -- and recovers both exponents directly
from simulation.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.statphys.chapters.kpz_growth import KPZInterface
from physicskit.statphys.utils.finite_size_scaling import power_law_exponent
from physicskit.statphys.visualizers.kpz_render import plot_width_growth

# %%
# Early-time growth: w(t) ~ t^(1/3)
# ------------------------------------------------------------
# A single large interface (L=400) is grown far short of its own
# saturation time (~L^1.5), so the width is still in the free, unsaturated
# growth regime throughout.
interface = KPZInterface(L=400, seed=0)
times, widths = interface.run_growth_curve(t_max=1500, n_points=30)
beta_fit, _ = power_law_exponent(times, widths)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
plot_width_growth(times, widths, ax=axes[0], beta=1.0 / 3.0)
axes[0].set_title(f"Growth regime: fitted beta = {beta_fit:.3f} (theory 1/3)")

# %%
# Saturation: w_sat(L) ~ L^(1/2)
# ------------------------------------------------------------
# Several smaller interfaces are grown well *past* their own saturation
# time (5 L^1.5 sweeps), and their long-time-averaged width is recorded --
# the finite-size-limited plateau the growth regime above crosses over
# into.
L_values = np.array([32, 64, 96, 128])
w_sat = []
for L in L_values:
    iface = KPZInterface(L=int(L), seed=1)
    iface.grow(n_sweeps=int(5 * L**1.5))  # run well past the L^1.5 saturation time
    samples = []
    for _ in range(20):
        iface.grow(n_sweeps=int(0.2 * L**1.5))
        samples.append(iface.width())
    w_sat.append(np.mean(samples))
w_sat = np.array(w_sat)
alpha_fit, _ = power_law_exponent(L_values, w_sat)

axes[1].loglog(L_values, w_sat, marker="o", linestyle="none", label="simulation")
amplitude = w_sat[0] / L_values[0] ** 0.5
axes[1].loglog(L_values, amplitude * L_values**0.5, linestyle="--", label=r"$L^{1/2}$")
axes[1].set_xlabel("L")
axes[1].set_ylabel(r"$w_{\text{sat}}(L)$")
axes[1].set_title(f"Saturation regime: fitted alpha = {alpha_fit:.3f} (theory 1/2)")
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Growth exponent beta = {beta_fit:.3f} (KPZ theory: 1/3)")
print(f"Roughness exponent alpha = {alpha_fit:.3f} (KPZ theory: 1/2)")
