r"""
Landau damping from first principles with particle-in-cell
=================================================================

In 1946, Lev Landau solved the linearized Vlasov equation for a wave
on a Maxwellian plasma and found something that seemed to violate
reversible mechanics: the wave damps, exponentially, with no
collisions or dissipation anywhere in the equations. Particles moving
at very nearly the wave's phase velocity "surf" it, and because a
thermal distribution always has slightly more slightly slower
particles being accelerated than slightly faster particles being
decelerated, the net exchange drains the wave.

The system is a collisionless electron gas of distribution function
:math:`f(x,v,t)` against a fixed, uniform, charge-neutralizing ion
background, coupled to its own self-consistent electric field :math:`E`
through Poisson's equation:

.. math::

   \partial_t f + v\,\partial_x f - \frac{e}{m}E\,\partial_v f = 0,
   \qquad \partial_x E = \rho.

A single density perturbation of wavenumber :math:`k` is seeded on a
Maxwellian background of thermal speed :math:`v_{th}` in a periodic box
of length :math:`L=2\pi/k`; linearizing this system predicts the wave's
field energy decays at the rate

.. math::

   \gamma = -\omega_{pe}\sqrt{\frac{\pi}{8}}\,\frac{1}{(k\lambda_D)^3}\,
   \exp\!\left(-\frac{1}{2(k\lambda_D)^2}-\frac{3}{2}\right),
   \qquad \lambda_D = v_{th}/\omega_{pe},

valid for weak damping, :math:`k\lambda_D \lesssim 0.5`.

:func:`~physicskit.plasma.kinetic.landau_damping_rate` evaluates the
analytic weak-damping formula above, and
:func:`~physicskit.plasma.kinetic.pic_simulate` reproduces the decay
from first principles with a particle-in-cell solution of the same
Vlasov equation Landau linearized -- no damping term appears anywhere
in :func:`~physicskit.plasma.kinetic.pic_step`; the decay is a pure
consequence of the particle orbits.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

import physicskit as pk

# %%
# Classic benchmark: k * lambda_D = 0.5, one wavelength in the box
# ------------------------------------------------------------------------

k_mode, v_th = 0.5, 1.0
L = 2 * np.pi / k_mode
x0, v0 = pk.plasma.landau_damping_ic(4000, L=L, k_mode=k_mode, alpha=0.01, v_th=v_th, seed=0)

# %%
# Evolve the Vlasov-Poisson system with particles alone -- no collision
# operator or damping term appears anywhere in pic_step.

result = pk.plasma.pic_simulate(x0, v0, L=L, ng=64, dt=0.1, steps=300)

# %%
# Compare the simulated field-energy decay against Landau's analytic
# rate (field energy ~ E^2, so it decays as exp(2*gamma*t)).

gamma = pk.plasma.landau_damping_rate(k=k_mode, v_th=v_th)
t, fe = result["t"], result["field_energy"]
fig, ax = pk.plasma.plot_field_energy_history(t, fe)
t0 = 50
ax.semilogy(t, fe[t0] * np.exp(2 * gamma * (t - t[t0])), "--", label=f"analytic $2\\gamma={2 * gamma:.3f}$")
ax.set_title("Landau damping: PIC simulation vs. analytic rate")
ax.legend()
fig.tight_layout()

plt.show()

# %%
# Where the energy actually goes: phase-space flattening
# ------------------------------------------------------------------------
# The exponentially decaying field energy above is only the field's share
# of the story; the same mechanism that damps the wave is visible directly
# in the particle distribution. Comparing the initial and final phase-space
# density shows the resonant particles near the wave's phase velocity
# being scattered into a flattened plateau, exactly the quasilinear
# signature of the particles absorbing the field energy the wave loses.

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
pk.plasma.plot_phase_space(x0, v0, ax=axes[0])
axes[0].set_title("t=0: seeded density ripple")
pk.plasma.plot_phase_space(result["x"], result["v"], ax=axes[1])
axes[1].set_title(f"t={t[-1]:.0f}: resonant plateau, wave damped")
fig.suptitle("Landau damping: phase-space flattening, not just field decay")
fig.tight_layout()

plt.show()
