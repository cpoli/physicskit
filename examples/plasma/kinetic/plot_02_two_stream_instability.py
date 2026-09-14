r"""
The two-stream instability: exponential growth and phase-space vortices
=============================================================================

Studying velocity-modulated electron beams for microwave tubes, Andrew
Haeff (1948) and John Pierce (1948) independently found that a beam of
electrons streaming through a background plasma spontaneously breaks up
into density bunches that grow exponentially. David Bohm and Eugene
Gross (1949) showed the instability is the mirror image of Landau
damping (:doc:`plot_01_landau_damping`): two counter-streaming beams
create a region of *positive* velocity-space slope between their peaks,
and it is precisely that positive slope -- formalized a decade later as
the Penrose criterion -- that pumps energy from the particles into the
wave instead of draining it.

The governing equations are identical to :doc:`plot_01_landau_damping`'s
-- the collisionless Vlasov equation for the electron distribution
:math:`f(x,v,t)` coupled to its own field through Poisson's equation,

.. math::

   \partial_t f + v\,\partial_x f - \frac{e}{m}E\,\partial_v f = 0,
   \qquad \partial_x E = \rho.

Here the electrons are split into two counter-streaming Maxwellian
beams centered at :math:`v=\pm v_{drift}` with thermal spread
:math:`v_{th}`, so :math:`f(v)` is doubly-peaked instead of single-humped.
Whenever :math:`v_{drift}` is large enough relative to :math:`v_{th}` for
a region of positive slope, :math:`\partial f/\partial v > 0`, to appear
between the two peaks, the Penrose criterion is violated and a small
seeded density ripple grows exponentially instead of Landau-damping away.

:func:`~physicskit.plasma.kinetic.two_stream_ic` seeds exactly this
doubly-peaked, counter-streaming initial condition; run through the
same :func:`~physicskit.plasma.kinetic.pic_simulate` used for Landau
damping, it reproduces exponential *growth* instead of decay from the
identical Vlasov-Poisson machinery -- proof that both phenomena are two
faces of one kinetic mechanism.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

import physicskit as pk

# %%
# Two counter-streaming beams, drift speed well above thermal spread
# ------------------------------------------------------------------------
# so the combined distribution is doubly-peaked and unstable.

k_seed = 0.3
L = 2 * np.pi / k_seed
x0, v0 = pk.plasma.two_stream_ic(20000, L=L, v_drift=3.0, v_th=0.5, seed=0)

# %%
# The same deposit-solve-gather-push PIC loop used for Landau damping
# now amplifies the seeded ripple instead of damping it.

result = pk.plasma.pic_simulate(x0, v0, L=L, ng=64, dt=0.05, steps=400)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
pk.plasma.plot_field_energy_history(result["t"], result["field_energy"], ax=axes[0])
axes[0].set_title("Exponential field-energy growth")

# %%
# The saturated phase-space snapshot shows the beams rolled into a
# single vortex, not two straight, undisturbed streams.

pk.plasma.plot_phase_space(result["x"], result["v"], ax=axes[1])
axes[1].set_title("Saturated phase-space vortex")
fig.tight_layout()

plt.show()

# %%
# Watching the vortex form, continuously
# ------------------------------------------------------------------------
# Rather than just a before/after snapshot, redrawing the full particle
# scatter every few PIC steps shows the two straight, separate beams
# wrapping continuously around each other into the single saturated
# phase-space vortex.

x0, v0 = pk.plasma.two_stream_ic(8000, L=L, v_drift=3.0, v_th=0.5, seed=0)

anim = pk.plasma.animate_two_stream_phase_space(x0, v0, L=L, ng=64, dt=0.05, steps_per_frame=4, n_frames=60)

plt.show()
