r"""
Sine-Gordon solitons: kink propagation and a kink-antikink breather
=========================================================================

:class:`~physicskit.classical.systems.chains.SineGordonChain` is a
discretized chain of :math:`N` coupled pendulums (the Frenkel-Kontorova
model) with free (Neumann-like) ends, so a topological kink can
propagate off either end. The potential energy is

.. math::

    V(q) = \sum_{i} \left[\frac{k}{2}(q_{i+1} - q_i)^2
        + (1 - \cos q_i)\right] ,

giving the lattice equation of motion :math:`m\ddot q_i =
k(q_{i+1} - 2q_i + q_{i-1}) - \sin q_i`. In the continuum limit
(:math:`m = k = 1`) this becomes the sine-Gordon equation
:math:`\partial_t^2 q - \partial_x^2 q + \sin q = 0`, whose exact
traveling-wave (kink) solution is the Lorentz-boosted profile

.. math::

    q(x, t) = \pm\, 4 \arctan\!\left[
        \exp\!\left(\gamma\,\frac{x - x_0 - v t}{w}\right)\right],
    \qquad \gamma = \frac{1}{\sqrt{1 - v^2}} ,

with :math:`w = \sqrt{k/m} = 1` the natural soliton width, :math:`v`
the boost velocity (:math:`|v| < 1`, the natural wave speed), and the
sign giving a kink (+) or antikink (-).
:meth:`~physicskit.classical.systems.chains.SineGordonChain.kink`
builds this continuum-limit initial condition on the discrete lattice
for a topological kink (or antikink) with a chosen boost velocity --
``width`` must equal ``sqrt(k/m)`` (1, for the default k=m=1) for this
to actually be a valid soliton profile, not an arbitrary shape.

1. A single kink keeps its shape as it propagates (a genuine soliton),
   though on this *discrete* lattice it settles to an effective speed
   below its nominal boost velocity -- lattice discreteness itself
   acts as a drag on the continuum soliton solution, a real effect in
   Frenkel-Kontorova-type chains, not a bug.
2. A kink and antikink launched toward each other at a slow-to-moderate
   speed do not simply pass through: they form a *breather*, a bound,
   periodically-collapsing-and-reforming oscillating state -- one of
   the celebrated exact solutions of the (continuum) sine-Gordon
   equation, visible here as the discrete analogue.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.classical.systems.chains import SineGordonChain

n = 300

# %%
# A single kink: shape-preserving propagation
# --------------------------------------------------

q0, p0 = SineGordonChain.kink(n, center=60, width=1.0, velocity=0.4, polarity=1)
solo = SineGordonChain(n=n, q0=q0, p0=p0)
e0 = solo.energy()
result_solo = solo.integrate((0, 300), dt=0.005, method="yoshida4")
drift_solo = np.max(np.abs(result_solo.energy - e0)) / abs(e0)

front_start = np.interp(np.pi, result_solo.q[0], np.arange(n))
front_end = np.interp(np.pi, result_solo.q[-1], np.arange(n))
effective_speed = (front_end - front_start) / result_solo.t[-1]
print(f"solo kink: energy drift = {drift_solo:.3e}, nominal boost velocity = 0.4, effective lattice speed = {effective_speed:.3f}")

fig1, ax1 = plt.subplots(figsize=(8, 4.5))
for frac in (0.0, 0.33, 0.66, 1.0):
    idx = int(frac * (len(result_solo.t) - 1))
    ax1.plot(np.arange(n), result_solo.q[idx], label=f"t={result_solo.t[idx]:.0f}")
ax1.set_xlabel("lattice site")
ax1.set_ylabel(r"$q_i$")
ax1.set_title("A single kink keeps its shape while it propagates")
ax1.legend()

# %%
# Kink + antikink collision: a breather, not a pass-through
# ------------------------------------------------------------------

q_k, p_k = SineGordonChain.kink(n, center=100, width=1.0, velocity=0.4, polarity=1)
q_ak, p_ak = SineGordonChain.kink(n, center=200, width=1.0, velocity=-0.4, polarity=-1)
chain = SineGordonChain(n=n, q0=q_k + q_ak, p0=p_k + p_ak)
e0b = chain.energy()
result = chain.integrate((0, 600), dt=0.005, method="yoshida4")
drift = np.max(np.abs(result.energy - e0b)) / abs(e0b)
print(f"collision: energy drift = {drift:.3e}")

fig2, ax2 = plt.subplots(figsize=(7, 5.5))
extent = [0, n, result.t[-1], result.t[0]]
im = ax2.imshow(result.q[::100], aspect="auto", extent=extent, cmap="twilight", vmin=-2 * np.pi, vmax=2 * np.pi)
ax2.set_xlabel("lattice site")
ax2.set_ylabel("t")
ax2.set_title("Kink + antikink collision forms a breather (bound, oscillating), not a pass-through")
fig2.colorbar(im, ax=ax2, label=r"$q_i(t)$")

plt.show()
