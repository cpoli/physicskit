r"""
The Elastic Pendulum and 1:2 Autoparametric Resonance
============================================================

:class:`~physicskit.classical.systems.lagrangian.ElasticPendulum` is a
mass :math:`m` on a Hookean spring of natural length :math:`L_0` and
stiffness :math:`k`, free to both stretch and swing. With generalized
coordinates :math:`q = (s, \theta)` -- :math:`s` the spring's stretch
beyond :math:`L_0` (so the pivot-to-mass distance is :math:`r = L_0 +
s`) and :math:`\theta` the swing angle from the downward vertical --
its Lagrangian is

.. math::

    L = \frac{m}{2}\left[\dot s^2 + (L_0 + s)^2 \dot\theta^2\right]
        - \frac{k}{2} s^2 + m g (L_0 + s)\cos\theta .

The :math:`(L_0+s)^2\dot\theta^2` term is what couples the (otherwise
independent, linear) stretch and swing motions: whenever the mass
swings, its varying distance from the pivot pumps the spring, and vice
versa. Written alone each mode would oscillate at its own natural
frequency, :math:`\omega_s = \sqrt{k/m}` for the stretch and
:math:`\omega_\theta = \sqrt{g/L_0}` for the swing; the coupling
becomes strong -- and famously resonant -- whenever
:math:`\omega_s \approx 2\,\omega_\theta` (*1:2 autoparametric
resonance*), where energy started as pure vertical stretching
oscillation periodically leaks into, and back out of, swinging.
``ElasticPendulum``'s default parameters (:math:`k = 4mg/L_0`) are set
to this 1:2 resonance condition exactly.
"""

# %%
import matplotlib.pyplot as plt

from physicskit.classical.systems.lagrangian import ElasticPendulum
from physicskit.classical.visualizers.animations import animate_elastic_pendulum

# %%
# Energy exchange between stretching and swinging
# ------------------------------------------------------
# Start mostly stretched, with only a small swing seed, and integrate
# long enough to see the swing amplitude visibly beat.

system = ElasticPendulum(q0=[0.4, 0.05], qdot0=[0.0, 0.0])
result = system.integrate((0.0, 60.0), dt=0.1, method="implicit_midpoint")

fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax1.plot(result.t, result.q[:, 0], color="steelblue")
ax1.set_ylabel("s (stretch)")
ax1.set_title("Elastic pendulum: energy leaking between modes")
ax2.plot(result.t, result.q[:, 1], color="firebrick")
ax2.set_ylabel(r"$\theta$ (swing)")
ax2.set_xlabel("t")
fig1.tight_layout()

# %%
# Animation: the coil stretching and swinging together
# -------------------------------------------------------------
anim = animate_elastic_pendulum(system, result, interval=20, trail=300)

plt.show()

# %%
# To save the animation to a file instead of (or in addition to) displaying
# it interactively, use e.g.::
#
#     anim.save("elastic_pendulum_animation.gif", writer="pillow", fps=30)
