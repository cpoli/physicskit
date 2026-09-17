r"""
A topologically protected Sine-Gordon kink
==============================================

Yakov Frenkel and Tatiana Kontorova modeled a crystal dislocation as a
chain of atoms in a periodic substrate potential -- a chain of coupled
pendula; in the continuum limit this becomes the Sine-Gordon equation

.. math::

    \partial_t^2 u - \partial_x^2 u + \sin u = 0.

Its kink solution,

.. math::

    u(x,t) = 4\arctan\!\left(e^{\,\gamma(x-x_0-vt)}\right), \qquad
    \gamma = \frac{1}{\sqrt{1-v^2}},

a single :math:`2\pi` twist propagating at speed :math:`v<1` along the
chain, is a *topological* soliton: no local, continuous deformation can
untwist it. :func:`~physicskit.fields.solitons.sine_gordon_kink` builds
this exact traveling-wave solution and
:func:`~physicskit.fields.solitons.sine_gordon_evolve` propagates it
with independent leapfrog finite differences, confirming the twist
survives intact; the same run can be watched frame by frame with
:func:`~physicskit.fields.solitons.sine_gordon_evolve_frames`.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.fields import animate_field_1d, plot_field_1d, sine_gordon_evolve, sine_gordon_evolve_frames, sine_gordon_kink

# %%
# A kink launched at v = 0.5 (natural units, linear wave speed = 1)
# ------------------------------------------------------------------------

N, L, v, x0 = 4000, 200.0, 0.5, -50.0
x = np.linspace(-L / 2, L / 2, N)
dx = x[1] - x[0]
dt = 0.4 * dx
u_prev = sine_gordon_kink(x, -dt, v, x0)
u0 = sine_gordon_kink(x, 0.0, v, x0)

# %%
# Propagate with leapfrog finite differences on the discrete chain
# -----------------------------------------------------------------------

steps = 1000
u, _ = sine_gordon_evolve(u0, u_prev, x, dt, steps)

# %%
# The :math:`2\pi` twist survives intact, at the position predicted by the
# exact kink solution -- it cannot be untwisted by the local dynamics,
# only shifted.

expected = sine_gordon_kink(x, steps * dt, v, x0)
max_dev = np.max(np.abs(u - expected))

fig, ax = plot_field_1d(x, u0, label="t = 0")
plot_field_1d(x, u, ax=ax, label=f"t = {steps * dt:.0f}")
ax.set_title(f"max deviation from exact kink: {max_dev:.2e}")
fig.tight_layout()

print(f"field before the kink (x -> -L/2): {u[0]:.4f} (expect 0)")
print(f"field after the kink  (x -> +L/2): {u[-1]:.4f} (expect 2*pi = {2 * np.pi:.4f})")
print(f"max deviation from exact traveling kink: {max_dev:.2e}")

# %%
# Animating the traveling kink
# ----------------------------------
# :func:`~physicskit.fields.solitons.sine_gordon_evolve_frames` records
# the same leapfrog propagation as a sequence of snapshots, so the
# topologically protected twist can be watched traveling down the chain
# rather than compared only at the start and end.

frames, times = sine_gordon_evolve_frames(u0, u_prev, x, dt, steps_per_frame=steps // 40, n_frames=40)

anim = animate_field_1d(x, frames, times, ylabel="u(x, t)")
plt.show()

# %%
# To save the animation to a file instead of (or in addition to)
# displaying it interactively, use e.g.::
#
#     anim.save("sine_gordon_kink.gif", writer="pillow", fps=20)
