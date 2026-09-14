r"""
Russell's wave of translation
================================

In 1834, John Scott Russell watched a solitary heap of water detach from
a boat's bow wave on the Union Canal and roll on for miles "without
change of form or diminution of speed." Sixty years later, Korteweg and
de Vries explained this with the equation now bearing their names,

.. math::

    \partial_t u + 6 u\,\partial_x u + \partial_x^3 u = 0,

whose exact single-hump traveling-wave solution,
:math:`u(x,t) = \tfrac{c}{2}\,\mathrm{sech}^2\!\big(\tfrac{\sqrt{c}}{2}(x-x_0-ct)\big)`,
:func:`~physicskit.fields.solitons.kdv_soliton` constructs directly.
This reproduces Russell's observation itself by propagating that hump
forward on a periodic domain with :func:`~physicskit.fields.solitons.kdv_evolve`
and checking it against its own exact translated copy, and animates the
whole "roll down the canal" as a frame-by-frame movie via
:func:`~physicskit.fields.solitons.kdv_evolve_frames`.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.fields import animate_field_1d, kdv_evolve, kdv_evolve_frames, kdv_soliton, plot_field_1d

# %%
# A solitary "heap of water": a single KdV hump
# ------------------------------------------------

N, L, c = 512, 60.0, 4.0
x = np.linspace(-L / 2, L / 2, N, endpoint=False)
u0 = kdv_soliton(x, c=c, x0=-15.0)

# %%
# Let it "roll on for miles" down the periodic canal
# ------------------------------------------------------

steps, dt = 6000, 0.0005
u = kdv_evolve(u0, x, dt=dt, steps=steps)

# %%
# Russell's claim holds to five decimal places: the propagated hump is
# identical in shape to the exact soliton shifted by exactly ``c * t`` --
# "without change of form or diminution of speed."

predicted = kdv_soliton(x, c=c, x0=-15.0 + c * steps * dt)
shape_error = np.max(np.abs(u - predicted))

fig, ax = plot_field_1d(x, u0, label="t = 0")
plot_field_1d(x, u, ax=ax, label=f"t = {steps * dt:.0f}")
ax.set_title(f"shape error after propagation: {shape_error:.1e}")
fig.tight_layout()

print(f"shape error vs. exact translated soliton: {shape_error:.2e}")
print(f"peak amplitude: t=0 -> {u0.max():.4f}, t={steps * dt:.0f} -> {u.max():.4f} (c/2 = {c / 2:.4f})")

# %%
# Animating the roll down the canal
# -------------------------------------
# :func:`~physicskit.fields.solitons.kdv_evolve_frames` records the same
# propagation as a sequence of snapshots, so the "unchanged in form,
# rolling on for miles" behavior can be watched directly rather than
# compared only at two instants.

frames, times = kdv_evolve_frames(u0, x, dt=dt, steps_per_frame=steps // 40, n_frames=40)

anim = animate_field_1d(x, frames, times, ylabel="u(x, t)")
plt.show()

# %%
# To save the animation to a file instead of (or in addition to)
# displaying it interactively, use e.g.::
#
#     anim.save("kdv_translation.gif", writer="pillow", fps=20)
