r"""
Extreme Sensitivity: a Slightly Truncated Circle
=====================================================

A billiard particle moves in a straight line and reflects specularly off
its boundary, :math:`\mathbf{v}' = \mathbf{v} - 2(\mathbf{v}\cdot\mathbf{n})
\,\mathbf{n}`. The circular billiard (boundary :math:`x^2+y^2=r^2`) is
*exactly* integrable: :math:`\sin\phi` -- the sine of the angle of incidence
at each bounce -- is conserved forever, so two rays launched a hair's
breadth apart stay a hair's breadth apart forever too. Slicing off even a
tiny flat chord (turning it into a :class:`~physicskit.chaos.systems.billiards.TruncatedCircleBilliard`
of boundary parameter ``cut``) destroys this completely. This example
launches two rays :math:`10^{-8}` radians apart into a circle truncated by a
chord just 5% of the radius deep -- barely different from a perfect circle
by eye -- and watches them track each other indistinguishably for dozens of
bounces before a single grazing hit near the chord sends them down
completely different paths.
"""

import matplotlib.pyplot as plt

from physicskit.chaos.systems.billiards import CircleBilliard, TruncatedCircleBilliard
from physicskit.chaos.visualizers import (
    animate_billiard_divergence,
    billiard_trajectory_divergence,
    plot_billiard_divergence,
)

# %%
# A perfect circle never diverges
# -------------------------------------
# As a baseline, launch the same pair of nearly-identical rays into an
# *untruncated* circle: because ``sin(phi)`` is exactly conserved, their
# separation is exactly preserved too, bounce after bounce -- no exponential
# growth, no sensitivity, because there is no chaos to be sensitive to.
circle = CircleBilliard(radius=1.0)
_, separation_circle = billiard_trajectory_divergence(circle, pos=circle.sample_interior_point(), vel=(0.2, 0.0), delta_0=1e-8, n_bounces=100)
print(f"Perfect circle: separation after 100 bounces = {separation_circle[-1]:.3e} (started at 1e-08)")

# %%
# Animation: two rays, one tiny chord
# ------------------------------------------
# Watch the reference (blue) and perturbed (red) rays bounce together,
# overlapping so closely they look like a single ray, until they don't.
billiard = TruncatedCircleBilliard(radius=1.0, cut=0.001)
pos = billiard.sample_interior_point()
vel = (0.2, 0.0)

anim = animate_billiard_divergence(billiard, pos, vel, delta_0=1e-8, n_bounces=100, trail=25)

plt.show()

# %%
# To save the animation to a file instead of (or in addition to) displaying
# it interactively, use e.g.::
#
#     anim.save("extreme_sensitivity_animation.gif", writer="pillow", fps=25)

# %%
# Quantifying it: exponential divergence
# ---------------------------------------------
# ``ln(separation / delta_0)`` climbs in a distinctive staircase -- flat
# stretches where both rays are still circling together far from the chord,
# punctuated by jumps each time one (and then, soon after, the other) ray
# grazes it -- before saturating once the two rays are effectively
# uncorrelated, at the scale of the table itself.
fig, ax, lam = plot_billiard_divergence(billiard, pos, vel, delta_0=1e-8, n_bounces=100)
ax.set_title(f"{ax.get_title()} (cut = {billiard.cut})")

plt.show()
