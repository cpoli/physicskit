r"""
Flamm's paraboloid: gravity as curved geometry, not a force
==================================================================

General Relativity's central insight is that gravity is not a force
propagating through space, but the curvature of spacetime itself -- massive
objects don't pull on other objects, they bend the geometry that other
objects move through in a straight line (a geodesic). Restricting the
Schwarzschild metric to a constant-time, equatorial (:math:`\theta=\pi/2`)
slice leaves a curved 2D spatial geometry with proper length element

.. math::

    d\ell^2 = \frac{dr^2}{1 - 2M/r} + r^2 d\phi^2

Flamm's paraboloid makes this curvature literal: embedding this 2-surface
as a surface of revolution :math:`z(r)` in ordinary flat 3D Euclidean space,

.. math::

    z(r) = 2\sqrt{2M(r - 2M)}, \qquad r \ge 2M

reproduces exactly the same proper distances as the curved metric above, so
that walking along the resulting funnel-shaped surface covers the same
proper distance as walking through the real curved space around the black
hole -- for masses :math:`M` of increasing size (and correspondingly larger
horizons :math:`r=2M`).
"""

import matplotlib.pyplot as plt

from physicskit.relativity.chapters.schwarzschild import SchwarzschildBlackHole
from physicskit.relativity.visualizers.spacetime_3d import plot_flamm_paraboloid

# %%
# The embedding surface for black holes of increasing mass
# ------------------------------------------------------------
fig = plt.figure(figsize=(13, 4.5))
for i, M in enumerate([0.5, 1.0, 2.0]):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    plot_flamm_paraboloid(M, ax=ax, r_max=15.0)
    bh = SchwarzschildBlackHole(M=M)
    ax.set_title(f"M={M} (horizon at r={bh.horizon_radius}M)")
plt.tight_layout()
plt.show()
