r"""
Animating the shadow: a black hole spun up from rest
================================================================================

The static, side-by-side spin comparison in
:doc:`plot_kerr_shadow_and_redshift` already shows that a black hole's
shadow shrinks and its accretion disk creeps inward as the spin :math:`a`
increases -- but a continuous sweep makes the *mechanism* far more vivid:
watch the innermost stable circular orbit (ISCO) contract from
:math:`6M` (Schwarzschild) down toward :math:`M` (extremal Kerr, prograde),
dragging the disk's bright inner edge inward with it, while the shadow
itself flattens and shifts off-center from frame dragging.

.. math::

    r_{\text{ISCO}}(a) : \quad
    6M \ \xrightarrow{a \to M} \ M \qquad \text{(prograde)},

each frame's disk inner edge set to exactly this spin-dependent ISCO.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.relativity.chapters.kerr import KerrBlackHole
from physicskit.relativity.visualizers.shadow_render import animate_shadow_spin_sweep

# %%
# Sweep the spin from Schwarzschild to near-extremal Kerr
# ------------------------------------------------------------
# Each frame is a full independent backward ray-trace at that frame's spin,
# with the disk's inner edge set to that spin's ISCO and the disk shaded by
# its combined gravitational and Doppler redshift, so the Doppler-brightened
# approaching side becomes progressively more pronounced as the disk edges
# closer to the horizon.
anim = animate_shadow_spin_sweep(
    M=1.0,
    ny=140,
    nx=140,
    inclination=1.3,
    r_disk_outer=16.0,
    redshift=True,
)
plt.show()

# %%
# To save the animation to a file instead of (or in addition to) displaying
# it interactively, use e.g.::
#
#     anim.save("kerr_shadow_spin_sweep.gif", writer="pillow", fps=5)


# %%
# Check
# -----
def isco_bpt(a, prograde=True):
    # Bardeen, Press and Teukolsky (1972), M = 1.
    z1 = 1 + (1 - a**2) ** (1 / 3) * ((1 + a) ** (1 / 3) + (1 - a) ** (1 / 3))
    z2 = np.sqrt(3 * a**2 + z1**2)
    return 3 + z2 - np.sqrt((3 - z1) * (3 + z1 + 2 * z2)) * (1 if prograde else -1)


# Each frame's disk starts at the spin's ISCO, contracting from 6M toward
# M as the spin grows (Bardeen-Press-Teukolsky).
spins = np.linspace(0.0, 0.98, 50)
isco = np.array([6.0] + [KerrBlackHole(M=1.0, a=a).isco_radius(prograde=True) for a in spins[1:]])
np.testing.assert_allclose(isco, isco_bpt(spins), rtol=1e-6)
assert np.all(np.diff(isco) < 0) and isco[-1] < 1.7
