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
