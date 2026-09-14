r"""
The Dzhanibekov effect, literally: a tumbling 3D rigid body
==================================================================

The same torque-free rigid body as :doc:`plot_01_dzhanibekov` --
:class:`~physicskit.classical.systems.rotations.EulerTop`, governed by
Euler's equations
:math:`I_i\dot\omega_i = (I_j - I_k)\,\omega_j\omega_k` (cyclic in
:math:`i,j,k`) for the body-frame angular velocity, coupled to the
orientation quaternion. The SO(3) momentum-sphere picture
(:doc:`plot_01_dzhanibekov`) shows the *abstract* geometry behind the
intermediate axis theorem. This instead draws the actual rigid body --
a flattened "paddle" (book/phone-like) box, elongated along its
smallest-moment axis and flattened along its largest-moment axis,
exactly like the real object in the famous ISS video -- and rotates it
in 3D with
:func:`~physicskit.classical.visualizers.animations.animate_rigid_body_tumble`,
which builds the wireframe box from ``half_extents`` and rotates its
vertices at each frame via
:meth:`~physicskit.classical.systems.rotations.EulerTop.rotation_matrix`,
built from the integrated orientation quaternion. Spin it about the
intermediate axis and it periodically flips end over end; spin it
about either other principal axis and it just spins.
"""

# %%
import matplotlib.pyplot as plt

from physicskit.classical.systems.rotations import EulerTop
from physicskit.classical.visualizers.animations import animate_rigid_body_tumble

# Half-extents (a, b, c) along body axes (1, 2, 3): a > b > c makes
# I1 = b^2+c^2 (smallest) < I2 = a^2+c^2 < I3 = a^2+b^2 (largest),
# matching the EulerTop's I1=1, I2=2, I3=3 below.
HALF_EXTENTS = (1.5, 1.0, 0.4)

# %%
# Spin about the intermediate axis: it periodically flips end over end
# --------------------------------------------------------------------------

top = EulerTop(omega0=[0.01, 1.0, 0.01], I1=1.0, I2=2.0, I3=3.0)
result = top.integrate((0, 30.0), dt=1e-3, method="implicit_midpoint")

anim = animate_rigid_body_tumble(top, result, half_extents=HALF_EXTENTS, interval=50, stride=200)

plt.show()

# %%
# To save the animation to a file instead of (or in addition to) displaying
# it interactively, use e.g.::
#
#     anim.save("rigid_body_tumble_animation.gif", writer="pillow", fps=30)
