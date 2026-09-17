r"""
Interactive 3D Viewing with PyVista
=========================================

Matplotlib's ``mplot3d`` axes project a 3D scene onto a 2D canvas with only
an approximation of real depth (and no interactive rotation once a figure is
saved as a static image). :func:`physicskit.chaos.visualizers.viewer3d.pyvista_trajectory`
renders the same kind of trajectory -- here, the Rossler attractor,

.. math::

    \dot{x} = -y - z, \qquad \dot{y} = x + a y, \qquad
    \dot{z} = b + z (x - c),

with the classic chaotic parameters :math:`a=0.2`, :math:`b=0.2`,
:math:`c=5.7` -- in a real, camera-navigable 3D scene via
`PyVista <https://pyvista.org>`_ (the optional ``physicskit.chaos[viz3d]``
extra). Call ``.show()`` on the returned plotter for an interactive window;
this example instead renders off-screen and displays the result as a static
image, purely so it can be captured for this gallery. To make the
"only an approximation of real depth" claim concrete rather than asserted,
the same trajectory is also drawn with a plain Matplotlib ``mplot3d`` axes
side by side with the PyVista render, so the two can be compared directly on
the same attractor.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from physicskit.chaos.systems.continuous import Rossler

try:
    from physicskit.chaos.visualizers.viewer3d import pyvista_trajectory

    _HAS_PYVISTA = True
except ImportError:
    _HAS_PYVISTA = False

# %%
# Render the Rossler attractor
# ---------------------------------
# Colored by sample index (time), so the direction of travel around the
# attractor's two lobes is visible at a glance.
system = Rossler()
_, states = system.trajectory(n_steps=20000, dt=0.02)

# %%
# Side by side: projected depth vs. a real 3D scene
# -------------------------------------------------------
# Left: the same attractor drawn with a plain ``mplot3d`` axes -- a single
# fixed camera angle baked in at draw time, with depth conveyed only by
# perspective and occlusion. Right: the PyVista render, which (interactively)
# can be freely rotated to disentangle the two lobes from any angle; here it
# is only a static screenshot, but even that single frame already shows a
# cleaner separation of near/far strands than the ``mplot3d`` panel, thanks
# to PyVista's real depth-buffered rendering.
fig = plt.figure(figsize=(13, 6))
ax_mpl = fig.add_subplot(1, 2, 1, projection="3d")
ax_pv = fig.add_subplot(1, 2, 2)

ax_mpl.plot(states[:, 0], states[:, 1], states[:, 2], lw=0.3, color="teal")
ax_mpl.set_xlabel("x")
ax_mpl.set_ylabel("y")
ax_mpl.set_zlabel("z")
ax_mpl.set_title("Matplotlib mplot3d\n(projected depth, fixed camera)")

if _HAS_PYVISTA:
    plotter = pyvista_trajectory(states, color_by="index", off_screen=True)
    plotter.camera.zoom(1.3)
    screenshot = plotter.screenshot()
    plotter.close()

    ax_pv.imshow(screenshot)
    ax_pv.axis("off")
    ax_pv.set_title("PyVista\n(real 3D scene, off-screen render for this gallery)")
else:
    ax_pv.text(
        0.5,
        0.5,
        "PyVista is not installed\n(pip install physicskit.chaos[viz3d])\n-- skipping the 3D render for this gallery build.",
        ha="center",
        va="center",
        transform=ax_pv.transAxes,
    )
    ax_pv.axis("off")
    print("PyVista is not installed (pip install physicskit.chaos[viz3d]) -- skipping the 3D render for this gallery build.")

fig.suptitle("Rossler attractor: projected (mplot3d) vs. real (PyVista) 3D depth")
fig.tight_layout()

plt.show()
