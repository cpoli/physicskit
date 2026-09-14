"""Interactive 3D trajectory viewing via PyVista (the ``physicskit.chaos[viz3d]``
optional extra).

PyVista renders in a real, camera-navigable 3D scene, unlike Matplotlib's
projected-2D-with-fake-depth ``mplot3d`` axes -- most useful for genuinely
3D structures like the Lorenz/Rossler attractors, a double pendulum's
configuration-space trajectory, or a restricted three-body orbit.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from physicskit.chaos.visualizers import theme

try:
    import pyvista as pv
except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
    raise ImportError("physicskit.chaos.visualizers.viewer3d requires PyVista; install it with `pip install physicskit.chaos[viz3d]`.") from exc


def pyvista_trajectory(
    states: ArrayLike,
    components: tuple[int, int, int] = (0, 1, 2),
    cmap: str = theme.SEQUENTIAL_CMAP,
    line_width: float = 2.0,
    color_by: str = "index",
    off_screen: bool = False,
) -> pv.Plotter:
    """Render a 3D trajectory as an interactive PyVista scene.

    Parameters
    ----------
    states : array_like of float, shape (n, dim), dim >= 3
        Trajectory samples; three of each state's components are used as
        ``(x, y, z)``.
    components : tuple of int, default (0, 1, 2)
        Indices into each state's columns to use as ``(x, y, z)`` -- e.g.
        ``(0, 1, 2)`` selects the trajectory itself, but any three state
        components (or, for a system with more than three, any subset) can
        be chosen.
    cmap : str, default "viridis" (:data:`physicskit.chaos.visualizers.theme.SEQUENTIAL_CMAP`)
        Colormap the trajectory is colored by, per `color_by`.
    line_width : float, default 2.0
        Rendered line width, in screen pixels.
    color_by : {"index", "speed"}, default "index"
        Whether to color the trajectory by sample index (i.e. time) or by
        local speed (the Euclidean distance between consecutive selected
        points) -- the latter highlights where the trajectory moves fastest.
    off_screen : bool, default False
        Render off-screen (e.g. to save a screenshot from a script or in CI)
        instead of opening an interactive window.

    Returns
    -------
    pyvista.Plotter
        Call :meth:`~pyvista.Plotter.show` to display interactively, or
        :meth:`~pyvista.Plotter.screenshot` when `off_screen=True`.
    """
    states = np.asarray(states, dtype=np.float64)
    points = states[:, components]

    if color_by == "speed":
        deltas = np.diff(points, axis=0)
        scalars = np.concatenate([[0.0], np.linalg.norm(deltas, axis=1)])
    else:
        scalars = np.arange(points.shape[0], dtype=np.float64)

    line = pv.lines_from_points(points)
    line["scalars"] = scalars

    plotter = pv.Plotter(off_screen=off_screen)
    plotter.add_mesh(
        line,
        scalars="scalars",
        cmap=cmap,  # type: ignore[arg-type]  # pyvista stubs: cmap accepts any registered name
        line_width=line_width,
        show_scalar_bar=False,
    )
    plotter.show_grid()  # type: ignore[call-arg]  # pyvista stubs mistype this bound method
    return plotter
