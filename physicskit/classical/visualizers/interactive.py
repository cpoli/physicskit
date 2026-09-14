"""Interactive Plotly visualizers.

The Matplotlib-based plots in :mod:`physicskit.classical.visualizers.phase_space` are
static and best suited to documentation/reports. For exploratory work
-- rotating a rigid body's momentum trajectory around to see how it
sits on the Casimir sphere, or zooming into a precessing orbit's
perihelion -- an interactive, mouse-drag-and-zoom view is much more
useful, which is exactly what Plotly is for.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

__all__ = ["interactive_so3_momentum_sphere", "interactive_orbit"]


def interactive_so3_momentum_sphere(omega_trajectory: np.ndarray, I1: float, I2: float, I3: float, n_grid: int = 40, title: str = None) -> go.Figure:
    """Interactive 3D Plotly view of a rigid body's body-frame angular
    momentum L(t) = (I1*w1, I2*w2, I3*w3) on the Casimir sphere
    ``|L| = |L(0)|``, draggable/zoomable in a way the static Matplotlib
    version (:func:`physicskit.classical.visualizers.phase_space.plot_so3_momentum_sphere`)
    cannot be -- useful for directly inspecting how close a trajectory
    passes to the unstable intermediate-axis pole.

    Parameters
    ----------
    omega_trajectory : array, shape (n_steps, 3)
        Body-frame angular velocity trajectory, e.g.
        ``result.y[:, :3]`` from integrating an
        :class:`~physicskit.classical.systems.rotations.EulerTop`.
    I1, I2, I3 : float
        Principal moments of inertia.
    n_grid : int
        Resolution of the wireframe sphere.
    title : str, optional
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    omega_trajectory = np.asarray(omega_trajectory)
    L = np.column_stack([I1 * omega_trajectory[:, 0], I2 * omega_trajectory[:, 1], I3 * omega_trajectory[:, 2]])
    radius = np.linalg.norm(L[0])

    u = np.linspace(0, 2 * np.pi, n_grid)
    v = np.linspace(0, np.pi, n_grid)
    xs = radius * np.outer(np.cos(u), np.sin(v))
    ys = radius * np.outer(np.sin(u), np.sin(v))
    zs = radius * np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()
    fig.add_surface(x=xs, y=ys, z=zs, opacity=0.25, showscale=False, colorscale=[[0, "lightgray"], [1, "lightgray"]], name="Casimir sphere")
    fig.add_scatter3d(x=L[:, 0], y=L[:, 1], z=L[:, 2], mode="lines", line=dict(color="crimson", width=4), name="L(t)")
    fig.add_scatter3d(x=[L[0, 0]], y=[L[0, 1]], z=[L[0, 2]], mode="markers", marker=dict(color="green", size=5), name="start")
    fig.update_layout(
        title=title or "Body-frame angular momentum on the Casimir sphere",
        scene=dict(xaxis_title="L1", yaxis_title="L2", zaxis_title="L3", aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def interactive_orbit(result, system=None, title: str = None) -> go.Figure:
    """Interactive 2D Plotly view of a planar central-force orbit (e.g.
    from :class:`~physicskit.classical.systems.newtonian.KeplerSystem`), with the
    Laplace-Runge-Lenz vector overlaid at start and end if ``system``
    exposes ``lrl_vector`` -- draggable/zoomable so a slow apsidal
    precession is easy to inspect directly, unlike a static plot.

    Parameters
    ----------
    result : physicskit.classical.core.base_system.SimulationResult
        Output of ``system.integrate(...)``.
    system : object, optional
        The system the result came from; if it defines
        ``lrl_vector(q, p) -> array``, the LRL direction at the start
        and end of the trajectory is drawn as an arrow from the origin.
    title : str, optional
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    q = result.q
    fig = go.Figure()
    fig.add_scatter(x=q[:, 0], y=q[:, 1], mode="lines", line=dict(color="steelblue", width=1.5), name="orbit")
    fig.add_scatter(x=[0], y=[0], mode="markers", marker=dict(color="gold", size=12), name="focus")

    if system is not None and hasattr(system, "lrl_vector"):
        for idx, (label, color) in ((0, ("LRL(0)", "green")), (-1, ("LRL(end)", "crimson"))):
            vec = system.lrl_vector(q[idx], result.p[idx])
            scale = np.max(np.abs(q)) / (np.linalg.norm(vec) + 1e-300) * 0.6
            fig.add_scatter(
                x=[0, vec[0] * scale],
                y=[0, vec[1] * scale],
                mode="lines+markers",
                line=dict(color=color, width=2, dash="dash"),
                marker=dict(size=[0, 6]),
                name=label,
            )

    fig.update_layout(
        title=title or "Orbit",
        xaxis_title="x",
        yaxis_title="y",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
