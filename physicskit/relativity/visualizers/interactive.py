"""Interactive Plotly visualizations.

Every other module in :mod:`physicskit.relativity.visualizers` produces static Matplotlib
figures. This module offers pan/zoom/rotate-enabled Plotly figures instead
-- convenient for interactively exploring a 3D orbit or a ray-traced image
in a Jupyter notebook, or for embedding in a standalone HTML page.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

__all__ = ["interactive_orbit_3d", "interactive_shadow_image"]


def interactive_orbit_3d(trajectory, M=1.0, horizon_radius=None):
    """Build an interactive 3D plot of an integrated geodesic trajectory.

    Parameters
    ----------
    trajectory : dict of str -> ndarray
        Output of ``SchwarzschildBlackHole.integrate_geodesic`` (keys
        ``"r"``, ``"theta"``, ``"phi"``) or a dict with equivalent
        equatorial ``"r"``, ``"phi"`` arrays (``"theta"`` then defaults to
        :math:`\\pi/2` throughout).
    M : float, default=1.0
        Black hole mass, used to draw the event horizon sphere.
    horizon_radius : float, optional
        Overrides the horizon radius (defaults to ``2*M``).

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> from physicskit.relativity.chapters.schwarzschild import SchwarzschildBlackHole
    >>> bh = SchwarzschildBlackHole(M=1.0)
    >>> y0 = bh.circular_orbit_initial_state(r0=10.0)
    >>> traj = bh.integrate_geodesic(y0, dtau=0.05, n_steps=200)
    >>> fig = interactive_orbit_3d(traj, M=1.0)
    >>> len(fig.data) > 0
    True
    """
    r = np.asarray(trajectory["r"], dtype=np.float64)
    theta = np.asarray(trajectory.get("theta", np.full_like(r, np.pi / 2.0)), dtype=np.float64)
    phi = np.asarray(trajectory["phi"], dtype=np.float64)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    if horizon_radius is None:
        horizon_radius = 2.0 * M

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line={"color": "steelblue", "width": 4}, name="geodesic"))

    u = np.linspace(0.0, 2.0 * np.pi, 40)
    v = np.linspace(0.0, np.pi, 20)
    hx = horizon_radius * np.outer(np.cos(u), np.sin(v))
    hy = horizon_radius * np.outer(np.sin(u), np.sin(v))
    hz = horizon_radius * np.outer(np.ones_like(u), np.cos(v))
    fig.add_trace(
        go.Surface(
            x=hx,
            y=hy,
            z=hz,
            showscale=False,
            colorscale=[[0, "black"], [1, "black"]],
            name="horizon",
        )
    )

    fig.update_layout(
        title="Geodesic trajectory",
        scene={
            "xaxis_title": "x [M]",
            "yaxis_title": "y [M]",
            "zaxis_title": "z [M]",
            "aspectmode": "data",
        },
        height=650,
    )
    return fig


def interactive_shadow_image(result):
    """Build an interactive heatmap of a ray-traced black hole image.

    Parameters
    ----------
    result : dict
        Output of ``physicskit.relativity.visualizers.shadow_render.render_black_hole_image``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    outcomes = result["outcomes"]
    hit_radii = result["hit_radii"]
    extent = result["extent"]
    ny, nx = outcomes.shape

    display = np.where(outcomes == 3, hit_radii, np.where(outcomes == 1, -1.0, np.nan))
    x = np.linspace(extent[0], extent[1], nx)
    y = np.linspace(extent[2], extent[3], ny)

    fig = go.Figure(
        data=go.Heatmap(
            z=display,
            x=x,
            y=y,
            colorscale="inferno",
            colorbar={"title": "disk radius [M]<br>(-1 = shadow)"},
            hoverongaps=False,
        )
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title="Black hole shadow and accretion disk (interactive)",
        xaxis_title="impact parameter x [M]",
        yaxis_title="impact parameter y [M]",
        height=650,
    )
    return fig
