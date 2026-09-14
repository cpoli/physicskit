"""Interactive Plotly visualizations.

Every other module in :mod:`physicskit.statphys.visualizers` produces static Matplotlib
figures, well suited to publication and to this documentation's example
gallery. This module offers the same information as pan/zoom/hover-enabled
Plotly figures instead -- convenient for interactively exploring a parameter
sweep or a dense vector field in a Jupyter notebook, or for embedding in a
standalone HTML page.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

__all__ = [
    "interactive_particle_snapshot",
    "interactive_temperature_sweep",
    "interactive_vortex_field",
]


def interactive_temperature_sweep(result, T_c=None):
    """Build an interactive 2x2 subplot figure of a temperature sweep result.

    Parameters
    ----------
    result : dict of str -> ndarray
        Output of e.g. ``Ising2D.run_temperature_sweep`` or
        ``PottsModel2D.run_temperature_sweep``.
    T_c : float, optional
        Critical temperature to mark with a vertical line on every panel.

    Returns
    -------
    plotly.graph_objects.Figure
        A 2x2 subplot figure with hoverable energy, order-parameter,
        specific-heat, and susceptibility traces.

    Examples
    --------
    >>> from physicskit.statphys.chapters.ising_lattice import Ising2D
    >>> model = Ising2D(L=10, seed=0)
    >>> result = model.run_temperature_sweep([2.0, 2.269, 2.5], n_equil=10, n_measure=10)
    >>> fig = interactive_temperature_sweep(result, T_c=model.T_C)
    >>> len(fig.data) > 0
    True
    """
    T = result["T"]
    panels = [
        ("E", "Energy per site"),
        ("M" if "M" in result else "m", "Order parameter"),
        ("C_v", "Specific heat"),
        ("chi", "Susceptibility"),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[label for _, label in panels])
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for (key, label), (row, col) in zip(panels, positions):
        if key not in result:
            continue
        fig.add_trace(go.Scatter(x=T, y=result[key], mode="lines+markers", name=label), row=row, col=col)
        if T_c is not None:
            fig.add_vline(x=T_c, line_dash="dash", line_color="black", row=row, col=col)
    fig.update_layout(title="Temperature sweep", showlegend=False, height=600)
    return fig


def interactive_vortex_field(theta, vorticity=None, step=1):
    """Build an interactive XY spin field plot with vortex/antivortex markers.

    Parameters
    ----------
    theta : ndarray of shape (L, L)
        XY model spin angles, e.g. ``XYModel2D.theta``.
    vorticity : ndarray of shape (L, L), optional
        Precomputed plaquette vorticity; computed from ``theta`` if omitted.
    step : int, default=1
        Subsampling factor for the arrow field.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if vorticity is None:
        from physicskit.statphys.core.monte_carlo import xy_plaquette_vorticity

        vorticity = xy_plaquette_vorticity(theta)

    L = theta.shape[0]
    idx = np.arange(0, L, step)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    sub_theta = theta[np.ix_(idx, idx)]

    scale = 0.4 * step
    xs, ys = [], []
    for a, b, t in zip(ii.ravel(), jj.ravel(), sub_theta.ravel()):
        xs += [float(b), float(b) + scale * np.cos(t), None]
        ys += [float(a), float(a) + scale * np.sin(t), None]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": "steelblue"},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    vi, vj = np.nonzero(vorticity > 0.5)
    ai, aj = np.nonzero(vorticity < -0.5)
    fig.add_trace(go.Scatter(x=vj, y=vi, mode="markers", marker={"color": "red", "size": 10}, name="vortex"))
    fig.add_trace(
        go.Scatter(
            x=aj,
            y=ai,
            mode="markers",
            marker={"color": "blue", "size": 10, "symbol": "x"},
            name="antivortex",
        )
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    fig.update_layout(title="XY spin field and vortices", height=600)
    return fig


def interactive_particle_snapshot(gas):
    """Build an interactive scatter plot of a Lennard-Jones gas snapshot, colored by speed.

    Parameters
    ----------
    gas : LennardJonesGas
        Simulation instance to render.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    speeds = gas.speeds()
    fig = go.Figure(
        data=go.Scatter(
            x=gas.positions[:, 0],
            y=gas.positions[:, 1],
            mode="markers",
            marker={
                "color": speeds,
                "colorscale": "Plasma",
                "showscale": True,
                "size": 8,
                "colorbar": {"title": "speed"},
            },
            text=[f"speed={s:.2f}" for s in speeds],
            hoverinfo="text",
        )
    )
    fig.update_xaxes(range=[0, gas.box_size])
    fig.update_yaxes(range=[0, gas.box_size], scaleanchor="x", scaleratio=1)
    fig.update_layout(title=f"Lennard-Jones gas, t={gas.time:.2f}", height=600)
    return fig
