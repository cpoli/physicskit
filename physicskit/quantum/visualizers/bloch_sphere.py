r"""Interactive 3D Bloch-sphere qubit trajectories (Plotly), and a frame-by-frame
matplotlib animation of the vector advancing along a trajectory."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation

from ..core.operators import sigma_x, sigma_y, sigma_z

__all__ = ["bloch_vector", "state_to_bloch_trajectory", "plot_bloch_sphere", "animate_bloch_sphere"]


def bloch_vector(psi: np.ndarray) -> np.ndarray:
    r"""The Bloch-sphere coordinates of a single-qubit pure state.

    .. math::

        \mathbf n = \left(\langle\psi\rvert\sigma_x\lvert\psi\rangle,\;
                           \langle\psi\rvert\sigma_y\lvert\psi\rangle,\;
                           \langle\psi\rvert\sigma_z\lvert\psi\rangle\right).

    Parameters
    ----------
    psi : numpy.ndarray
        A 2-component qubit state vector.

    Returns
    -------
    numpy.ndarray
        The 3-component Bloch vector ``(x, y, z)``.
    """
    psi = np.asarray(psi, dtype=complex)
    x = np.real(np.vdot(psi, sigma_x @ psi))
    y = np.real(np.vdot(psi, sigma_y @ psi))
    z = np.real(np.vdot(psi, sigma_z @ psi))
    return np.array([x, y, z])


def state_to_bloch_trajectory(states: np.ndarray) -> np.ndarray:
    """Convert a time series of qubit states to Bloch-vector coordinates.

    Parameters
    ----------
    states : numpy.ndarray
        Qubit state vectors over time, shape ``(n_times, 2)``.

    Returns
    -------
    numpy.ndarray
        Bloch vectors, shape ``(n_times, 3)``.
    """
    return np.array([bloch_vector(psi) for psi in states])


def _sphere_mesh(n: int = 40):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    return xs, ys, zs


def plot_bloch_sphere(trajectory: np.ndarray | None = None, vectors: list[np.ndarray] | None = None, labels: list[str] | None = None) -> go.Figure:
    """Render the Bloch sphere.

    Parameters
    ----------
    trajectory : numpy.ndarray or None, optional
        Bloch vectors traced over time, shape ``(n_times, 3)`` (e.g. from
        :func:`state_to_bloch_trajectory`).
    vectors : list of numpy.ndarray or None, optional
        Static vectors to mark, each a 3-component array.
    labels : list of str or None, optional
        Labels for ``vectors``; defaults to ``v0, v1, ...``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    xs, ys, zs = _sphere_mesh()
    fig = go.Figure()
    fig.add_trace(go.Surface(x=xs, y=ys, z=zs, opacity=0.15, showscale=False, colorscale=[[0, "lightblue"], [1, "lightblue"]]))

    axis_len = 1.3
    for axis, name in zip(np.eye(3), ["x", "y", "z"]):
        fig.add_trace(
            go.Scatter3d(
                x=[-axis_len * axis[0], axis_len * axis[0]],
                y=[-axis_len * axis[1], axis_len * axis[1]],
                z=[-axis_len * axis[2], axis_len * axis[2]],
                mode="lines",
                line=dict(color="gray", width=2),
                showlegend=False,
            )
        )
        fig.add_trace(go.Scatter3d(x=[axis_len * axis[0]], y=[axis_len * axis[1]], z=[axis_len * axis[2]], mode="text", text=[name], showlegend=False))

    if trajectory is not None:
        fig.add_trace(
            go.Scatter3d(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=trajectory[:, 2],
                mode="lines+markers",
                marker=dict(size=3, color=np.arange(len(trajectory)), colorscale="Viridis"),
                line=dict(color="darkred", width=4),
                name="trajectory",
            )
        )

    if vectors is not None:
        for i, v in enumerate(vectors):
            label = labels[i] if labels else f"v{i}"
            fig.add_trace(
                go.Scatter3d(
                    x=[0, v[0]],
                    y=[0, v[1]],
                    z=[0, v[2]],
                    mode="lines+markers",
                    line=dict(width=6),
                    marker=dict(size=[0, 6]),
                    name=label,
                )
            )

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1.4, 1.4]),
            yaxis=dict(range=[-1.4, 1.4]),
            zaxis=dict(range=[-1.4, 1.4]),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title="Bloch sphere",
    )
    return fig


def animate_bloch_sphere(trajectory: np.ndarray, times: np.ndarray | None = None, interval: int = 30, trail: bool = True, ax=None) -> FuncAnimation:
    """A matplotlib mplot3d ``FuncAnimation`` of the Bloch vector advancing along a trajectory.

    Unlike :func:`plot_bloch_sphere` (which draws the whole path as one
    static Plotly trace), this genuinely animates: the arrow from the
    origin sweeps frame-by-frame along ``trajectory``, optionally leaving a
    fading trail.

    Parameters
    ----------
    trajectory : numpy.ndarray
        Bloch vectors over time, shape ``(n_frames, 3)`` (e.g. from
        :func:`state_to_bloch_trajectory`).
    times : numpy.ndarray or None, optional
        Time of each frame, used to label the animation title.
    interval : int, default=30
        Delay between frames, in milliseconds.
    trail : bool, default=True
        Whether to draw the path swept out so far.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D or None, optional
        Axis to draw on; a new 3D figure/axis is created if omitted.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="lightblue", alpha=0.2, linewidth=0.5)

    axis_len = 1.3
    for axis in np.eye(3):
        ax.plot(*zip(-axis_len * axis, axis_len * axis), color="gray", lw=1)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(-1.3, 1.3)
    ax.set_box_aspect((1, 1, 1))

    (arrow_line,) = ax.plot([0, trajectory[0, 0]], [0, trajectory[0, 1]], [0, trajectory[0, 2]], color="darkred", lw=3, marker="o", markevery=[1])
    (trail_line,) = ax.plot([], [], [], color="darkorange", lw=1.5, alpha=0.7)

    def update(i):
        vec = trajectory[i]
        arrow_line.set_data([0, vec[0]], [0, vec[1]])
        arrow_line.set_3d_properties([0, vec[2]])
        if trail:
            trail_line.set_data(trajectory[: i + 1, 0], trajectory[: i + 1, 1])
            trail_line.set_3d_properties(trajectory[: i + 1, 2])
        title = f"t = {times[i]:.3f}" if times is not None else f"frame {i}"
        ax.set_title(title)
        return arrow_line, trail_line

    return FuncAnimation(fig, update, frames=len(trajectory), interval=interval, blit=False)
