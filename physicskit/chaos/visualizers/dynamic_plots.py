"""Matplotlib animation helpers, trajectory color-coding, and interactive Plotly figures."""

from __future__ import annotations

from typing import Any, Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from physicskit.chaos.core.base_system import BilliardSystem, DiscreteMap
from physicskit.chaos.systems.continuous import DoublePendulum, DrivenPendulum, MagneticPendulum, RestrictedThreeBody, lagrange_points
from physicskit.chaos.systems.maps import BakersMap
from physicskit.chaos.visualizers import theme


class _TrajectorySystem(Protocol):
    """Structural type for a :class:`~physicskit.chaos.core.base_system.DynamicalSystem`
    that also implements its own ``trajectory`` method (every concrete
    continuous system in physicskit.chaos does, but the base class itself does not
    declare it, so it isn't part of ``DynamicalSystem``'s static type)."""

    dim: int

    def trajectory(
        self,
        state0: NDArray[np.float64] | None = ...,
        t0: float = ...,
        dt: float = ...,
        n_steps: int = ...,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


def animate_billiard_trajectory(
    billiard: BilliardSystem,
    pos: ArrayLike,
    vel: ArrayLike,
    n_bounces: int = 100,
    interval: int = 50,
    trail: int = 200,
) -> FuncAnimation:
    """Animate a trajectory bouncing inside a billiard's boundary.

    Draws the real-space trajectory (left panel) alongside its
    live-updating Poincare section ``(s, sin phi)`` (right panel).

    Parameters
    ----------
    billiard : BilliardSystem
        Billiard to animate.
    pos : array_like of float, shape (2,)
        Initial position ``(x, y)``; must lie in the billiard's interior.
    vel : array_like of float, shape (2,)
        Initial velocity direction ``(vx, vy)``; normalized internally.
    n_bounces : int, default 100
        Number of reflections to trace and animate.
    interval : int, default 50
        Delay between animation frames, in milliseconds.
    trail : int, default 200
        Number of most-recent trajectory points to keep visible as a trail.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object; assign it to a variable to keep it alive, and
        display it with ``plt.show()`` or save it with ``anim.save(...)``.
    """
    boundary = billiard.boundary_polyline()
    path, result = billiard.trajectory_segments(pos, vel, n_bounces)
    s, sin_phi = result["s"], result["sin_phi"]

    fig, (ax_space, ax_poincare) = plt.subplots(1, 2, figsize=(12, 5.5))

    ax_space.plot(boundary[:, 0], boundary[:, 1], color=theme.STRUCTURE, lw=1.5)
    ax_space.set_aspect("equal")
    ax_space.set_title(f"{billiard.__class__.__name__} trajectory")
    (trail_line,) = ax_space.plot([], [], color=theme.ACCENT, lw=0.8)
    (ball,) = ax_space.plot([], [], "o", color=theme.ACCENT, markersize=5)

    ax_poincare.set_xlim(0.0, billiard.perimeter())
    ax_poincare.set_ylim(-1.0, 1.0)
    ax_poincare.set_xlabel("s")
    ax_poincare.set_ylabel(r"$\sin\varphi$")
    ax_poincare.set_title("Poincare section (live)")
    poincare_scatter = ax_poincare.scatter([], [], s=6, color=theme.PRIMARY)

    def init() -> tuple:
        trail_line.set_data([], [])
        ball.set_data([], [])
        poincare_scatter.set_offsets(np.empty((0, 2)))
        return trail_line, ball, poincare_scatter

    def update(frame: int) -> tuple:
        lo = max(0, frame - trail)
        trail_line.set_data(path[lo : frame + 1, 0], path[lo : frame + 1, 1])
        ball.set_data([path[frame, 0]], [path[frame, 1]])
        n_hits = max(0, frame)
        poincare_scatter.set_offsets(np.column_stack([s[:n_hits], sin_phi[:n_hits]]))
        return trail_line, ball, poincare_scatter

    anim = FuncAnimation(fig, update, frames=path.shape[0], init_func=init, interval=interval, blit=False)
    return anim


def animate_billiard_divergence(
    billiard: BilliardSystem,
    pos: ArrayLike,
    vel: ArrayLike,
    delta_0: float = 1e-8,
    n_bounces: int = 100,
    interval: int = 50,
    trail: int = 200,
) -> FuncAnimation:
    """Animate two initially-nearby billiard rays bouncing side by side, diverging.

    Both rays start at the same position and are launched at angles
    `delta_0` radians apart; watch them bounce together, indistinguishably,
    until the perturbation is amplified enough (a single bounce off a
    defocusing or truncated wall can be all it takes) to send them down
    completely different paths -- sensitive dependence on initial
    conditions, seen directly rather than only in a divergence plot.

    Parameters
    ----------
    billiard : BilliardSystem
        Billiard to animate both rays within.
    pos : array_like of float, shape (2,)
        Shared initial position ``(x, y)``; must lie in the billiard's
        interior.
    vel : array_like of float, shape (2,)
        Initial velocity direction ``(vx, vy)`` of the reference ray;
        normalized internally.
    delta_0 : float, default 1e-8
        Initial angular perturbation (radians) applied to `vel` to launch
        the second ray.
    n_bounces : int, default 100
        Number of reflections to trace and animate for each ray.
    interval : int, default 50
        Delay between animation frames, in milliseconds.
    trail : int, default 200
        Number of most-recent trajectory points to keep visible as a trail,
        for each ray.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object; assign it to a variable to keep it alive, and
        display it with ``plt.show()`` or save it with ``anim.save(...)``.
    """
    vel_ref = np.asarray(vel, dtype=np.float64)
    vel_ref = vel_ref / np.linalg.norm(vel_ref)
    angle = float(np.arctan2(vel_ref[1], vel_ref[0])) + delta_0
    vel_perturbed = np.array([np.cos(angle), np.sin(angle)])

    boundary = billiard.boundary_polyline()
    path_ref, _ = billiard.trajectory_segments(pos, vel_ref, n_bounces)
    path_pert, _ = billiard.trajectory_segments(pos, vel_perturbed, n_bounces)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(boundary[:, 0], boundary[:, 1], color=theme.STRUCTURE, lw=1.5)
    ax.set_aspect("equal")
    ax.set_title(f"{billiard.__class__.__name__}: two rays {delta_0:.0e} rad apart")

    (trail_ref,) = ax.plot([], [], color=theme.PRIMARY, lw=0.8)
    (ball_ref,) = ax.plot([], [], "o", color=theme.PRIMARY, markersize=5, label="reference")
    (trail_pert,) = ax.plot([], [], color=theme.ACCENT, lw=0.8)
    (ball_pert,) = ax.plot([], [], "o", color=theme.ACCENT, markersize=5, label="perturbed")
    ax.legend(loc="upper right")

    def init() -> tuple:
        for artist in (trail_ref, ball_ref, trail_pert, ball_pert):
            artist.set_data([], [])
        return trail_ref, ball_ref, trail_pert, ball_pert

    def update(frame: int) -> tuple:
        lo = max(0, frame - trail)
        trail_ref.set_data(path_ref[lo : frame + 1, 0], path_ref[lo : frame + 1, 1])
        ball_ref.set_data([path_ref[frame, 0]], [path_ref[frame, 1]])
        trail_pert.set_data(path_pert[lo : frame + 1, 0], path_pert[lo : frame + 1, 1])
        ball_pert.set_data([path_pert[frame, 0]], [path_pert[frame, 1]])
        return trail_ref, ball_ref, trail_pert, ball_pert

    anim = FuncAnimation(fig, update, frames=path_ref.shape[0], init_func=init, interval=interval, blit=False)
    return anim


def plotly_billiard_trajectory(billiard: BilliardSystem, pos: ArrayLike, vel: ArrayLike, n_bounces: int = 100) -> go.Figure:
    """Build an interactive Plotly figure of a trajectory inside a billiard.

    Parameters
    ----------
    billiard : BilliardSystem
        Billiard to trace the trajectory within.
    pos : array_like of float, shape (2,)
        Initial position ``(x, y)``; must lie in the billiard's interior.
    vel : array_like of float, shape (2,)
        Initial velocity direction ``(vx, vy)``; normalized internally.
    n_bounces : int, default 100
        Number of reflections to trace.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive figure with the boundary and trajectory as line traces.
    """
    boundary = billiard.boundary_polyline()
    path, _ = billiard.trajectory_segments(pos, vel, n_bounces)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=boundary[:, 0],
            y=boundary[:, 1],
            mode="lines",
            line={"color": theme.STRUCTURE, "width": 2},
            name="boundary",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=path[:, 0],
            y=path[:, 1],
            mode="lines",
            line={"color": theme.ACCENT, "width": 1},
            name="trajectory",
        )
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=f"{billiard.__class__.__name__} trajectory",
        xaxis_title="x",
        yaxis_title="y",
    )
    return fig


def plotly_poincare_section(billiard: BilliardSystem, n_rays: int = 50, n_bounces: int = 200, seed: int | None = None) -> go.Figure:
    """Build an interactive Plotly scatter of a billiard's boundary phase space.

    Parameters
    ----------
    billiard : BilliardSystem
        Billiard to sample.
    n_rays : int, default 50
        Number of independent trajectories to launch.
    n_bounces : int, default 200
        Number of reflections to trace per trajectory.
    seed : int, optional
        Seed for the random angle generator, for reproducibility.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive scatter figure of ``(s, sin phi)`` boundary hits.
    """
    rng = np.random.default_rng(seed)
    interior = billiard.sample_interior_point()
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_rays)
    result = billiard.simulate_many_rays(interior, angles, n_bounces)
    s, sin_phi = result["s"], result["sin_phi"]

    fig = go.Figure(
        go.Scattergl(
            x=s,
            y=sin_phi,
            mode="markers",
            marker={"size": 3, "color": theme.PRIMARY, "opacity": 0.6},
        )
    )
    fig.update_layout(
        title=f"{billiard.__class__.__name__} Poincare section",
        xaxis_title="s (boundary arclength)",
        yaxis_title="sin(phi)",
        yaxis_range=[-1.0, 1.0],
        xaxis_range=[0.0, billiard.perimeter()],
    )
    return fig


def plot_colored_trajectory(
    x: ArrayLike,
    y: ArrayLike,
    values: ArrayLike,
    ax: Axes | None = None,
    cmap: str = theme.SEQUENTIAL_CMAP,
    colorbar: bool = True,
    colorbar_label: str = "",
    **line_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a 2D path colored segment-by-segment by an arbitrary scalar.

    Useful for highlighting where a trajectory is "interesting" -- e.g. a
    billiard trajectory colored by bounce index (to see how quickly it
    explores the table), or a continuous trajectory colored by local speed or
    an instantaneous divergence-rate estimate.

    Parameters
    ----------
    x, y : array_like of float, shape (n,)
        Path coordinates.
    values : array_like of float, shape (n,)
        Scalar value at each point, used to color the path; each line
        segment between consecutive points is colored by the value at its
        starting point.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure and axes are created if omitted.
    cmap : str, default "viridis"
        Colormap name.
    colorbar : bool, default True
        Whether to add a colorbar.
    colorbar_label : str, default ""
        Label for the colorbar.
    **line_kwargs
        Additional keyword arguments forwarded to
        ``matplotlib.collections.LineCollection`` (e.g. ``linewidth``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = cast(Figure, ax.figure)

    kwargs: dict[str, Any] = {"linewidth": 1.5}
    kwargs.update(line_kwargs)
    line_collection = LineCollection(cast(Any, segments), cmap=cmap, **kwargs)
    line_collection.set_array(values[:-1])
    ax.add_collection(line_collection)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    if colorbar:
        fig.colorbar(line_collection, ax=ax, label=colorbar_label)

    return fig, ax


def plotly_3d_trajectory(
    states: ArrayLike,
    labels: tuple[str, str, str] = ("x", "y", "z"),
    color_by: ArrayLike | None = None,
    colorbar_label: str = "",
    title: str | None = None,
) -> go.Figure:
    """Build an interactive, orbit/pan/zoom-able 3D Plotly trajectory figure.

    Parameters
    ----------
    states : array_like of float, shape (n, 3)
        3D trajectory points (e.g. Lorenz/Rossler/Chua state history).
    labels : tuple of str, default ("x", "y", "z")
        Axis labels.
    color_by : array_like of float, shape (n,), optional
        Scalar value at each point used to color the trajectory (e.g. local
        speed, or time); a single solid color is used if omitted.
    colorbar_label : str, default ""
        Label for the colorbar, if `color_by` is given.
    title : str, optional
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    states = np.asarray(states, dtype=np.float64)

    if color_by is None:
        line = {"color": theme.PRIMARY, "width": 2}
    else:
        line = {
            "color": np.asarray(color_by, dtype=np.float64),
            "width": 3,
            "colorscale": "Viridis",
            "colorbar": {"title": colorbar_label},
        }

    fig = go.Figure(
        go.Scatter3d(
            x=states[:, 0],
            y=states[:, 1],
            z=states[:, 2],
            mode="lines",
            line=line,
        )
    )
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": labels[0],
            "yaxis_title": labels[1],
            "zaxis_title": labels[2],
        },
    )
    return fig


def animate_bakers_map(
    system: BakersMap,
    n_points: int = 10000,
    n_iterations: int = 20,
    frames_per_iteration: int = 15,
    interval: int = 60,
    stripe_colors: tuple[str, str] = (theme.PRIMARY, theme.ACCENT),
    seed: int | None = 0,
) -> FuncAnimation:
    """Animate the baker's map's "stretch, cut, and stack" action, in three
    explicit phases per iteration.

    Starts from a regular grid of square markers, colored in exactly two
    vertical bands (left half / right half) -- which, for the default
    ``alpha = 0.5``, lines up with the map's own cut point, so the first
    cut slices exactly along the original color boundary. Each iteration is
    then animated as three distinct, visually separated phases rather than
    one smooth blend, so the mechanism itself is legible:

    1. **Stretch** -- the square is squashed vertically by a factor
       ``alpha`` and stretched horizontally onto ``[0, 2)``: the left branch
       (``x < alpha``) stretches onto ``[0, 1)`` in place, and the right
       branch (``x >= alpha``) stretches onto its own ``[0, 1)`` slot, shown
       offset to ``[1, 2)`` so the two pieces don't overlap mid-flight.
    2. **Cut** -- a brief pause with a dashed guide line at ``x = 1``
       marking where the stretched strip is about to be sliced in two.
    3. **Stack** -- the right-hand piece slides left by 1 and up onto
       ``y in [alpha, 1)``, landing exactly on top of the left piece, which
       has not moved since the stretch phase. The square is whole again, now
       twice as finely interleaved.

    Because every marker keeps its original color, the two bands are seen
    getting sliced and restacked into progressively thinner, more numerous
    stripes -- and because the whole grid is present from frame 0, the
    increasing fineness of that interleaving (the mixing) is directly
    visible, iteration after iteration.

    Parameters
    ----------
    system : BakersMap
        Map to animate.
    n_points : int, default 10000
        Approximate number of square markers: laid out on a regular
        ``round(sqrt(n_points))``-by-``round(sqrt(n_points))`` grid so they
        tile the unit square with no gaps at the start.
    n_iterations : int, default 20
        Number of map iterations to animate through.
    frames_per_iteration : int, default 15
        Number of frames used to animate each iteration's stretch, cut, and
        stack phases combined.
    interval : int, default 60
        Delay between animation frames, in milliseconds.
    stripe_colors : tuple of str, default (theme.PRIMARY, theme.ACCENT)
        Colors for the left half and right half of the initial grid,
        respectively.
    seed : int, optional
        Seed for a small random jitter applied to the grid positions. A
        perfectly regular grid sits at exact dyadic rationals, which this
        map's doubling action on the ``x`` coordinate degenerates in very
        few iterations (the same finite-precision collapse documented on
        :class:`~physicskit.chaos.systems.maps.BakersMap`); a tiny jitter, much
        smaller than the grid spacing, avoids that without visibly
        disturbing the tiling.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Assign it to a variable to keep it alive, and display it with
        ``plt.show()`` or save it with ``anim.save(...)``.
    """
    alpha = system.alpha
    n_side = max(2, round(np.sqrt(n_points)))
    centers = (np.arange(n_side) + 0.5) / n_side
    grid_x, grid_y = np.meshgrid(centers, centers)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.15, 0.15, size=(2,) + grid_x.shape) / n_side
    x0 = np.clip(grid_x + jitter[0], 0.0, 1.0 - 1e-12).ravel()
    y0 = np.clip(grid_y + jitter[1], 0.0, 1.0 - 1e-12).ravel()

    colors = np.where(x0 < 0.5, stripe_colors[0], stripe_colors[1])

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.add_patch(plt.Rectangle((0.0, 0.0), 1.0, 1.0, fill=False, edgecolor=theme.STRUCTURE, lw=1.2, zorder=3))
    cut_line = ax.axvline(1.0, color=theme.STRUCTURE, ls="--", lw=1.2, zorder=3, visible=False)
    title = ax.set_title("Baker's map: stretch (iteration 0 -> 1)")

    # Marker size is chosen once, from the (fixed) data-to-pixel scale, so
    # that markers exactly tile the resting unit square without gaps or
    # overlaps -- `scatter`'s `s` is in fixed points^2, not data units, so
    # this only works because the axis limits never change during the
    # animation (a zooming camera would desync marker size from grid spacing).
    fig.canvas.draw()
    p0 = ax.transData.transform((0.0, 0.0))
    p1 = ax.transData.transform((1.0 / n_side, 0.0))
    marker_points = (p1[0] - p0[0]) * 72.0 / fig.dpi
    scatter = ax.scatter(x0, y0, s=marker_points**2, c=colors, marker="s", linewidths=0, zorder=2)

    # Frame budget within each iteration: a stretch phase, a single-frame
    # pause on the cut, and a stack phase.
    n_stretch = max(1, round(0.45 * frames_per_iteration))
    n_pause = 1 if frames_per_iteration >= 3 else 0
    n_stack = max(1, frames_per_iteration - n_stretch - n_pause)
    frames_per_iteration = n_stretch + n_pause + n_stack
    total_frames = n_iterations * frames_per_iteration

    # `state` holds the grid's *resting* (post-stack) positions, carried over
    # between iterations; the phase-1 and final targets for the iteration in
    # progress are derived from it once, at that iteration's first sub-frame,
    # and held fixed while sub-frames interpolate between them.
    state: dict[str, Any] = {"x": x0.copy(), "y": y0.copy()}

    def update(frame: int) -> tuple:
        it, sub = divmod(frame, frames_per_iteration)
        if sub == 0:
            rx, ry = state["x"], state["y"]
            left = rx < alpha
            # Stretch target: left branch lands on its true final position
            # directly; right branch stretches onto its own [0, 1) slot,
            # offset to [1, 2) so it doesn't overlap the left branch.
            x_stretch = np.where(left, rx / alpha, (rx - alpha) / (1.0 - alpha) + 1.0)
            y_stretch = alpha * ry
            x_final = np.where(left, rx / alpha, (rx - alpha) / (1.0 - alpha))
            y_final = np.where(left, alpha * ry, alpha + (1.0 - alpha) * ry)
            state["left"] = left
            state["x_stretch"], state["y_stretch"] = x_stretch, y_stretch
            state["x_final"], state["y_final"] = x_final, y_final
            state["x_rest"], state["y_rest"] = rx, ry

        left = state["left"]
        rx, ry = state["x_rest"], state["y_rest"]
        x_stretch, y_stretch = state["x_stretch"], state["y_stretch"]
        x_final, y_final = state["x_final"], state["y_final"]

        if sub < n_stretch:
            t = (sub + 1) / n_stretch
            x = rx + t * (x_stretch - rx)
            y = ry + t * (y_stretch - ry)
            cut_line.set_visible(True)
            phase = "stretch"
        elif sub < n_stretch + n_pause:
            x, y = x_stretch, y_stretch
            cut_line.set_visible(True)
            phase = "cut"
        else:
            s = sub - n_stretch - n_pause
            t = (s + 1) / n_stack
            x = np.where(left, x_final, x_stretch + t * (x_final - x_stretch))
            y = np.where(left, y_final, y_stretch + t * (y_final - y_stretch))
            cut_line.set_visible(False)
            phase = "stack"

        scatter.set_offsets(np.column_stack([x, y]))
        title.set_text(f"Baker's map: {phase} (iteration {it} -> {it + 1})")
        if sub == frames_per_iteration - 1:
            state["x"], state["y"] = x_final, y_final
        return scatter, title, cut_line

    anim = FuncAnimation(fig, update, frames=total_frames, interval=interval, blit=False)
    return anim


def animate_map_orbit(
    system: DiscreteMap,
    state0: ArrayLike | None = None,
    n_iter: int = 4000,
    points_per_frame: int = 8,
    interval: int = 40,
    color: str = theme.PRIMARY,
    marker_size: float = 3.0,
    labels: tuple[str, str] = ("x", "y"),
    title: str | None = None,
) -> FuncAnimation:
    """Animate a 2D map's orbit being built up, iteration by iteration.

    Useful for watching a strange attractor (e.g. the Henon map's) emerge
    from a single seed point, rather than seeing only the finished scatter.

    Parameters
    ----------
    system : DiscreteMap
        Map to iterate; must have ``dim == 2``.
    state0 : array_like of float, shape (2,), optional
        Initial state; defaults to ``system.initial_state()``.
    n_iter : int, default 4000
        Total number of iterations to animate.
    points_per_frame : int, default 8
        Number of new orbit points revealed per animation frame.
    interval : int, default 40
        Delay between animation frames, in milliseconds.
    color : str, default theme.PRIMARY
        Marker color.
    marker_size : float, default 3.0
        Scatter marker size.
    labels : tuple of str, default ("x", "y")
        Axis labels.
    title : str, optional
        Axes title; defaults to ``"<ClassName> orbit"``.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    state0 = system.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
    traj = system.trajectory(state0, n_iter=n_iter)

    fig, ax = plt.subplots(figsize=(7, 6))
    pad_x = 0.05 * (traj[:, 0].max() - traj[:, 0].min() + 1e-12)
    pad_y = 0.05 * (traj[:, 1].max() - traj[:, 1].min() + 1e-12)
    ax.set_xlim(traj[:, 0].min() - pad_x, traj[:, 0].max() + pad_x)
    ax.set_ylim(traj[:, 1].min() - pad_y, traj[:, 1].max() + pad_y)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title or f"{system.__class__.__name__} orbit")

    scatter = ax.scatter([], [], s=marker_size, color=color, alpha=0.6)

    n_frames = int(np.ceil(traj.shape[0] / points_per_frame))

    def update(frame: int) -> tuple:
        end = min(traj.shape[0], (frame + 1) * points_per_frame)
        scatter.set_offsets(traj[:end])
        return (scatter,)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    return anim


def _nonsingular_lim(lo: float, hi: float) -> tuple[float, float]:
    """Pad a degenerate ``(lo, hi)`` axis range so it isn't singular.

    Mirrors what Matplotlib's own autoscaling does for a zero-width range,
    but ahead of time, so ``ax.set_xlim``/``set_ylim`` never has to fall
    back to its (warning-emitting) auto-expansion.
    """
    if lo == hi:
        pad = abs(lo) * 0.05 or 0.5
        return lo - pad, hi + pad
    return lo, hi


def animate_multi_orbit_map(
    trajectories: list[ArrayLike],
    interval: int = 60,
    colors: list[Any] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    labels: tuple[str, str] = ("x", "y"),
    title: str | None = None,
    marker_size: float = 2.0,
    n_frames: int = 100,
) -> FuncAnimation:
    """Animate a family of 2D orbits growing together, iteration by iteration.

    Useful for a phase-space portrait built from many initial conditions
    (e.g. the standard map's islands-in-a-chaotic-sea picture): every orbit
    grows one new point per frame (all together, not one whole orbit at a
    time), so the picture keeps visibly filling in throughout the animation
    rather than mostly settling within the first few frames.

    Parameters
    ----------
    trajectories : list of array_like of float, each shape (n_i, 2)
        Orbits to grow together; need not all have the same length (each
        stops advancing once it runs out of points).
    interval : int, default 60
        Delay between animation frames, in milliseconds.
    colors : list, optional
        Per-orbit colors; a `theme.SEQUENTIAL_CMAP` sweep is used if omitted.
    xlim, ylim : tuple of float, optional
        Axis limits; inferred from the data if omitted.
    labels : tuple of str, default ("x", "y")
        Axis labels.
    title : str, optional
        Axes title, updated each frame with the running iteration count.
    marker_size : float, default 2.0
        Marker size for each orbit's points (uses a real, visibly-sized
        marker, unlike Matplotlib's 1-device-pixel ``","`` marker which
        ignores this parameter entirely).
    n_frames : int, default 100
        Number of animation frames spanning every orbit's full length; each
        frame reveals ``1 / n_frames`` of the points, evenly paced so growth
        stays visible for the whole animation instead of being front-loaded.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    traj_arrays: list[NDArray[np.float64]] = [np.asarray(t, dtype=np.float64) for t in trajectories]
    n_orbits = len(traj_arrays)
    max_len = max(t.shape[0] for t in traj_arrays)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    if xlim is None or ylim is None:
        all_pts = np.concatenate(traj_arrays, axis=0)
        xlim = xlim or (float(all_pts[:, 0].min()), float(all_pts[:, 0].max()))
        ylim = ylim or (float(all_pts[:, 1].min()), float(all_pts[:, 1].max()))
    ax.set_xlim(*_nonsingular_lim(*xlim))
    ax.set_ylim(*_nonsingular_lim(*ylim))
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    cmap = plt.get_cmap(theme.SEQUENTIAL_CMAP)
    orbit_colors = [cmap(i / max(1, n_orbits - 1)) for i in range(n_orbits)] if colors is None else list(colors)

    lines = [ax.plot([], [], ".", color=orbit_colors[i], alpha=0.7, markersize=marker_size, lw=0)[0] for i in range(n_orbits)]

    frame_ends = np.linspace(1, max_len, n_frames).astype(int)

    def update(frame: int) -> tuple:
        end = int(frame_ends[frame])
        for line, traj in zip(lines, traj_arrays):
            stop = min(end, traj.shape[0])
            line.set_data(traj[:stop, 0], traj[:stop, 1])
        if title:
            ax.set_title(f"{title} (iteration {end}/{max_len})")
        return tuple(lines)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    return anim


def animate_map_cobweb(
    system: DiscreteMap,
    x0: float | None = None,
    n_iter: int = 30,
    interval: int = 400,
    x_range: tuple[float, float] = (0.0, 1.0),
    n_curve: int = 400,
    title: str | None = None,
) -> FuncAnimation:
    """Animate a cobweb (staircase) diagram for a 1D map ``x' = f(x)``.

    Alternates vertical jumps (up to the curve, i.e. computing ``f(x)``) with
    horizontal jumps (back down to the diagonal ``y = x``, i.e. feeding that
    value back in as the next ``x``) -- the classic way to see fixed points,
    cycles, and chaos geometrically for a 1D map.

    Parameters
    ----------
    system : DiscreteMap
        Map to iterate; must have ``dim == 1``.
    x0 : float, optional
        Starting point; defaults to ``system.initial_state()[0]``.
    n_iter : int, default 30
        Number of map iterations (staircase steps) to animate.
    interval : int, default 400
        Delay between animation frames, in milliseconds.
    x_range : tuple of float, default (0.0, 1.0)
        Range over which to plot the map curve ``f(x)`` and diagonal.
    n_curve : int, default 400
        Number of points used to draw the smooth ``f(x)`` curve.
    title : str, optional
        Axes title.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    x0 = float(system.initial_state()[0]) if x0 is None else float(x0)

    xs_curve = np.linspace(x_range[0], x_range[1], n_curve)
    ys_curve = np.array([system.step(np.array([xv]))[0] for xv in xs_curve])

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(xs_curve, ys_curve, color=theme.PRIMARY, lw=1.5, label="f(x)")
    ax.plot(x_range, x_range, color=theme.MUTED, lw=1.0, ls="--", label="y = x")
    ax.set_xlim(*x_range)
    ax.set_ylim(*x_range)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(title or f"{system.__class__.__name__} cobweb diagram")
    ax.legend(loc="upper left")

    (web,) = ax.plot([], [], color=theme.ACCENT, lw=1.0)
    (point,) = ax.plot([], [], "o", color=theme.ACCENT, markersize=5)

    web_x = [x0]
    web_y = [x0]
    x = x0
    for _ in range(n_iter):
        fx = float(system.step(np.array([x]))[0])
        web_x += [x, fx]
        web_y += [fx, fx]
        x = fx
    web_x_arr = np.array(web_x)
    web_y_arr = np.array(web_y)

    def update(frame: int) -> tuple:
        end = min(len(web_x_arr), 2 * frame + 1)
        web.set_data(web_x_arr[:end], web_y_arr[:end])
        point.set_data([web_x_arr[end - 1]], [web_y_arr[end - 1]])
        return web, point

    anim = FuncAnimation(fig, update, frames=n_iter + 1, interval=interval, blit=False)
    return anim


def animate_trajectory_3d(
    states: ArrayLike,
    labels: tuple[str, str, str] = ("x", "y", "z"),
    interval: int = 50,
    skip: int = 1,
    trail: int | None = None,
    color: str = theme.PRIMARY,
    title: str | None = None,
) -> FuncAnimation:
    """Animate a 3D trajectory being traced out over time.

    Draws the path as a growing line with a bright marker at its leading
    edge -- e.g. watching the Lorenz butterfly or a Rossler/Chua scroll being
    drawn stroke by stroke, rather than seeing only the finished attractor.

    Parameters
    ----------
    states : array_like of float, shape (n, 3)
        Trajectory to animate (e.g. from
        :meth:`~physicskit.chaos.core.base_system.DynamicalSystem.trajectory`).
    labels : tuple of str, default ("x", "y", "z")
        Axis labels.
    interval : int, default 50
        Delay between animation frames, in milliseconds.
    skip : int, default 1
        Number of trajectory points advanced per animation frame; increase
        for long trajectories so the animation covers them in a reasonable
        number of frames.
    trail : int, optional
        Number of most-recent points to keep visible; the whole path-so-far
        is kept if omitted.
    color : str, default theme.PRIMARY
        Line and marker color.
    title : str, optional
        Axes title.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    states = np.asarray(states, dtype=np.float64)
    n = states.shape[0]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(states[:, 0].min(), states[:, 0].max())
    ax.set_ylim(states[:, 1].min(), states[:, 1].max())
    ax.set_zlim(states[:, 2].min(), states[:, 2].max())
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    if title:
        ax.set_title(title)

    (line,) = ax.plot([], [], [], color=color, lw=0.8)
    (head,) = ax.plot([], [], [], "o", color=theme.ACCENT, markersize=5)

    frame_ends = np.arange(skip, n + skip, skip)
    frame_ends[-1] = n

    def update(frame: int) -> tuple:
        end = int(frame_ends[frame])
        lo = 0 if trail is None else max(0, end - trail)
        line.set_data_3d(states[lo:end, 0], states[lo:end, 1], states[lo:end, 2])
        head.set_data_3d([states[end - 1, 0]], [states[end - 1, 1]], [states[end - 1, 2]])
        return line, head

    anim = FuncAnimation(fig, update, frames=len(frame_ends), interval=interval, blit=False)
    return anim


def animate_phase_volume_contraction(
    system: _TrajectorySystem,
    center_state: ArrayLike | None = None,
    ball_radius: float = 3.0,
    n_points: int = 800,
    t_max: float = 8.0,
    dt: float = 0.01,
    n_frames: int = 150,
    interval: int = 50,
    warmup_steps: int = 500,
    seed: int | None = 0,
) -> FuncAnimation:
    """Animate a small ball of nearby states collapsing under a dissipative 3D flow.

    Liouville's theorem (``d(log V)/dt = trace(Jacobian)``) is usually
    illustrated by a single number -- the cumulative log-volume of an
    infinitesimal ball, computed along one trajectory (see
    :func:`~physicskit.chaos.utils.metrics.phase_volume_expansion`). This animates
    what that number actually *means*: a literal ball of `n_points` nearby
    initial conditions, each integrated forward independently, visibly
    flattening from a sphere into a thin sheet draped over the attractor --
    the geometric mechanism, for a dissipative system, behind both volume
    contraction and the emergence of a lower-dimensional (often fractal)
    attractor.

    Parameters
    ----------
    system : DynamicalSystem
        System to integrate; must have ``dim == 3``.
    center_state : array_like of float, shape (3,), optional
        Center of the initial ball. If omitted, a short `warmup_steps`
        integration from ``system.initial_state()`` is used to land near the
        attractor first, so the ball starts already collapsing onto it
        rather than spending time drifting there.
    ball_radius : float, default 3.0
        Radius of the initial ball of points.
    n_points : int, default 800
        Number of points sampled uniformly inside the ball.
    t_max : float, default 8.0
        Total integration time for every point.
    dt : float, default 0.01
        Integration step size.
    n_frames : int, default 150
        Number of animation frames, evenly spaced in time over ``[0, t_max]``.
    interval : int, default 50
        Delay between animation frames, in milliseconds.
    warmup_steps : int, default 500
        Number of integration steps used to find a default `center_state`
        near the attractor; ignored if `center_state` is given.
    seed : int, optional
        Seed for the random point sampler, for reproducibility.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Assign it to a variable to keep it alive, and display it with
        ``plt.show()`` or save it with ``anim.save(...)``.

    Raises
    ------
    ValueError
        If ``system.dim != 3``.
    """
    if system.dim != 3:
        raise ValueError("animate_phase_volume_contraction only supports 3D systems")

    if center_state is None:
        _, warmup = system.trajectory(n_steps=warmup_steps, dt=dt)
        center = warmup[-1]
    else:
        center = np.asarray(center_state, dtype=np.float64)

    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_points, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = ball_radius * rng.uniform(size=n_points) ** (1.0 / 3.0)
    points0 = center + directions * radii[:, None]

    n_steps = round(t_max / dt)
    all_states = np.empty((n_points, n_steps + 1, 3))
    for i in range(n_points):
        _, states = system.trajectory(state0=points0[i], dt=dt, n_steps=n_steps)
        all_states[i] = states

    frame_indices = np.linspace(0, n_steps, n_frames).astype(int)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    flat = all_states.reshape(-1, 3)
    ax.set_xlim(flat[:, 0].min(), flat[:, 0].max())
    ax.set_ylim(flat[:, 1].min(), flat[:, 1].max())
    ax.set_zlim(flat[:, 2].min(), flat[:, 2].max())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    scatter = ax.scatter(points0[:, 0], points0[:, 1], points0[:, 2], s=4, color=theme.ACCENT, alpha=0.6)
    title = ax.set_title(f"{system.__class__.__name__}: phase-space volume contracting (t=0.00)")

    def update(frame: int) -> tuple:
        idx = int(frame_indices[frame])
        pts = all_states[:, idx, :]
        scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        title.set_text(f"{system.__class__.__name__}: phase-space volume contracting (t={idx * dt:.2f})")
        return scatter, title

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    return anim


def animate_trajectory_2d(
    states: ArrayLike,
    labels: tuple[str, str] = ("x", "y"),
    interval: int = 50,
    skip: int = 1,
    trail: int | None = None,
    color: str = theme.PRIMARY,
    markers: list[dict[str, Any]] | None = None,
    equal_aspect: bool = False,
    title: str | None = None,
) -> FuncAnimation:
    """Animate a 2D trajectory being traced out over time.

    Draws the path as a growing line with a bright marker at its leading
    edge, e.g. a Duffing oscillator's phase portrait ``(x, v)`` unspooling or
    a restricted-three-body orbit looping around its primaries.

    Parameters
    ----------
    states : array_like of float, shape (n, 2)
        Trajectory to animate; only the first two columns are used, so a
        higher-dimensional state (e.g. ``(x, y, vx, vy)``) can be passed
        directly to plot its first two components.
    labels : tuple of str, default ("x", "y")
        Axis labels.
    interval : int, default 50
        Delay between animation frames, in milliseconds.
    skip : int, default 1
        Number of trajectory points advanced per animation frame; increase
        for long trajectories so the animation covers them in a reasonable
        number of frames.
    trail : int, optional
        Number of most-recent points to keep visible; the whole path-so-far
        is kept if omitted.
    color : str, default theme.PRIMARY
        Line and marker color.
    markers : list of dict, optional
        Extra fixed reference points to scatter once (e.g. the two primaries
        of a restricted three-body orbit); each dict needs ``"x"`` and
        ``"y"`` keys plus any ``ax.scatter`` keyword arguments.
    equal_aspect : bool, default False
        Whether to force an equal aspect ratio.
    title : str, optional
        Axes title.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    states = np.asarray(states, dtype=np.float64)
    n = states.shape[0]

    fig, ax = plt.subplots(figsize=(6.5, 6))
    pad_x = 0.05 * (states[:, 0].max() - states[:, 0].min() + 1e-12)
    pad_y = 0.05 * (states[:, 1].max() - states[:, 1].min() + 1e-12)
    ax.set_xlim(states[:, 0].min() - pad_x, states[:, 0].max() + pad_x)
    ax.set_ylim(states[:, 1].min() - pad_y, states[:, 1].max() + pad_y)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    if equal_aspect:
        ax.set_aspect("equal")
    if title:
        ax.set_title(title)

    for m in markers or []:
        ax.scatter([m["x"]], [m["y"]], **{k: v for k, v in m.items() if k not in ("x", "y")})
    if markers and any("label" in m for m in markers):
        ax.legend(loc="upper right", fontsize=8)

    (line,) = ax.plot([], [], color=color, lw=0.8)
    (head,) = ax.plot([], [], "o", color=theme.ACCENT, markersize=6)

    frame_ends = np.arange(skip, n + skip, skip)
    frame_ends[-1] = n

    def update(frame: int) -> tuple:
        end = int(frame_ends[frame])
        lo = 0 if trail is None else max(0, end - trail)
        line.set_data(states[lo:end, 0], states[lo:end, 1])
        head.set_data([states[end - 1, 0]], [states[end - 1, 1]])
        return line, head

    anim = FuncAnimation(fig, update, frames=len(frame_ends), interval=interval, blit=False)
    return anim


def animate_double_pendulum(
    system: DoublePendulum,
    state0: ArrayLike | None = None,
    dt: float = 0.01,
    n_steps: int = 1000,
    interval: int = 40,
    trail: int = 300,
    skip: int = 1,
) -> FuncAnimation:
    """Animate the double pendulum swinging, arms and all.

    Integrates the pendulum and renders it kinematically -- the two rigid
    rods and bobs swinging in real ``(x, y)`` space, with a fading trail
    behind the outer bob -- rather than only the abstract
    ``(theta, omega)`` trajectory.

    Parameters
    ----------
    system : DoublePendulum
        Pendulum to animate.
    state0 : array_like of float, shape (4,), optional
        Initial state ``(theta1, theta2, omega1, omega2)``; defaults to
        :meth:`~physicskit.chaos.systems.continuous.DoublePendulum.initial_state`.
    dt : float, default 0.01
        Integration step size.
    n_steps : int, default 1000
        Number of integration steps to integrate.
    interval : int, default 40
        Delay between animation frames, in milliseconds.
    trail : int, default 300
        Number of most-recent integration steps to keep visible as a trail
        behind the outer bob.
    skip : int, default 1
        Number of integration steps advanced per animation frame; increase
        for long integrations so the animation covers them in a reasonable
        number of frames.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    state0 = system.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
    _, states = system.trajectory(state0, dt=dt, n_steps=n_steps)
    th1, th2 = states[:, 0], states[:, 1]
    l1, l2 = system.l1, system.l2

    x1 = l1 * np.sin(th1)
    y1 = -l1 * np.cos(th1)
    x2 = x1 + l2 * np.sin(th2)
    y2 = y1 - l2 * np.cos(th2)

    reach = l1 + l2
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.05 * reach, 1.05 * reach)
    ax.set_ylim(-1.05 * reach, 1.05 * reach)
    ax.set_aspect("equal")
    ax.set_title("Double pendulum")

    (trail_line,) = ax.plot([], [], color=theme.ACCENT, lw=0.8, alpha=0.7)
    (rods,) = ax.plot([], [], "-", color=theme.STRUCTURE, lw=1.5)
    (bobs,) = ax.plot([], [], "o", color=theme.PRIMARY, markersize=9)

    n = states.shape[0]
    frame_ends = np.arange(skip, n + skip, skip)
    frame_ends[-1] = n

    def update(frame: int) -> tuple:
        end = int(frame_ends[frame]) - 1
        lo = max(0, end - trail)
        trail_line.set_data(x2[lo : end + 1], y2[lo : end + 1])
        rods.set_data([0.0, x1[end], x2[end]], [0.0, y1[end], y2[end]])
        bobs.set_data([x1[end], x2[end]], [y1[end], y2[end]])
        return trail_line, rods, bobs

    anim = FuncAnimation(fig, update, frames=len(frame_ends), interval=interval, blit=False)
    return anim


def animate_magnetic_pendulum(
    system: MagneticPendulum,
    state0: ArrayLike | None = None,
    dt: float = 0.05,
    n_steps: int = 600,
    interval: int = 40,
    trail: int = 150,
    skip: int = 1,
) -> FuncAnimation:
    """Animate a magnetic pendulum bob swinging in toward whichever magnet wins.

    Parameters
    ----------
    system : MagneticPendulum
        Pendulum to animate.
    state0 : array_like of float, shape (4,), optional
        Initial state ``(x, y, vx, vy)``; defaults to
        :meth:`~physicskit.chaos.systems.continuous.MagneticPendulum.initial_state`.
    dt : float, default 0.05
        Integration step size.
    n_steps : int, default 600
        Number of integration steps to integrate.
    interval : int, default 40
        Delay between animation frames, in milliseconds.
    trail : int, default 150
        Number of most-recent integration steps to keep visible as a trail.
    skip : int, default 1
        Number of integration steps advanced per animation frame; increase
        for long integrations so the animation covers them in a reasonable
        number of frames.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    state0 = system.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
    _, states = system.trajectory(state0, dt=dt, n_steps=n_steps)
    x, y = states[:, 0], states[:, 1]
    magnets = system.magnet_positions

    extent = max(1.5, float(np.abs(magnets).max()) * 1.4, float(np.abs(states[:, :2]).max()) * 1.1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.scatter(
        magnets[:, 0],
        magnets[:, 1],
        marker="X",
        s=120,
        color=theme.STRUCTURE,
        zorder=3,
        label="magnets",
    )
    ax.set_title("Magnetic pendulum")
    ax.legend(loc="upper right")

    (trail_line,) = ax.plot([], [], color=theme.PRIMARY, lw=1.0, alpha=0.8)
    (bob,) = ax.plot([], [], "o", color=theme.ACCENT, markersize=8, zorder=4)

    n = states.shape[0]
    frame_ends = np.arange(skip, n + skip, skip)
    frame_ends[-1] = n

    def update(frame: int) -> tuple:
        end = int(frame_ends[frame]) - 1
        lo = max(0, end - trail)
        trail_line.set_data(x[lo : end + 1], y[lo : end + 1])
        bob.set_data([x[end]], [y[end]])
        return trail_line, bob

    anim = FuncAnimation(fig, update, frames=len(frame_ends), interval=interval, blit=False)
    return anim


def animate_restricted_three_body(
    system: RestrictedThreeBody,
    state0: ArrayLike | None = None,
    dt: float = 0.0005,
    n_steps: int = 20000,
    interval: int = 30,
    trail: int = 1000,
    skip: int = 10,
) -> FuncAnimation:
    """Animate a CR3BP trajectory in the rotating frame, alongside both
    primaries and all five Lagrange points.

    The two primaries sit fixed (by construction of the rotating frame)
    at ``(-mu, 0)`` and ``(1 - mu, 0)``; the five Lagrange points (see
    :func:`~physicskit.chaos.systems.continuous.lagrange_points`) are marked and
    labeled once, since they too are fixed equilibria of the rotating-frame
    dynamics -- watching the test particle's trajectory loop past them
    directly shows how close (or not) a given orbit passes to each one.

    Parameters
    ----------
    system : RestrictedThreeBody
        CR3BP system to animate.
    state0 : array_like of float, shape (4,), optional
        Initial state ``(x, y, vx, vy)``; defaults to
        :meth:`~physicskit.chaos.systems.continuous.RestrictedThreeBody.initial_state`.
    dt : float, default 0.0005
        Integration step size.
    n_steps : int, default 20000
        Number of integration steps to integrate.
    interval : int, default 30
        Delay between animation frames, in milliseconds.
    trail : int, default 1000
        Number of most-recent integration steps to keep visible as a trail.
    skip : int, default 10
        Number of integration steps advanced per animation frame; increase
        for long integrations so the animation covers them in a reasonable
        number of frames.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    state0 = system.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
    _, states = system.trajectory(state0, dt=dt, n_steps=n_steps)
    x, y = states[:, 0], states[:, 1]
    mu = system.mu
    lpoints = lagrange_points(mu)

    fig, ax = plt.subplots(figsize=(7, 7))
    pad = 0.15 * max(1.0, float(np.abs(np.concatenate([x, y, lpoints.ravel()])).max()))
    ax.set_xlim(min(x.min(), lpoints[:, 0].min()) - pad, max(x.max(), lpoints[:, 0].max()) + pad)
    ax.set_ylim(min(y.min(), lpoints[:, 1].min()) - pad, max(y.max(), lpoints[:, 1].max()) + pad)
    ax.set_aspect("equal")
    ax.set_title(f"{system.__class__.__name__} (rotating frame)")

    ax.scatter([-mu], [0.0], marker="o", s=100, color=theme.STRUCTURE, zorder=3, label="primary")
    ax.scatter([1.0 - mu], [0.0], marker="o", s=35, color=theme.MUTED, zorder=3, label="secondary")
    ax.scatter(lpoints[:, 0], lpoints[:, 1], marker="^", s=40, color=theme.ACCENT, zorder=3, label="Lagrange points")
    for i, (lx, ly) in enumerate(lpoints, start=1):
        ax.annotate(f"L{i}", (lx, ly), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.legend(loc="upper right", fontsize=8)

    (trail_line,) = ax.plot([], [], color=theme.PRIMARY, lw=0.8)
    (head,) = ax.plot([], [], "o", color=theme.PRIMARY, markersize=6, zorder=4)

    n = states.shape[0]
    frame_ends = np.arange(skip, n + skip, skip)
    frame_ends[-1] = n

    def update(frame: int) -> tuple:
        end = int(frame_ends[frame]) - 1
        lo = max(0, end - trail)
        trail_line.set_data(x[lo : end + 1], y[lo : end + 1])
        head.set_data([x[end]], [y[end]])
        return trail_line, head

    anim = FuncAnimation(fig, update, frames=len(frame_ends), interval=interval, blit=False)
    return anim


def animate_driven_pendulum(
    system: DrivenPendulum,
    state0: ArrayLike | None = None,
    dt: float = 0.02,
    n_steps: int = 1500,
    interval: int = 40,
    trail: int = 200,
    skip: int = 1,
    length: float = 1.0,
) -> FuncAnimation:
    """Animate the driven, damped pendulum swinging in real space.

    Parameters
    ----------
    system : DrivenPendulum
        Pendulum to animate.
    state0 : array_like of float, shape (2,), optional
        Initial state ``(theta, omega)``; defaults to
        :meth:`~physicskit.chaos.systems.continuous.DrivenPendulum.initial_state`.
    dt : float, default 0.02
        Integration step size.
    n_steps : int, default 1500
        Number of integration steps to integrate.
    interval : int, default 40
        Delay between animation frames, in milliseconds.
    trail : int, default 200
        Number of most-recent integration steps to keep visible as a trail.
    skip : int, default 1
        Number of integration steps advanced per animation frame; increase
        for long integrations so the animation covers them in a reasonable
        number of frames.
    length : float, default 1.0
        Visual rod length; only ``g/l`` enters the dynamics, so the actual
        drawn length is an arbitrary visualization choice.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    state0 = system.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
    _, states = system.trajectory(state0, dt=dt, n_steps=n_steps)
    theta = states[:, 0]
    x = length * np.sin(theta)
    y = -length * np.cos(theta)

    reach = length * 1.2
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal")
    ax.set_title("Driven, damped pendulum")

    (trail_line,) = ax.plot([], [], color=theme.ACCENT, lw=0.8, alpha=0.7)
    (rod,) = ax.plot([], [], "-", color=theme.STRUCTURE, lw=1.5)
    (bob,) = ax.plot([], [], "o", color=theme.PRIMARY, markersize=10)

    n = states.shape[0]
    frame_ends = np.arange(skip, n + skip, skip)
    frame_ends[-1] = n

    def update(frame: int) -> tuple:
        end = int(frame_ends[frame]) - 1
        lo = max(0, end - trail)
        trail_line.set_data(x[lo : end + 1], y[lo : end + 1])
        rod.set_data([0.0, x[end]], [0.0, y[end]])
        bob.set_data([x[end]], [y[end]])
        return trail_line, rod, bob

    anim = FuncAnimation(fig, update, frames=len(frame_ends), interval=interval, blit=False)
    return anim
