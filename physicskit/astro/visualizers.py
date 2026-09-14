"""Plotting helpers: N-body trajectories, Lane-Emden profiles, and rotation curves.

Matplotlib is used throughout. Every function returns its figure object
rather than calling ``show()``, so it composes cleanly into larger
figures or headless pipelines.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from .cosmic_web import zeldovich_position

__all__ = [
    "plot_nbody_trajectories",
    "plot_lane_emden",
    "plot_rotation_curve",
    "animate_nbody_trajectories",
    "animate_zeldovich_collapse",
    "plot_zeldovich_snapshot",
    "animate_stellar_convection",
    "animate_dynamo_wave",
    "plot_dynamo_butterfly_diagram",
]


def plot_nbody_trajectories(history, ax=None):
    """Plot the (x, y) trajectories of every body in an N-body simulation.

    Parameters
    ----------
    history : ndarray of shape (n_steps + 1, N, 3)
        Position history, e.g. from
        :meth:`physicskit.astro.nbody.NBodySystem.simulate`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.astro.nbody import NBodySystem
    >>> pos = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    >>> vel = np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]])
    >>> system = NBodySystem(pos, vel, np.array([1.0, 1.0]))
    >>> history = system.simulate(dt=0.01, n_steps=50)
    >>> fig, ax = plot_nbody_trajectories(history)
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    n_bodies = history.shape[1]
    for b in range(n_bodies):
        ax.plot(history[:, b, 0], history[:, b, 1], label=f"body {b + 1}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    return fig, ax


def plot_lane_emden(xi, theta, ax=None):
    r"""Plot the Lane-Emden function :math:`\theta(\xi)`.

    Parameters
    ----------
    xi, theta : ndarray
        From :func:`physicskit.astro.stellar_structure.lane_emden`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> from physicskit.astro.stellar_structure import lane_emden
    >>> xi, theta = lane_emden(1.5)
    >>> fig, ax = plot_lane_emden(xi, theta)
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(xi, theta)
    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.scatter([xi[-1]], [theta[-1]], color="C1", zorder=3, label="surface")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\theta(\xi)$")
    ax.legend()
    return fig, ax


def plot_rotation_curve(r, v_model, ax=None, v_observed=None):
    """Plot a galactic rotation curve, optionally overlaid with observed data.

    Parameters
    ----------
    r : ndarray
        Radii.
    v_model : ndarray
        Model circular velocity, e.g. from
        :func:`physicskit.astro.galactic_dynamics.circular_velocity`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.
    v_observed : ndarray, optional
        Observed velocities to overlay as points.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> r = np.linspace(1, 20, 30)
    >>> v = np.sqrt(1.0 / r)
    >>> fig, ax = plot_rotation_curve(r, v)
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(r, v_model, label="model")
    if v_observed is not None:
        ax.scatter(r, v_observed, color="C1", s=15, label="observed")
    ax.set_xlabel("r")
    ax.set_ylabel(r"$v_c(r)$")
    ax.legend()
    return fig, ax


def animate_nbody_trajectories(history, interval: int = 30, trail: int = 200, skip: int = 1, colors=None, title: str | None = None):
    """Animate an N-body simulation's real-space trajectories being traced out.

    Draws each body's ``(x, y)`` path (the ``z`` component, if any, is
    ignored) with a fading trail -- older trail segments are more
    transparent -- and a bright marker at the body's current position, in
    a distinct color per body.

    Parameters
    ----------
    history : ndarray of shape (n_steps + 1, N, 3)
        Position history, e.g. from :meth:`physicskit.astro.nbody.NBodySystem.simulate`.
    interval : int, default 30
        Delay between animation frames, in milliseconds.
    trail : int, default 200
        Number of most-recent steps kept visible as each body's trail.
    skip : int, default 1
        Number of simulation steps advanced per animation frame; increase
        for long simulations so the animation covers them in a reasonable
        number of frames.
    colors : list, optional
        Per-body colors; a `tab10` sweep is used if omitted.
    title : str, optional
        Axes title.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Assign it to a variable to keep it alive, and display it with
        ``plt.show()`` or save it with ``anim.save(...)``.

    Examples
    --------
    >>> from physicskit.astro.nbody import NBodySystem, figure_eight_initial_conditions
    >>> pos, vel, masses = figure_eight_initial_conditions()
    >>> system = NBodySystem(pos, vel, masses)
    >>> history = system.simulate(dt=0.002, n_steps=200)
    >>> anim = animate_nbody_trajectories(history)
    >>> anim.__class__.__name__
    'FuncAnimation'
    """
    history = np.asarray(history, dtype=np.float64)
    n_steps_plus1, n_bodies, _ = history.shape
    xy = history[:, :, :2]

    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(n_bodies)]
    base_rgba = [np.array(mcolors.to_rgba(c)) for c in colors]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    pad = 0.1 * (np.abs(xy).max() + 1e-12)
    ax.set_xlim(xy[..., 0].min() - pad, xy[..., 0].max() + pad)
    ax.set_ylim(xy[..., 1].min() - pad, xy[..., 1].max() + pad)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)

    trail_collections = []
    markers = []
    for b in range(n_bodies):
        lc = LineCollection([], colors=[base_rgba[b]], linewidths=1.5)
        ax.add_collection(lc)
        trail_collections.append(lc)
        (marker,) = ax.plot([], [], "o", color=colors[b], markersize=8, label=f"body {b + 1}")
        markers.append(marker)
    ax.legend(loc="upper right", fontsize=8)

    frame_ends = np.arange(skip, n_steps_plus1 + skip, skip)
    frame_ends[-1] = n_steps_plus1

    def update(frame: int):
        end = int(frame_ends[frame])
        lo = max(0, end - trail)
        artists = []
        for b in range(n_bodies):
            pts = xy[lo:end, b, :]
            if pts.shape[0] >= 2:
                segments = np.stack([pts[:-1], pts[1:]], axis=1)
                n_seg = segments.shape[0]
                seg_colors = np.tile(base_rgba[b], (n_seg, 1))
                seg_colors[:, 3] = np.linspace(0.05, 1.0, n_seg)
                trail_collections[b].set_segments(segments)
                trail_collections[b].set_color(seg_colors)
            else:
                trail_collections[b].set_segments([])
            markers[b].set_data([xy[end - 1, b, 0]], [xy[end - 1, b, 1]])
            artists += [trail_collections[b], markers[b]]
        return artists

    anim = FuncAnimation(fig, update, frames=len(frame_ends), interval=interval, blit=False)
    return anim


def _local_density_on_grid(xy, x_edges, y_edges):
    """Local point-density proxy: the 2D-histogram bin count each point of
    ``xy`` falls into, evaluated against a shared, precomputed set of bin
    edges so densities are comparable across an animation's frames.

    Chosen over a KD-tree nearest-neighbor density estimate (e.g.
    ``scipy.spatial.cKDTree``, used elsewhere in this codebase) for
    :func:`animate_zeldovich_collapse` and :func:`plot_zeldovich_snapshot`
    because it needs no tree queries, no per-particle neighbor-count
    hyperparameter, and directly reflects the same coarse-grained
    overdensity -- particles piling up into the same small region of space
    -- that the eye picks out in the scatter plot as filaments and nodes.
    """
    counts, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[x_edges, y_edges])
    n_bins_x = len(x_edges) - 1
    n_bins_y = len(y_edges) - 1
    ix = np.clip(np.digitize(xy[:, 0], x_edges) - 1, 0, n_bins_x - 1)
    iy = np.clip(np.digitize(xy[:, 1], y_edges) - 1, 0, n_bins_y - 1)
    return counts[ix, iy]


def animate_zeldovich_collapse(q, D_values, k_vectors, amplitudes, phases, interval=80, s=2):
    r"""Animate the Zel'dovich approximation's collapse of a Lagrangian grid into the cosmic web.

    Shows a scatter plot of :func:`physicskit.astro.cosmic_web.zeldovich_position`
    evaluated at each growth factor in ``D_values``, frame by frame -- the
    initially regular grid of tracer particles progressively streaming
    together into filaments, sheets ("pancakes"), and dense nodes at
    filament intersections. Points are colored by a local point-density
    proxy (see :func:`_local_density_on_grid`): a fixed 2D histogram of the
    particle positions, shared across all frames, so density colors are
    directly comparable as structure grows in from D_values[0] to
    D_values[-1].

    Parameters
    ----------
    q : ndarray, shape (n_particles, 2)
        Lagrangian tracer positions, e.g. from
        :func:`physicskit.astro.cosmic_web.lagrangian_grid`.
    D_values : array_like
        Growth-factor values to animate through, one frame each.
    k_vectors : ndarray, shape (n_modes, 2)
    amplitudes : ndarray, shape (n_modes,)
    phases : ndarray, shape (n_modes,)
        Displacement-potential parameters, e.g. from
        :func:`physicskit.astro.cosmic_web.random_displacement_potential`.
    interval : int, default=80
        Delay between animation frames, in milliseconds.
    s : float, default=2
        Marker size.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Assign it to a variable to keep it alive, and display it with
        ``plt.show()`` or save it with ``anim.save(...)``.

    See Also
    --------
    plot_zeldovich_snapshot : The single-frame, non-animated companion.

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.astro.cosmic_web import lagrangian_grid, random_displacement_potential, first_caustic_time
    >>> q = lagrangian_grid(20, 1.0)
    >>> k_vectors, amplitudes, phases = random_displacement_potential(6, k_min=2 * np.pi, k_max=6 * np.pi, amplitude_scale=0.02, seed=0)
    >>> D_collapse = first_caustic_time(q, k_vectors, amplitudes, phases)
    >>> D_values = np.linspace(0.0, 1.2 * D_collapse, 10)
    >>> anim = animate_zeldovich_collapse(q, D_values, k_vectors, amplitudes, phases)
    >>> anim.__class__.__name__
    'FuncAnimation'
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    D_values = np.asarray(D_values, dtype=float)
    n_frames = D_values.shape[0]
    n_particles = q.shape[0]

    positions = np.empty((n_frames, n_particles, 2))
    for i, D in enumerate(D_values):
        positions[i] = zeldovich_position(q, D, k_vectors, amplitudes, phases)

    x_min, x_max = positions[..., 0].min(), positions[..., 0].max()
    y_min, y_max = positions[..., 1].min(), positions[..., 1].max()
    pad_x = 0.03 * (x_max - x_min + 1e-12)
    pad_y = 0.03 * (y_max - y_min + 1e-12)
    x_min, x_max = x_min - pad_x, x_max + pad_x
    y_min, y_max = y_min - pad_y, y_max + pad_y

    n_bins = max(20, int(np.sqrt(n_particles)))
    x_edges = np.linspace(x_min, x_max, n_bins + 1)
    y_edges = np.linspace(y_min, y_max, n_bins + 1)

    log_density = np.empty((n_frames, n_particles))
    for i in range(n_frames):
        log_density[i] = np.log10(_local_density_on_grid(positions[i], x_edges, y_edges) + 1.0)
    vmax = max(float(log_density.max()), 1e-12)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    scatter = ax.scatter(positions[0, :, 0], positions[0, :, 1], c=log_density[0], cmap="inferno", s=s, vmin=0.0, vmax=vmax)
    title = ax.set_title(f"D = {D_values[0]:.4f}")
    fig.colorbar(scatter, ax=ax, label=r"local density (log$_{10}$ scale)")

    def update(frame):
        scatter.set_offsets(positions[frame])
        scatter.set_array(log_density[frame])
        title.set_text(f"D = {D_values[frame]:.4f}")
        return scatter, title

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    return anim


def plot_zeldovich_snapshot(q, D, k_vectors, amplitudes, phases, ax=None, s=2):
    r"""A single static snapshot of the Zel'dovich-evolved particle field at one growth factor.

    The non-animated companion to :func:`animate_zeldovich_collapse`,
    using the same 2D-histogram local-density coloring.

    Parameters
    ----------
    q : ndarray, shape (n_particles, 2)
        Lagrangian tracer positions, e.g. from
        :func:`physicskit.astro.cosmic_web.lagrangian_grid`.
    D : float
        Growth factor at which to evaluate the Zel'dovich mapping.
    k_vectors : ndarray, shape (n_modes, 2)
    amplitudes : ndarray, shape (n_modes,)
    phases : ndarray, shape (n_modes,)
        Displacement-potential parameters, e.g. from
        :func:`physicskit.astro.cosmic_web.random_displacement_potential`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.
    s : float, default=2
        Marker size.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    See Also
    --------
    animate_zeldovich_collapse : The animated version of this snapshot.

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.astro.cosmic_web import lagrangian_grid, random_displacement_potential, first_caustic_time
    >>> q = lagrangian_grid(20, 1.0)
    >>> k_vectors, amplitudes, phases = random_displacement_potential(6, k_min=2 * np.pi, k_max=6 * np.pi, amplitude_scale=0.02, seed=0)
    >>> D_collapse = first_caustic_time(q, k_vectors, amplitudes, phases)
    >>> fig, ax = plot_zeldovich_snapshot(q, D_collapse, k_vectors, amplitudes, phases)
    >>> isinstance(fig, plt.Figure)
    True
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    x = zeldovich_position(q, D, k_vectors, amplitudes, phases)
    n_particles = x.shape[0]

    x_min, x_max = x[:, 0].min(), x[:, 0].max()
    y_min, y_max = x[:, 1].min(), x[:, 1].max()
    pad_x = 0.03 * (x_max - x_min + 1e-12)
    pad_y = 0.03 * (y_max - y_min + 1e-12)
    x_edges = np.linspace(x_min - pad_x, x_max + pad_x, max(20, int(np.sqrt(n_particles))) + 1)
    y_edges = np.linspace(y_min - pad_y, y_max + pad_y, max(20, int(np.sqrt(n_particles))) + 1)

    log_density = np.log10(_local_density_on_grid(x, x_edges, y_edges) + 1.0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
    else:
        fig = ax.figure
    scatter = ax.scatter(x[:, 0], x[:, 1], c=log_density, cmap="inferno", s=s)
    fig.colorbar(scatter, ax=ax, label=r"local density (log$_{10}$ scale)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"D = {D:.4f}")
    return fig, ax


def animate_stellar_convection(times, omega_snapshots, T_snapshots, interval=80):
    """Animate 2D convective rolls: vorticity and temperature side by side.

    Two panels sharing a time axis -- vorticity on the left, temperature
    perturbation on the right -- each redrawn frame by frame from
    :func:`physicskit.astro.stellar_dynamo.simulate_stellar_convection`'s
    output, showing the small initial noise roll up into churning
    convective plumes.

    Parameters
    ----------
    times : ndarray, shape (n_saved,)
    omega_snapshots : ndarray, shape (n_saved, ny, nx)
        Vorticity field snapshots.
    T_snapshots : ndarray, shape (n_saved, ny, nx)
        Temperature-perturbation field snapshots.
    interval : int, default=80
        Delay between animation frames, in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Assign it to a variable to keep it alive, and display it with
        ``plt.show()`` or save it with ``anim.save(...)``.

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.astro.stellar_dynamo import simulate_stellar_convection
    >>> times, omega_snaps, T_snaps = simulate_stellar_convection(
    ...     24, 24, 2 * np.pi, 2 * np.pi, n_steps=20, save_every=10, seed=0)
    >>> anim = animate_stellar_convection(times, omega_snaps, T_snaps)
    >>> anim.__class__.__name__
    'FuncAnimation'
    """
    omega_snapshots = np.asarray(omega_snapshots, dtype=np.float64)
    T_snapshots = np.asarray(T_snapshots, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    omega_max = np.abs(omega_snapshots).max() + 1e-12
    T_max = np.abs(T_snapshots).max() + 1e-12

    fig, (ax_omega, ax_T) = plt.subplots(1, 2, figsize=(11, 5))
    im_omega = ax_omega.imshow(omega_snapshots[0], origin="lower", cmap="RdBu_r", vmin=-omega_max, vmax=omega_max, aspect="auto")
    im_T = ax_T.imshow(T_snapshots[0], origin="lower", cmap="inferno", vmin=-T_max, vmax=T_max, aspect="auto")
    ax_omega.set_title("vorticity $\\omega$")
    ax_T.set_title("temperature perturbation $T'$")
    for ax in (ax_omega, ax_T):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im_omega, ax=ax_omega, fraction=0.046)
    fig.colorbar(im_T, ax=ax_T, fraction=0.046)
    suptitle = fig.suptitle(f"t = {times[0]:.3f}")
    fig.tight_layout()

    def update(frame):
        im_omega.set_data(omega_snapshots[frame])
        im_T.set_data(T_snapshots[frame])
        suptitle.set_text(f"t = {times[frame]:.3f}")
        return im_omega, im_T, suptitle

    anim = FuncAnimation(fig, update, frames=omega_snapshots.shape[0], interval=interval, blit=False)
    return anim


def animate_dynamo_wave(times, A_snapshots, B_snapshots, x, interval=80):
    """Animate the alpha-omega dynamo's poloidal and toroidal fields as traveling, growing waves.

    Draws :math:`A(x)` and :math:`B(x)` as two line plots sharing an
    axis, redrawn frame by frame from
    :func:`physicskit.astro.stellar_dynamo.simulate_alpha_omega_dynamo`'s
    output. In the unstable/growing regime, both the growing amplitude
    and the lateral migration of the wave pattern across `x` -- the
    dynamo wave whose space-time trace is the solar butterfly diagram --
    are visible as the animation runs; the y-axis is rescaled each frame
    to the current amplitude so the growing wave doesn't run off scale.

    Parameters
    ----------
    times : ndarray, shape (n_saved,)
    A_snapshots, B_snapshots : ndarray, shape (n_saved, nx)
        Poloidal-flux-proxy and toroidal-field snapshots.
    x : ndarray, shape (nx,)
        Spatial grid the snapshots live on.
    interval : int, default=80
        Delay between animation frames, in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Assign it to a variable to keep it alive, and display it with
        ``plt.show()`` or save it with ``anim.save(...)``.

    See Also
    --------
    plot_dynamo_butterfly_diagram : The static space-time summary of the same wave.

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.astro.stellar_dynamo import simulate_alpha_omega_dynamo
    >>> Lx = 2 * np.pi
    >>> x = np.linspace(0, Lx, 32, endpoint=False)
    >>> A0 = 1e-3 * np.cos(x)
    >>> B0 = np.zeros_like(A0)
    >>> times, A_snaps, B_snaps = simulate_alpha_omega_dynamo(A0, B0, alpha=1.0, shear=5.0, eta=0.05, Lx=Lx, dt=0.01, n_steps=20, save_every=5)
    >>> anim = animate_dynamo_wave(times, A_snaps, B_snaps, x)
    >>> anim.__class__.__name__
    'FuncAnimation'
    """
    times = np.asarray(times, dtype=np.float64)
    A_snapshots = np.asarray(A_snapshots, dtype=np.float64)
    B_snapshots = np.asarray(B_snapshots, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    (line_A,) = ax.plot(x, A_snapshots[0], label="$A(x)$ (poloidal-flux proxy)")
    (line_B,) = ax.plot(x, B_snapshots[0], label="$B(x)$ (toroidal field)")
    ax.set_xlabel("x (latitude/radius proxy)")
    ax.set_ylabel("field amplitude")
    ax.set_xlim(x.min(), x.max())
    ax.legend(loc="upper right")
    title = ax.set_title(f"t = {times[0]:.3f}")

    def update(frame):
        a, b = A_snapshots[frame], B_snapshots[frame]
        line_A.set_ydata(a)
        line_B.set_ydata(b)
        ymax = max(np.abs(a).max(), np.abs(b).max(), 1e-12) * 1.2
        ax.set_ylim(-ymax, ymax)
        title.set_text(f"t = {times[frame]:.3f}")
        return line_A, line_B, title

    anim = FuncAnimation(fig, update, frames=A_snapshots.shape[0], interval=interval, blit=False)
    return anim


def plot_dynamo_butterfly_diagram(times, B_snapshots, x, ax=None):
    """Static space-time (Hovmoller) plot of the alpha-omega dynamo wave: the "butterfly diagram".

    Displays :math:`B(x,t)` as an image with time along the x-axis and
    the spatial coordinate `x` (standing in for stellar latitude) along
    the y-axis. In the unstable regime, the migrating dynamo wave shows
    up as diagonal stripes of alternating sign drifting across the
    plot -- the same characteristic pattern, and the same underlying
    alpha-omega mechanism, behind the real Sun's sunspot-latitude
    butterfly diagram.

    Parameters
    ----------
    times : ndarray, shape (n_saved,)
    B_snapshots : ndarray, shape (n_saved, nx)
        Toroidal-field snapshots, e.g. from
        :func:`physicskit.astro.stellar_dynamo.simulate_alpha_omega_dynamo`.
    x : ndarray, shape (nx,)
        Spatial grid the snapshots live on.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    See Also
    --------
    animate_dynamo_wave : The animated, time-domain view of the same field.

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.astro.stellar_dynamo import simulate_alpha_omega_dynamo
    >>> Lx = 2 * np.pi
    >>> x = np.linspace(0, Lx, 32, endpoint=False)
    >>> A0 = 1e-3 * np.cos(x)
    >>> B0 = np.zeros_like(A0)
    >>> times, A_snaps, B_snaps = simulate_alpha_omega_dynamo(A0, B0, alpha=1.0, shear=5.0, eta=0.05, Lx=Lx, dt=0.01, n_steps=200, save_every=5)
    >>> fig, ax = plot_dynamo_butterfly_diagram(times, B_snaps, x)
    >>> isinstance(fig, plt.Figure)
    True
    """
    times = np.asarray(times, dtype=np.float64)
    B_snapshots = np.asarray(B_snapshots, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
    vmax = np.abs(B_snapshots).max() + 1e-12
    mesh = ax.pcolormesh(times, x, B_snapshots.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    fig.colorbar(mesh, ax=ax, label="$B(x, t)$")
    ax.set_xlabel("time")
    ax.set_ylabel("x (latitude/radius proxy)")
    ax.set_title("Dynamo wave butterfly diagram")
    return fig, ax
