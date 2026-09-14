"""Matplotlib animation helpers.

:class:`SideBySideAnimator` is the production entry point: given a
:class:`physicskit.classical.core.base_system.SimulationResult`, it renders physical
space motion on the left panel and a live-updating phase-space
trajectory or energy-vs-time trace on the right, driven by a single
:class:`~matplotlib.animation.FuncAnimation`. Physical-space rendering
is necessarily system-specific (pendulum rods, lattice-chain beads, an
orbit, a 3D rigid body), so it is supplied via two small callbacks;
:func:`orbit_trace_animation` and :func:`pendulum_animation` are
ready-made convenience wrappers for the two most common cases.
"""

from __future__ import annotations

from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

__all__ = [
    "SideBySideAnimator",
    "orbit_trace_animation",
    "pendulum_animation",
    "animate_elastic_pendulum",
    "animate_rigid_body_tumble",
    "animate_eulers_disk",
    "animate_rattleback",
]


class SideBySideAnimator:
    """Side-by-side animation: physical space (left) + phase-space or
    energy diagnostics (right), built from a
    :class:`~physicskit.classical.core.base_system.SimulationResult`.

    Parameters
    ----------
    result : SimulationResult
        Output of ``system.integrate(...)``.
    init_physical : Callable[[Axes], artists]
        Called once with the left-panel axes; must create and return
        whatever Matplotlib artists it will later update (a tuple/list).
    update_physical : Callable[[artists, SimulationResult, int], artists]
        Called every frame with the artists from ``init_physical``, the
        full result, and the current sample index; must update the
        artists in place and return them.
    mode : {"phase", "energy"}
        Right-panel content: a growing (q, p) trace for one degree of
        freedom, or a growing energy(t) trace.
    dof_index : int
        Which column of ``result.q``/``result.p`` to plot when mode="phase"
        and the system has more than one degree of freedom.
    physical_3d : bool
        If True, the left axes are created with a 3D projection.
    stride : int
        Render only every ``stride``-th recorded sample (for long runs).
    """

    def __init__(
        self,
        result,
        init_physical: Callable,
        update_physical: Callable,
        mode: str = "phase",
        dof_index: int = 0,
        figsize=(11, 5),
        physical_3d: bool = False,
        stride: int = 1,
        left_title: str = "Physical space",
    ):
        if mode not in ("phase", "energy"):
            raise ValueError("mode must be 'phase' or 'energy'")
        self.result = result
        self.update_physical = update_physical
        self.mode = mode
        self.dof_index = dof_index
        self.stride = max(1, stride)
        self.frame_indices = np.arange(0, len(result.t), self.stride)

        self.fig = plt.figure(figsize=figsize)
        self.ax_left = self.fig.add_subplot(1, 2, 1, projection="3d" if physical_3d else None)
        self.ax_left.set_title(left_title)
        self.ax_right = self.fig.add_subplot(1, 2, 2)
        self._physical_artists = init_physical(self.ax_left)
        self._setup_right_panel()
        self.anim: Optional[FuncAnimation] = None

    def _dof(self, arr):
        return arr[:, self.dof_index] if arr.ndim == 2 else arr

    def _setup_right_panel(self):
        if self.mode == "phase":
            q, p = self._dof(self.result.q), self._dof(self.result.p)
            self.ax_right.plot(q, p, color="lightgray", linewidth=0.7)
            self.ax_right.set_xlabel("q")
            self.ax_right.set_ylabel("p")
            self.ax_right.set_title("Phase space")
            (self._right_trace,) = self.ax_right.plot([], [], color="crimson", linewidth=1.3)
            (self._right_point,) = self.ax_right.plot([], [], "o", color="crimson")
        else:
            e = self.result.energy
            self.ax_right.set_xlabel("t")
            self.ax_right.set_ylabel("Energy")
            self.ax_right.set_title("Energy")
            self.ax_right.set_xlim(self.result.t[0], self.result.t[-1])
            pad = 0.05 * (np.max(e) - np.min(e) + 1e-12)
            self.ax_right.set_ylim(np.min(e) - pad, np.max(e) + pad)
            (self._right_trace,) = self.ax_right.plot([], [], color="crimson")

    def _update_right(self, idx: int):
        if self.mode == "phase":
            q, p = self._dof(self.result.q)[: idx + 1], self._dof(self.result.p)[: idx + 1]
            self._right_trace.set_data(q, p)
            self._right_point.set_data(q[-1:], p[-1:])
            return [self._right_trace, self._right_point]
        self._right_trace.set_data(self.result.t[: idx + 1], self.result.energy[: idx + 1])
        return [self._right_trace]

    def _frame(self, frame_i: int):
        idx = self.frame_indices[frame_i]
        left_artists = self.update_physical(self._physical_artists, self.result, idx)
        right_artists = self._update_right(idx)
        return list(left_artists) + list(right_artists)

    def build(self, interval: int = 30, blit: bool = False) -> FuncAnimation:
        """Construct the underlying ``FuncAnimation``.

        Parameters
        ----------
        interval : int
            Delay between frames, in milliseconds.
        blit : bool
            Forwarded to ``FuncAnimation``.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            Note for docs/gallery use: Sphinx-Gallery's animation scraper
            only detects bare ``matplotlib.animation.Animation`` objects
            among a script's globals, not a ``SideBySideAnimator`` wrapping
            one. Assign *this* return value (not the ``SideBySideAnimator``
            instance) to the variable you want rendered as an animated GIF,
            e.g. ``anim = SideBySideAnimator(...).build()``.
        """
        self.anim = FuncAnimation(self.fig, self._frame, frames=len(self.frame_indices), interval=interval, blit=blit)
        return self.anim

    def save(self, filename: str, fps: int = None, **kwargs):
        """Save the animation, building it first if needed.

        ``fps`` defaults to 30 for the common case of a string/None
        ``writer`` (where Matplotlib constructs the writer for you);
        pass ``fps=None`` explicitly (the default here) when supplying
        an already-configured ``writer`` instance via ``kwargs`` --
        Matplotlib rejects ``fps`` together with a writer instance,
        since the instance already carries its own fps.

        Parameters
        ----------
        filename : str
            Output path.
        fps : int, optional
            Frame rate; see above for when to leave this ``None``.
        **kwargs
            Forwarded to ``FuncAnimation.save`` (e.g. ``writer=...``).
        """
        if self.anim is None:
            self.build()
        if fps is None and "writer" not in kwargs:
            fps = 30
        if fps is not None:
            kwargs["fps"] = fps
        self.anim.save(filename, **kwargs)


def pendulum_animation(result, positions_func: Callable[[np.ndarray], np.ndarray], **kwargs) -> SideBySideAnimator:
    """Ready-made :class:`SideBySideAnimator` for rod-and-bob systems
    (single or double pendulum, bead on a hoop).

    ``positions_func(q) -> flat array of (x, y) pairs`` maps a
    configuration ``q`` to the pivot-to-bob chain of Cartesian points
    to draw as connected rods (e.g. ``DoublePendulum.positions``).

    Parameters
    ----------
    result : physicskit.classical.core.base_system.SimulationResult
        Output of ``system.integrate(...)``.
    positions_func : callable
        ``q -> flat array of (x, y) pairs``.
    **kwargs
        Forwarded to :class:`SideBySideAnimator`.

    Returns
    -------
    SideBySideAnimator
    """
    p0 = positions_func(result.q[0]).reshape(-1, 2)
    reach = np.abs(p0).max() * 1.3 + 1e-6

    def init_physical(ax):
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_aspect("equal")
        (rod,) = ax.plot([], [], "o-", lw=2, color="steelblue")
        (trace,) = ax.plot([], [], lw=0.7, color="orange", alpha=0.6)
        return rod, trace

    def update_physical(artists, res, idx):
        rod, trace = artists
        pts = positions_func(res.q[idx]).reshape(-1, 2)
        xs = np.concatenate([[0.0], pts[:, 0]])
        ys = np.concatenate([[0.0], pts[:, 1]])
        rod.set_data(xs, ys)
        hist = np.array([positions_func(q).reshape(-1, 2)[-1] for q in res.q[: idx + 1 : max(1, idx // 200 or 1)]])
        trace.set_data(hist[:, 0], hist[:, 1])
        return rod, trace

    return SideBySideAnimator(result, init_physical, update_physical, left_title="Pendulum", **kwargs)


def orbit_trace_animation(result, **kwargs) -> SideBySideAnimator:
    """Ready-made :class:`SideBySideAnimator` for a 2D central-force orbit.

    E.g. :class:`physicskit.classical.systems.newtonian.KeplerSystem`: a growing
    orbital path on the left, phase/energy diagnostics on the right.

    Parameters
    ----------
    result : physicskit.classical.core.base_system.SimulationResult
        Output of ``system.integrate(...)``.
    **kwargs
        Forwarded to :class:`SideBySideAnimator`.

    Returns
    -------
    SideBySideAnimator
    """
    q = result.q
    reach = np.abs(q).max() * 1.2 + 1e-6

    def init_physical(ax):
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_aspect("equal")
        ax.plot(0, 0, "o", color="gold", markersize=10)
        (path,) = ax.plot([], [], color="steelblue", lw=1.0)
        (body,) = ax.plot([], [], "o", color="steelblue", markersize=6)
        return path, body

    def update_physical(artists, res, idx):
        path, body = artists
        path.set_data(res.q[: idx + 1, 0], res.q[: idx + 1, 1])
        body.set_data(res.q[idx : idx + 1, 0], res.q[idx : idx + 1, 1])
        return path, body

    return SideBySideAnimator(result, init_physical, update_physical, left_title="Orbit", **kwargs)


def animate_elastic_pendulum(system, result, interval: int = 30, trail: int = 200, n_coils: int = 12, coil_width: float = 0.06) -> FuncAnimation:
    """Animate a spring (elastic) pendulum: a zigzag coil connecting the
    fixed pivot to the swinging, stretching bob.

    Parameters
    ----------
    system : physicskit.classical.systems.lagrangian.ElasticPendulum
        Pendulum the `result` came from (used for its natural length `L0`).
    result : physicskit.classical.core.base_system.SimulationResult
        Output of ``system.integrate(...)``; ``result.q`` columns are
        ``(s, theta)``.
    interval : int, default 30
        Delay between animation frames, in milliseconds.
    trail : int, default 200
        Number of most-recent samples kept visible as a trail behind the bob.
    n_coils : int, default 12
        Number of zigzag half-periods drawn along the spring.
    coil_width : float, default 0.06
        Zigzag amplitude, as a fraction of the natural length `L0`.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    s, theta = result.q[:, 0], result.q[:, 1]
    r = system.L0 + s
    x = r * np.sin(theta)
    y = -r * np.cos(theta)

    reach = float(np.max(r)) * 1.15 + 1e-6
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal")
    ax.set_title("Elastic pendulum")

    (trail_line,) = ax.plot([], [], color="orange", lw=0.8, alpha=0.7)
    (spring,) = ax.plot([], [], "-", color="steelblue", lw=1.3)
    (bob,) = ax.plot([], [], "o", color="crimson", markersize=10)

    n_pts = 2 * n_coils + 1
    zigzag = coil_width * system.L0 * (-1.0) ** np.arange(n_pts)
    zigzag[0] = zigzag[-1] = 0.0
    frac = np.linspace(0.0, 1.0, n_pts)

    n = result.t.shape[0]

    def update(frame: int) -> tuple:
        lo = max(0, frame - trail)
        trail_line.set_data(x[lo : frame + 1], y[lo : frame + 1])

        rx, ry = x[frame], y[frame]
        perp = np.array([-ry, rx]) / (r[frame] + 1e-12)
        spring.set_data(frac * rx + perp[0] * zigzag, frac * ry + perp[1] * zigzag)
        bob.set_data([rx], [ry])
        return trail_line, spring, bob

    anim = FuncAnimation(fig, update, frames=n, interval=interval, blit=False)
    return anim


def animate_rigid_body_tumble(top, result, half_extents=(1.0, 0.6, 0.3), interval: int = 30, stride: int = 1) -> FuncAnimation:
    """Animate an :class:`~physicskit.classical.systems.rotations.EulerTop`'s
    principal-axis box tumbling in the lab frame -- a literal picture of
    the Dzhanibekov effect / tennis-racket theorem.

    Renders the rigid body as a simple wireframe box whose three edge
    half-lengths are chosen only to make the three principal axes visually
    distinguishable (they are *not* derived from the actual moments of
    inertia, which would require a full mass distribution, not just three
    scalars). Each frame rotates the box's body-frame vertices into the lab
    frame with ``top.rotation_matrix(state)``, built from the integrated
    orientation quaternion -- so a spin started dominantly about the
    intermediate-inertia axis is seen visibly flipping end-over-end.

    Parameters
    ----------
    top : physicskit.classical.systems.rotations.EulerTop
        The system that produced `result` (used for
        :meth:`~physicskit.classical.systems.rotations.EulerTop.rotation_matrix`).
    result : physicskit.classical.core.base_system.SimulationResult
        Output of ``top.integrate(...)``; ``result.y[:, :7]`` must hold
        ``(w1, w2, w3, qw, qx, qy, qz)``.
    half_extents : tuple of float, default (1.0, 0.6, 0.3)
        Box half-widths along the body's (1, 2, 3) principal axes.
    interval : int, default 30
        Delay between animation frames, in milliseconds.
    stride : int, default 1
        Render only every ``stride``-th recorded sample.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    a, b, c = half_extents
    verts_body = np.array([[sx * a, sy * b, sz * c] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(verts_body[i] != verts_body[j]) == 1]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    reach = float(np.linalg.norm(half_extents)) * 1.2
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_zlim(-reach, reach)
    ax.set_title("Rigid-body tumble (Dzhanibekov effect)")

    lines = [ax.plot([], [], [], color="steelblue", lw=1.5)[0] for _ in edges]

    frame_indices = np.arange(0, result.y.shape[0], stride)

    def update(frame_i: int) -> tuple:
        idx = frame_indices[frame_i]
        R = top.rotation_matrix(result.y[idx])
        verts_world = verts_body @ R.T
        for line, (i, j) in zip(lines, edges):
            line.set_data_3d(
                [verts_world[i, 0], verts_world[j, 0]],
                [verts_world[i, 1], verts_world[j, 1]],
                [verts_world[i, 2], verts_world[j, 2]],
            )
        return tuple(lines)

    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=interval, blit=False)
    return anim


def animate_eulers_disk(system, result, radius: float = 1.0, interval: int = 30, trail: int = 400) -> FuncAnimation:
    """Animate an :class:`~physicskit.classical.systems.rotations.EulersDisk`'s
    rolling contact point from above, spiraling inward and precessing
    faster as the disk runs down.

    The contact point's horizontal position is modeled as
    ``radius * sin(theta) * (cos(phi), sin(phi))`` -- as `theta` shrinks
    toward the finite-time collapse (see
    :class:`~physicskit.classical.systems.rotations.EulersDisk`), the spiral
    visibly tightens toward the center while `phi` (and hence the visible
    wobble/rattle rate) advances faster and faster.

    Parameters
    ----------
    system : physicskit.classical.systems.rotations.EulersDisk
        System the `result` came from (used for `radius` only implicitly,
        via the caller; kept for a consistent ``(system, result, ...)``
        signature with the other animators here).
    result : physicskit.classical.core.base_system.SimulationResult
        Output of ``system.integrate(...)``; ``result.y`` columns are
        ``(theta, phi)``.
    radius : float, default 1.0
        Visual disk radius (arbitrary; the model only fixes `theta`, `phi`).
    interval : int, default 30
        Delay between animation frames, in milliseconds.
    trail : int, default 400
        Number of most-recent samples kept visible as a trail.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    theta, phi = result.y[:, 0], result.y[:, 1]
    contact_x = radius * np.sin(theta) * np.cos(phi)
    contact_y = radius * np.sin(theta) * np.sin(phi)

    fig, ax = plt.subplots(figsize=(6, 6))
    reach = radius * 1.1
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal")
    ax.set_title("Euler's disk: contact point (top view)")

    (trail_line,) = ax.plot([], [], color="steelblue", lw=0.8, alpha=0.7)
    (point,) = ax.plot([], [], "o", color="crimson", markersize=8)

    n = result.t.shape[0]

    def update(frame: int) -> tuple:
        lo = max(0, frame - trail)
        trail_line.set_data(contact_x[lo : frame + 1], contact_y[lo : frame + 1])
        point.set_data([contact_x[frame]], [contact_y[frame]])
        return trail_line, point

    anim = FuncAnimation(fig, update, frames=n, interval=interval, blit=False)
    return anim


def animate_rattleback(system, result, interval: int = 30, stride: int = 1) -> FuncAnimation:
    """Animate a :class:`~physicskit.classical.systems.rotations.Rattleback`'s
    spin direction and spin-rate history side by side, so a reversal is
    directly visible both ways: the indicator arrow visibly reversing its
    sense of rotation, and the ``n3(t)`` trace crossing zero.

    Parameters
    ----------
    system : physicskit.classical.systems.rotations.Rattleback
        System the `result` came from (unused beyond duck-typing
        consistency with the other animators here).
    result : physicskit.classical.core.base_system.SimulationResult
        Output of ``system.integrate(...)``; ``result.y[:, 2]`` is `n3`
        (the spin-rate component).
    interval : int, default 30
        Delay between animation frames, in milliseconds.
    stride : int, default 1
        Render only every ``stride``-th recorded sample.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    t = result.t
    n3 = result.y[:, 2]
    frame_indices = np.arange(0, len(t), stride)

    # Cumulative rotation angle from integrating n3 dt: only its sign and
    # rate of change carry meaning (this is a reduced toy model, not a true
    # Euler angle), but it turns n3(t) into a visibly spinning indicator.
    dt_arr = np.diff(t, prepend=t[0])
    phase = np.cumsum(n3 * dt_arr)

    fig, (ax_spin, ax_trace) = plt.subplots(1, 2, figsize=(11, 5))

    ax_spin.set_xlim(-1.2, 1.2)
    ax_spin.set_ylim(-1.2, 1.2)
    ax_spin.set_aspect("equal")
    ax_spin.set_title("Spin direction indicator")
    ax_spin.plot([0], [0], "o", color="black", markersize=6)
    (arrow,) = ax_spin.plot([], [], "-", color="steelblue", lw=3)

    ax_trace.set_xlim(t[0], t[-1])
    y_max = float(np.max(np.abs(n3))) + 1e-12
    ax_trace.set_ylim(-1.05 * y_max, 1.05 * y_max)
    ax_trace.axhline(0.0, color="gray", lw=1.0, ls="--")
    ax_trace.set_xlabel("t")
    ax_trace.set_ylabel("n3 (spin rate)")
    ax_trace.set_title("Spin rate over time")
    (trace,) = ax_trace.plot([], [], color="crimson", lw=1.2)
    (point,) = ax_trace.plot([], [], "o", color="crimson")

    def update(frame_i: int) -> tuple:
        idx = frame_indices[frame_i]
        ang = phase[idx]
        arrow.set_data([0.0, np.cos(ang)], [0.0, np.sin(ang)])
        arrow.set_color("steelblue" if n3[idx] >= 0 else "crimson")
        trace.set_data(t[: idx + 1], n3[: idx + 1])
        point.set_data([t[idx]], [n3[idx]])
        return arrow, trace, point

    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=interval, blit=False)
    return anim
