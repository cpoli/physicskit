"""Poincare sections and boundary phase-space (s, sin phi) plots for billiards."""

from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from physicskit.chaos.core.base_system import BilliardSystem
from physicskit.chaos.visualizers import theme


def plot_poincare_section(
    billiard: BilliardSystem,
    n_rays: int = 50,
    n_bounces: int = 200,
    pos: ArrayLike | None = None,
    ax: Axes | None = None,
    seed: int | None = None,
    **scatter_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Scatter-plot the boundary phase space ``(s, sin(phi))`` of a billiard.

    Launches `n_rays` trajectories from an interior point at random angles
    and collects `n_bounces` reflections from each, pooling all the resulting
    boundary hits into one Poincare section. Integrable billiards (Circle,
    Rectangle) trace out smooth invariant curves; chaotic (defocusing)
    billiards (Sinai, Stadium) fill the section with a dense chaotic sea.
    Rays are traced in a single call to
    :meth:`~physicskit.chaos.systems.billiards._RayTracingBilliard.simulate_many_rays`,
    which parallelizes across rays internally (Numba ``prange``).

    Parameters
    ----------
    billiard : BilliardSystem
        Billiard to sample.
    n_rays : int, default 50
        Number of independent trajectories to launch.
    n_bounces : int, default 200
        Number of reflections to trace per trajectory.
    pos : array_like of float, shape (2,), optional
        Shared launch position ``(x, y)`` for every ray; defaults to
        ``billiard.sample_interior_point()``. Worth overriding for a billiard
        whose default interior point is a point of special symmetry -- e.g.
        :class:`~physicskit.chaos.systems.billiards.CircleBilliard`'s is its exact
        center, from which *every* ray hits the boundary head-on
        (``sin(phi) = 0``), collapsing the whole section onto a single line
        instead of the family of invariant curves an off-center point traces
        out.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure and axes are created if omitted.
    seed : int, optional
        Seed for the random angle generator, for reproducibility.
    **scatter_kwargs
        Additional keyword arguments forwarded to ``ax.scatter``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    rng = np.random.default_rng(seed)
    interior = billiard.sample_interior_point() if pos is None else np.asarray(pos, dtype=np.float64)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_rays)
    result = billiard.simulate_many_rays(interior, angles, n_bounces)
    s, sin_phi = result["s"], result["sin_phi"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = cast(Figure, ax.figure)

    kwargs: dict[str, Any] = {"s": 0.6, "c": theme.PRIMARY, "alpha": 0.6}
    kwargs.update(scatter_kwargs)
    ax.scatter(s, sin_phi, **kwargs)
    ax.set_xlim(0.0, billiard.perimeter())
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("s (boundary arclength)")
    ax.set_ylabel(r"$\sin\varphi$")
    ax.set_title(f"{billiard.__class__.__name__} Poincare section")
    return fig, ax


def plot_billiard_trajectory(
    billiard: BilliardSystem,
    pos: ArrayLike,
    vel: ArrayLike,
    n_bounces: int = 50,
    ax: Axes | None = None,
    **plot_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a single trajectory bouncing inside the billiard's boundary.

    Parameters
    ----------
    billiard : BilliardSystem
        Billiard to trace the trajectory within.
    pos : array_like of float, shape (2,)
        Initial position ``(x, y)``; must lie in the billiard's interior.
    vel : array_like of float, shape (2,)
        Initial velocity direction ``(vx, vy)``; normalized internally.
    n_bounces : int, default 50
        Number of reflections to trace.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure and axes are created if omitted.
    **plot_kwargs
        Additional keyword arguments forwarded to ``ax.plot`` for the
        trajectory line.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    boundary = billiard.boundary_polyline()
    path, _ = billiard.trajectory_segments(pos, vel, n_bounces)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = cast(Figure, ax.figure)

    ax.plot(boundary[:, 0], boundary[:, 1], color=theme.STRUCTURE, lw=1.5)
    kwargs: dict[str, Any] = {"color": theme.ACCENT, "lw": 0.8}
    kwargs.update(plot_kwargs)
    ax.plot(path[:, 0], path[:, 1], **kwargs)
    ax.set_aspect("equal")
    ax.set_title(f"{billiard.__class__.__name__} trajectory")
    return fig, ax
