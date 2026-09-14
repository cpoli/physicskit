"""Plotting helpers for quantum maps and quantum billiards.

Plots the objects :mod:`physicskit.chaos.quantum` produces -- Floquet eigenphases,
Husimi phase-space distributions, and billiard eigenfunctions -- following
the same ``(fig, ax)``-returning, ``ax=`` -accepting convention as the rest
of :mod:`physicskit.chaos.visualizers`. Deliberately absent: any plot of spectral
*statistics* (level-spacing histograms, spectral rigidity, and the like) --
that analysis, and the random-matrix-theory context for it, belongs to the
separate ``physicskit.rmt`` package, which can consume
:meth:`~physicskit.chaos.quantum.maps.QuantumKickedRotor.eigenphases`,
:meth:`~physicskit.chaos.quantum.maps.QuantumBakersMap.eigenphases`, or
:meth:`~physicskit.chaos.quantum.billiards.QuantumBilliard.wavenumbers` directly.
"""

from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from physicskit.chaos.quantum.billiards import QuantumBilliard
from physicskit.chaos.quantum.maps import QuantumBakersMap, QuantumKickedRotor
from physicskit.chaos.visualizers import theme


def plot_quantum_spectrum(eigenphases: ArrayLike, ax: Axes | None = None, **scatter_kwargs: Any) -> tuple[Figure, Axes]:
    """Scatter a quantum map's Floquet eigenphases on the unit circle.

    Parameters
    ----------
    eigenphases : array_like of float, shape (n,)
        Eigenphases in radians, e.g. from
        :meth:`~physicskit.chaos.quantum.maps.QuantumKickedRotor.eigenphases` or
        :meth:`~physicskit.chaos.quantum.maps.QuantumBakersMap.eigenphases`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure and axes are created if omitted.
    **scatter_kwargs
        Additional keyword arguments forwarded to ``ax.scatter``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    eigenphases = np.asarray(eigenphases, dtype=np.float64)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = cast(Figure, ax.figure)

    circle = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(circle), np.sin(circle), color=theme.MUTED, lw=0.8, zorder=1)
    kwargs: dict[str, Any] = {"s": 12, "color": theme.PRIMARY, "zorder": 2}
    kwargs.update(scatter_kwargs)
    ax.scatter(np.cos(eigenphases), np.sin(eigenphases), **kwargs)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\cos(\phi)$")
    ax.set_ylabel(r"$\sin(\phi)$")
    ax.set_title("Floquet eigenphases")
    return fig, ax


def plot_husimi(
    q_grid: ArrayLike,
    p_grid: ArrayLike,
    husimi: ArrayLike,
    ax: Axes | None = None,
    cmap: str = theme.SEQUENTIAL_CMAP,
    q_label: str = "q",
    p_label: str = "p",
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a Husimi phase-space distribution.

    Parameters
    ----------
    q_grid, p_grid : array_like of float, shape (resolution, resolution)
        Phase-space grid, as returned by
        :func:`~physicskit.chaos.quantum.husimi.husimi_function` (or
        :meth:`~physicskit.chaos.quantum.maps.QuantumKickedRotor.husimi` /
        :meth:`~physicskit.chaos.quantum.maps.QuantumBakersMap.husimi`).
    husimi : array_like of float, shape (resolution, resolution)
        The Husimi distribution.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure and axes are created if omitted.
    cmap : str, default theme.SEQUENTIAL_CMAP
        Colormap name.
    q_label, p_label : str, default "q", "p"
        Axis labels.
    title : str, optional
        Axes title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    q_grid = np.asarray(q_grid, dtype=np.float64)
    p_grid = np.asarray(p_grid, dtype=np.float64)
    husimi = np.asarray(husimi, dtype=np.float64)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6))
    else:
        fig = cast(Figure, ax.figure)

    mesh = ax.pcolormesh(q_grid, p_grid, husimi, cmap=cmap, shading="gouraud")
    fig.colorbar(mesh, ax=ax, label="Husimi intensity")
    ax.set_xlabel(q_label)
    ax.set_ylabel(p_label)
    if title:
        ax.set_title(title)
    return fig, ax


def plot_billiard_eigenstate(
    quantum_billiard: QuantumBilliard,
    eigenfunction: ArrayLike,
    ax: Axes | None = None,
    density: bool = False,
    show_boundary: bool = True,
) -> tuple[Figure, Axes]:
    """Plot one eigenfunction of a :class:`~physicskit.chaos.quantum.billiards.QuantumBilliard`.

    Parameters
    ----------
    quantum_billiard : QuantumBilliard
        The quantized billiard the eigenfunction belongs to (used for its
        grid and boundary outline).
    eigenfunction : array_like of float, shape (nx, ny)
        One eigenfunction, e.g. a single row of the array returned by
        :meth:`~physicskit.chaos.quantum.billiards.QuantumBilliard.eigenstates`; grid
        points outside the billiard should be ``nan``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure and axes are created if omitted.
    density : bool, default False
        If False (the default), plot the signed wavefunction ``psi`` itself
        with a diverging colormap, showing nodal lines directly; this is
        always available since :meth:`~physicskit.chaos.quantum.billiards.QuantumBilliard.eigenstates`
        solves a real symmetric eigenproblem, so prefer it for a consistent
        look across regular and chaotic billiards alike. If True, plot the
        probability density ``|psi|^2`` instead, with a sequential colormap.
    show_boundary : bool, default True
        Whether to overlay the billiard's boundary outline.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    eigenfunction = np.asarray(eigenfunction, dtype=np.float64)
    x_grid, y_grid = quantum_billiard.grid()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6))
    else:
        fig = cast(Figure, ax.figure)

    if density:
        field = eigenfunction**2
        cmap = theme.SEQUENTIAL_CMAP
        vmin, vmax = 0.0, np.nanmax(field)
        label = r"$|\psi|^2$"
    else:
        field = eigenfunction
        cmap = theme.DIVERGING_CMAP
        vmax = np.nanmax(np.abs(field))
        vmin = -vmax
        label = r"$\psi$"

    mesh = ax.pcolormesh(x_grid, y_grid, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    fig.colorbar(mesh, ax=ax, label=label)

    if show_boundary:
        boundary = quantum_billiard.billiard.boundary_polyline()
        ax.plot(boundary[:, 0], boundary[:, 1], color=theme.STRUCTURE, lw=1.2)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig, ax


def plot_weyl_law(quantum_billiard: QuantumBilliard, n_states: int = 30, ax: Axes | None = None) -> tuple[Figure, Axes]:
    """Compare a billiard's actual eigenvalue counting function to Weyl's law.

    Draws the exact (staircase) count of eigenvalues below each wavenumber
    alongside the smooth prediction from
    :meth:`~physicskit.chaos.quantum.billiards.QuantumBilliard.weyl_counting_function`.

    Parameters
    ----------
    quantum_billiard : QuantumBilliard
        The quantized billiard to check.
    n_states : int, default 30
        Number of lowest eigenstates to compute and count.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure and axes are created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    wavenumbers = quantum_billiard.wavenumbers(n_states)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = cast(Figure, ax.figure)

    counts = np.arange(1, n_states + 1)
    ax.step(wavenumbers, counts, where="post", color=theme.PRIMARY, label="actual $N(k)$")
    k_smooth = np.linspace(0.0, wavenumbers[-1] * 1.05, 400)
    ax.plot(
        k_smooth,
        quantum_billiard.weyl_counting_function(k_smooth),
        color=theme.ACCENT,
        ls="--",
        label="Weyl's law",
    )
    ax.set_xlabel("k")
    ax.set_ylabel("N(k)")
    ax.set_title(f"{quantum_billiard.billiard.__class__.__name__}: eigenvalue counting function")
    ax.legend(loc="upper left")
    return fig, ax


def animate_husimi_evolution(
    quantum_map: QuantumKickedRotor | QuantumBakersMap,
    states: ArrayLike,
    resolution: int = 80,
    n_wraps: int = 3,
    interval: int = 350,
    cmap: str = theme.SEQUENTIAL_CMAP,
    title: str | None = None,
) -> FuncAnimation:
    """Animate a quantum map state's Husimi distribution evolving, step by step.

    Watching a coherent wavepacket's Husimi distribution spread across phase
    space -- staying compact and orbit-like where the classical map is
    regular, filling in a chaotic sea where it is not -- is one of the most
    direct illustrations of quantum chaos: it shows the quantum-classical
    correspondence visibly *breaking down*.

    Parameters
    ----------
    quantum_map : QuantumKickedRotor or QuantumBakersMap
        The quantum map `states` were generated from (used for its `husimi`
        method).
    states : array_like of complex, shape (n_steps + 1, dim)
        State at each step, e.g. from
        :meth:`~physicskit.chaos.quantum.maps.QuantumKickedRotor.evolve` or
        :meth:`~physicskit.chaos.quantum.maps.QuantumBakersMap.evolve`.
    resolution : int, default 80
        Number of grid points along each phase-space axis.
    n_wraps : int, default 3
        Number of periodic images summed to periodize the coherent states
        used by :meth:`~physicskit.chaos.quantum.maps.QuantumKickedRotor.husimi`.
    interval : int, default 350
        Delay between animation frames, in milliseconds.
    cmap : str, default theme.SEQUENTIAL_CMAP
        Colormap name.
    title : str, optional
        Base title; each frame appends its step number.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Assign it to a variable to keep it alive, and display it with
        ``plt.show()`` or save it with ``anim.save(...)``.
    """
    states = np.asarray(states, dtype=np.complex128)
    n_steps = states.shape[0]

    q_grid, p_grid, husimi0 = quantum_map.husimi(states[0], resolution=resolution, n_wraps=n_wraps)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    mesh = ax.pcolormesh(q_grid, p_grid, husimi0, cmap=cmap, shading="gouraud", vmin=0.0, vmax=1.0)
    fig.colorbar(mesh, ax=ax, label="Husimi intensity")
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    base_title = title or f"{type(quantum_map).__name__} Husimi distribution"
    title_artist = ax.set_title(f"{base_title} (step 0)")

    def update(frame: int) -> tuple:
        _, _, husimi = quantum_map.husimi(states[frame], resolution=resolution, n_wraps=n_wraps)
        mesh.set_array(husimi.ravel())
        title_artist.set_text(f"{base_title} (step {frame})")
        return mesh, title_artist

    anim = FuncAnimation(fig, update, frames=n_steps, interval=interval, blit=False)
    return anim
