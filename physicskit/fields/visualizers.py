"""Plotting helpers for electrodynamics, soliton, and BEC fields.

Every function returns its figure object rather than calling ``show()``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

__all__ = [
    "plot_field_1d",
    "plot_poynting_field",
    "plot_bec_density",
    "plot_bec_phase",
    "animate_field_1d",
    "animate_field_2d",
    "animate_density_2d",
    "animate_flux_tube",
    "animate_casimir_modes",
]


def plot_field_1d(x: np.ndarray, u: np.ndarray, ax=None, label: str | None = None):
    """Plot a 1D field snapshot (a KdV, NLS envelope, or Sine-Gordon profile).

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    u : ndarray
        Field values (real; pass ``numpy.abs(psi)`` for a complex envelope).
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.
    label : str, optional
        Legend label for this trace.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.fields.solitons import kdv_soliton
    >>> x = np.linspace(-20, 20, 200)
    >>> fig, ax = plot_field_1d(x, kdv_soliton(x, c=4.0))
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(x, u, label=label)
    ax.set_xlabel("x")
    if label is not None:
        ax.legend()
    return fig, ax


def plot_poynting_field(X: np.ndarray, Y: np.ndarray, Sx: np.ndarray, Sy: np.ndarray, ax=None, stride: int = 4):
    """Quiver-plot the Poynting energy-flux field over a 2D grid.

    Parameters
    ----------
    X, Y : ndarray
        Real-space coordinate grids.
    Sx, Sy : ndarray
        Poynting vector components, as from
        :func:`physicskit.fields.electrodynamics.poynting_vector_tmz`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.
    stride : int, default=4
        Subsample every ``stride``-th grid point, to keep the quiver plot legible.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
    >>> Sx, Sy = -Y, X
    >>> fig, ax = plot_poynting_field(X, Y, Sx, Sy)
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    sl = (slice(None, None, stride), slice(None, None, stride))
    ax.quiver(X[sl], Y[sl], Sx[sl], Sy[sl])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    return fig, ax


def plot_bec_density(X: np.ndarray, Y: np.ndarray, psi: np.ndarray, ax=None):
    """Plot the condensate density :math:`|\\psi(\\mathbf{r})|^2` as a heatmap.

    Parameters
    ----------
    X, Y : ndarray
        Real-space coordinate grids.
    psi : ndarray of complex
        Condensate wavefunction.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    See Also
    --------
    plot_bec_phase : Plot the phase, which reveals vortex cores as singularities.

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.fields.quantum_fields import harmonic_trap_grid
    >>> X, Y, KX, KY, K2 = harmonic_trap_grid(40, 10.0)
    >>> psi = np.exp(-0.5 * (X ** 2 + Y ** 2)).astype(complex)
    >>> fig, ax = plot_bec_density(X, Y, psi)
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    im = ax.pcolormesh(X, Y, np.abs(psi) ** 2, cmap="viridis", shading="auto")
    fig.colorbar(im, ax=ax, label=r"$|\psi|^2$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    return fig, ax


def plot_bec_phase(X: np.ndarray, Y: np.ndarray, psi: np.ndarray, ax=None):
    """Plot the condensate phase :math:`\\arg\\psi(\\mathbf{r})`; vortex cores appear as :math:`2\\pi` singularities.

    Parameters
    ----------
    X, Y : ndarray
        Real-space coordinate grids.
    psi : ndarray of complex
        Condensate wavefunction.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    See Also
    --------
    plot_bec_density : Plot the density, which shows vortex cores as zeros.
    physicskit.fields.quantum_fields.count_vortices : Quantitative vortex detection.

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.fields.quantum_fields import harmonic_trap_grid, gpe_imprint_vortex
    >>> X, Y, KX, KY, K2 = harmonic_trap_grid(40, 10.0)
    >>> psi0 = np.exp(-0.5 * (X ** 2 + Y ** 2)).astype(complex)
    >>> psi = gpe_imprint_vortex(psi0, X, Y, [(0.0, 0.0)])
    >>> fig, ax = plot_bec_phase(X, Y, psi)
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    im = ax.pcolormesh(X, Y, np.angle(psi), cmap="twilight", vmin=-np.pi, vmax=np.pi, shading="auto")
    fig.colorbar(im, ax=ax, label="phase (rad)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    return fig, ax


def animate_field_1d(x: np.ndarray, frames: np.ndarray, times: np.ndarray | None = None, interval: int = 50, ylabel: str = "u(x, t)", ax=None) -> FuncAnimation:
    """Animate a sequence of 1D field snapshots as a line plot.

    Shared by the KdV, NLS, and Sine-Gordon evolvers: feed it the frames
    from :func:`physicskit.fields.solitons.kdv_evolve_frames`,
    :func:`~physicskit.fields.solitons.nls_evolve_frames`, or
    :func:`~physicskit.fields.solitons.sine_gordon_evolve_frames`.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    frames : ndarray, shape (n_frames, len(x))
        Field snapshots. Complex frames (an NLS wavefunction) are plotted
        as :math:`|\\psi|`; real frames (KdV, Sine-Gordon) are plotted directly.
    times : ndarray, optional
        Time of each frame, shown in the title.
    interval : int, default=50
        Delay between frames, in milliseconds.
    ylabel : str, default="u(x, t)"
        Y-axis label.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.fields.solitons import kdv_soliton, kdv_evolve_frames
    >>> x = np.linspace(-30, 30, 256, endpoint=False)
    >>> frames, times = kdv_evolve_frames(kdv_soliton(x, c=4.0, x0=-15), x, dt=0.001, steps_per_frame=50, n_frames=3)
    >>> anim = animate_field_1d(x, frames, times)
    >>> isinstance(anim, FuncAnimation)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    values = np.abs(frames) if np.iscomplexobj(frames) else np.asarray(frames)
    ymin, ymax = values.min(), values.max()
    pad = 0.1 * (ymax - ymin if ymax > ymin else 1.0)
    (line,) = ax.plot(x, values[0])
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)

    def update(i):
        line.set_ydata(values[i])
        ax.set_title(f"t = {times[i]:.4g}" if times is not None else f"frame {i}")
        return (line,)

    return FuncAnimation(fig, update, frames=len(values), interval=interval, blit=False)


def animate_field_2d(
    X: np.ndarray, Y: np.ndarray, frames: np.ndarray, times: np.ndarray | None = None, interval: int = 50, cmap: str = "RdBu_r", ax=None
) -> FuncAnimation:
    """Animate a sequence of 2D scalar field snapshots (e.g. FDTD ``Ez``) as an imshow heatmap.

    Uses a diverging colormap centered at zero, symmetric about the
    largest-magnitude value across all frames -- appropriate for an
    oscillating field like ``Ez``, unlike the non-negative densities
    handled by :func:`animate_density_2d`.

    Parameters
    ----------
    X, Y : ndarray
        Real-space coordinate grids (used only for their extent).
    frames : ndarray, shape (n_frames, Nx, Ny)
        Field snapshots, e.g. from :func:`physicskit.fields.electrodynamics.fdtd_2d_tmz_evolve`.
    times : ndarray, optional
        Time of each frame, shown in the title.
    interval : int, default=50
        Delay between frames, in milliseconds.
    cmap : str, default="RdBu_r"
        Diverging colormap.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.fields.electrodynamics import (
    ...     courant_limit_2d, fdtd_2d_tmz_evolve, oscillating_dipole_source)
    >>> Nx, Ny = 30, 30
    >>> x = np.arange(Nx) * 1e-3
    >>> y = np.arange(Ny) * 1e-3
    >>> X, Y = np.meshgrid(x, y, indexing="ij")
    >>> Ez0 = Hx0 = Hy0 = np.zeros((Nx, Ny))
    >>> eps_r = mu_r = np.ones((Nx, Ny))
    >>> dt = 0.5 * courant_limit_2d(1e-3, 1e-3)
    >>> source = oscillating_dipole_source(Nx // 2, Ny // 2, amplitude=1.0, freq=5e10)
    >>> frames, times = fdtd_2d_tmz_evolve(Ez0, Hx0, Hy0, eps_r, mu_r, steps=10, dt=dt, dx=1e-3, dy=1e-3, source=source, snapshot_stride=2)
    >>> anim = animate_field_2d(X, Y, frames, times)
    >>> isinstance(anim, FuncAnimation)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    vmax = np.max(np.abs(frames))
    vmax = vmax if vmax > 0 else 1.0
    extent = (X.min(), X.max(), Y.min(), Y.max())
    im = ax.imshow(frames[0].T, extent=extent, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax, label="Ez")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(i):
        im.set_data(frames[i].T)
        ax.set_title(f"t = {times[i]:.3g}" if times is not None else f"frame {i}")
        return (im,)

    return FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)


def animate_density_2d(
    frames: np.ndarray, extent: tuple | None = None, times: np.ndarray | None = None, interval: int = 50, cmap: str = "viridis", ax=None
) -> FuncAnimation:
    """Animate a sequence of non-negative 2D density snapshots (e.g. BEC or flux-tube energy density) as an imshow heatmap.

    Parameters
    ----------
    frames : ndarray, shape (n_frames, Nx, Ny)
        Snapshots. Complex frames (a wavefunction) are converted to
        :math:`|\\psi|^2`; real frames are used directly as the density/energy map.
    extent : tuple(float, float, float, float), optional
        ``(xmin, xmax, ymin, ymax)`` passed to ``imshow``; defaults to pixel indices.
    times : ndarray, optional
        Time (or other sweep parameter, e.g. propagation distance) of each frame.
    interval : int, default=50
        Delay between frames, in milliseconds.
    cmap : str, default="viridis"
        Colormap.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.fields.quantum_fields import harmonic_trap_grid, gpe_imprint_vortex, gpe_evolve
    >>> n, length = 32, 12.0
    >>> X, Y, KX, KY, K2 = harmonic_trap_grid(n, length)
    >>> V = 0.5 * (X ** 2 + Y ** 2)
    >>> psi0 = gpe_imprint_vortex(np.exp(-0.5 * (X ** 2 + Y ** 2)).astype(complex), X, Y, [(1.0, 0.0)])
    >>> frames, times = gpe_evolve(psi0, V, g=2.0, dt=1e-3, steps=20, K2=K2, snapshot_stride=5)
    >>> anim = animate_density_2d(frames, times=times)
    >>> isinstance(anim, FuncAnimation)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    density = np.abs(frames) ** 2 if np.iscomplexobj(frames) else np.asarray(frames)
    vmax = density.max()
    vmax = vmax if vmax > 0 else 1.0
    im = ax.imshow(density[0].T, extent=extent, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(i):
        im.set_data(density[i].T)
        ax.set_title(f"t = {times[i]:.3g}" if times is not None else f"frame {i}")
        return (im,)

    return FuncAnimation(fig, update, frames=len(density), interval=interval, blit=False)


def animate_flux_tube(shape: tuple, dx: float, dy: float, separations: np.ndarray, flux_quantum: float = 1.0, interval: int = 80, ax=None) -> FuncAnimation:
    """Animate the toy confinement flux tube stretching as two charges are pulled apart.

    Builds a frame for each separation in ``separations`` via
    :func:`physicskit.fields.electrodynamics.flux_tube_energy_density_2d`
    and draws it as a heatmap with two markers tracking the charge
    positions, so the confined-energy "tube" visibly stretches between them.

    Parameters
    ----------
    shape : tuple(int, int)
        ``(Nx, Ny)`` grid shape.
    dx, dy : float
        Grid spacing.
    separations : ndarray
        Sequence of charge separations to sweep over (the animation's "time" axis).
    flux_quantum : float, default=1.0
        Charge magnitude, as in :func:`~physicskit.fields.electrodynamics.flux_tube_field_1d`.
    interval : int, default=80
        Delay between frames, in milliseconds.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> anim = animate_flux_tube((120, 30), dx=0.2, dy=0.2, separations=np.linspace(4.0, 16.0, 5))
    >>> isinstance(anim, FuncAnimation)
    True
    """
    from physicskit.fields.electrodynamics import flux_tube_energy_density_2d

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    Nx, Ny = shape
    x = (np.arange(Nx) - Nx // 2) * dx
    y = (np.arange(Ny) - Ny // 2) * dy
    extent = (x.min(), x.max(), y.min(), y.max())
    frames = np.stack([flux_tube_energy_density_2d(shape, dx, dy, sep, flux_quantum) for sep in separations])
    vmax = frames.max()
    vmax = vmax if vmax > 0 else 1.0
    im = ax.imshow(frames[0].T, extent=extent, origin="lower", cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax, label="energy density")
    (markers,) = ax.plot([], [], "o", color="cyan", markersize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(i):
        im.set_data(frames[i].T)
        sep = separations[i]
        markers.set_data([-sep / 2, sep / 2], [0.0, 0.0])
        ax.set_title(f"separation = {sep:.2f}")
        return im, markers

    return FuncAnimation(fig, update, frames=len(separations), interval=interval, blit=False)


def animate_casimir_modes(d_values: np.ndarray, c: float = 1.0, n_show: int = 12, interval: int = 150) -> FuncAnimation:
    """Animate the discrete cavity mode spectrum and Casimir energy as the plate separation ``d`` is swept.

    Left panel: the first ``n_show`` discrete mode frequencies
    (:func:`physicskit.fields.quantum_fields.casimir_mode_frequencies`) as a
    stem plot, against the continuum they approach at large ``d`` (dashed
    line of slope :math:`\\pi c`) -- visibly denser (closer to the
    continuum) at large separation. Right panel: the regularized Casimir
    energy (:func:`physicskit.fields.quantum_fields.casimir_energy_1d`) as a
    curve over the full sweep, with a marker tracking the current ``d``.

    Parameters
    ----------
    d_values : ndarray
        Sequence of plate separations to sweep over (the animation's "time" axis).
    c : float, default=1.0
        Wave speed.
    n_show : int, default=12
        Number of discrete modes to show in the stem plot.
    interval : int, default=150
        Delay between frames, in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> anim = animate_casimir_modes(np.linspace(1.0, 5.0, 5))
    >>> isinstance(anim, FuncAnimation)
    True
    """
    from physicskit.fields.quantum_fields import casimir_energy_1d, casimir_mode_frequencies

    fig, (ax_modes, ax_energy) = plt.subplots(1, 2, figsize=(10, 4))
    n = np.arange(1, n_show + 1)
    energies = np.array([casimir_energy_1d(d, c) for d in d_values])

    ax_modes.set_xlim(0, n_show + 1)
    ax_modes.set_ylim(0, np.pi * c * n_show / d_values.min() * 1.1)
    ax_modes.set_xlabel("mode index n")
    ax_modes.set_ylabel(r"$\omega_n$")
    stem_container = ax_modes.stem(n, np.pi * c * n / d_values[0])
    (continuum_line,) = ax_modes.plot(n, np.pi * c * n / d_values[0], "k--", alpha=0.4, label="continuum slope")
    ax_modes.legend(loc="upper left")

    ax_energy.plot(d_values, energies, color="C0")
    (marker,) = ax_energy.plot([d_values[0]], [energies[0]], "o", color="C3", markersize=8)
    ax_energy.set_xlabel("plate separation d")
    ax_energy.set_ylabel("Casimir energy E(d)")

    def update(i):
        d = d_values[i]
        omega = casimir_mode_frequencies(d, c, n_show)
        stem_container.markerline.set_data(n, omega)
        segments = [[[xi, 0], [xi, yi]] for xi, yi in zip(n, omega)]
        stem_container.stemlines.set_segments(segments)
        continuum_line.set_ydata(np.pi * c * n / d)
        marker.set_data([d], [energies[i]])
        fig.suptitle(f"d = {d:.3g}")
        return stem_container.markerline, stem_container.stemlines, continuum_line, marker

    return FuncAnimation(fig, update, frames=len(d_values), interval=interval, blit=False)
