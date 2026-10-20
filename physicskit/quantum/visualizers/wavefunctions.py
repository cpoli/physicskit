r"""Phase-colored HSV complex wavefunction plots and density animations."""

from __future__ import annotations

from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PolyCollection

__all__ = ["complex_to_rgb", "plot_complex_wavefunction", "animate_density", "animate_density_2d"]


def complex_to_rgb(psi: np.ndarray, max_abs: Optional[float] = None) -> np.ndarray:
    r"""Map a complex array to RGB: hue = phase, value = amplitude.

    .. math::

        \text{hue} = \frac{\arg\psi + \pi}{2\pi}, \qquad
        \text{value} = \min\!\left(\frac{\lvert\psi\rvert}{\max\lvert\psi\rvert}, 1\right).

    Parameters
    ----------
    psi : numpy.ndarray
        Complex-valued array.
    max_abs : float or None, optional
        Amplitude that maps to full brightness; defaults to ``abs(psi).max()``.

    Returns
    -------
    numpy.ndarray
        RGB array of shape ``(..., 3)``, suitable for ``imshow``/plotting.
    """
    amp = np.abs(psi)
    phase = np.angle(psi)
    if max_abs is None:
        max_abs = amp.max() if amp.max() > 0 else 1.0
    hue = (phase + np.pi) / (2 * np.pi)
    saturation = np.ones_like(hue)
    value = np.clip(amp / max_abs, 0, 1)
    hsv = np.stack([hue, saturation, value], axis=-1)
    return mcolors.hsv_to_rgb(hsv)


def _segment_quads(x: np.ndarray, density: np.ndarray) -> np.ndarray:
    """Vertices of the filled quad under each grid segment, shape ``(len(x) - 1, 4, 2)``."""
    quads = np.empty((len(x) - 1, 4, 2))
    quads[:, :, 0] = np.stack([x[:-1], x[:-1], x[1:], x[1:]], axis=-1)
    quads[:, 1, 1] = density[:-1]
    quads[:, 2, 1] = density[1:]
    quads[:, [0, 3], 1] = 0.0
    return quads


def plot_complex_wavefunction(x: np.ndarray, psi: np.ndarray, ax=None, density_scale: float = 1.0):
    r"""Plot a phase-colored probability density curve.

    Plots :math:`\lvert\psi(x)\rvert^2` as a filled curve, colored by the
    local phase :math:`\arg\psi(x)` (HSV hue) -- a compact way to show both
    the probability density and the complex phase of a wavefunction at once.

    Parameters
    ----------
    x : numpy.ndarray
        Positions.
    psi : numpy.ndarray
        Complex-valued wavefunction sampled on ``x``.
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on; a new figure/axis is created if omitted.
    density_scale : float, default=1.0
        Multiplicative scale factor applied to the plotted density.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    density = np.abs(psi) ** 2 * density_scale
    colors = complex_to_rgb(psi)

    # one PolyCollection of per-segment quads, not one fill_between per segment
    quads = PolyCollection(_segment_quads(x, density), facecolors=colors[:-1], linewidths=0)  # type: ignore[arg-type]  # matplotlib stubs reject ndarray verts/colors, which work at runtime
    quads.sticky_edges.y[:] = [0.0, float(density.max())]  # same autoscale as fill_between
    ax.add_collection(quads)
    ax.autoscale_view()

    ax.set_xlabel("x")
    ax.set_ylabel(r"$|\psi(x)|^2$")

    sm = plt.cm.ScalarMappable(cmap=_phase_colormap(), norm=mcolors.Normalize(-np.pi, np.pi))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\arg\,\psi(x)$")
    return ax


def _phase_colormap():
    hues = np.linspace(0, 1, 256)
    rgb = mcolors.hsv_to_rgb(np.stack([hues, np.ones_like(hues), np.ones_like(hues)], axis=-1))
    return mcolors.ListedColormap(rgb)


def animate_density(
    x: np.ndarray, frames: np.ndarray, times: np.ndarray | None = None, interval: int = 30, phase_colored: bool = True, ax=None
) -> FuncAnimation:
    """Animate a stack of wavefunction snapshots over time.

    Parameters
    ----------
    x : numpy.ndarray
        Positions.
    frames : numpy.ndarray
        Wavefunction snapshots, shape ``(n_frames, len(x))``. May be complex
        (density and phase both shown via :func:`complex_to_rgb`) or real
        (plain probability density curve).
    times : numpy.ndarray or None, optional
        Time of each frame, used to label the animation title.
    interval : int, default=30
        Delay between frames, in milliseconds.
    phase_colored : bool, default=True
        Whether to color the curve by phase (only applies if ``frames`` is
        complex).
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on; a new figure/axis is created if omitted.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    density = np.abs(frames) ** 2
    ymax = density.max() * 1.1

    if phase_colored and np.iscomplexobj(frames):
        max_abs = np.abs(frames).max()
        quads = PolyCollection(_segment_quads(x, density[0]), facecolors=complex_to_rgb(frames[0], max_abs=max_abs)[:-1], linewidths=0)  # type: ignore[arg-type]  # matplotlib stubs reject ndarray verts/colors, which work at runtime
        ax.add_collection(quads)

        def init():
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(0, ymax)
            ax.set_xlabel("x")
            ax.set_ylabel(r"$|\psi(x,t)|^2$")
            return []

        def update(i):
            quads.set_verts(_segment_quads(x, density[i]))  # type: ignore[arg-type]  # matplotlib stubs reject ndarray verts/colors, which work at runtime
            quads.set_facecolor(complex_to_rgb(frames[i], max_abs=max_abs)[:-1])  # type: ignore[arg-type]  # matplotlib stubs reject ndarray verts/colors, which work at runtime
            title = f"t = {times[i]:.3f}" if times is not None else f"frame {i}"
            ax.set_title(title)
            return (quads,)

        return FuncAnimation(fig, update, init_func=init, frames=len(frames), interval=interval, blit=False)

    (line,) = ax.plot(x, density[0])
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|\psi(x,t)|^2$")

    def update_simple(i):
        line.set_ydata(density[i])
        title = f"t = {times[i]:.3f}" if times is not None else f"frame {i}"
        ax.set_title(title)
        return (line,)

    return FuncAnimation(fig, update_simple, frames=len(frames), interval=interval, blit=False)


def animate_density_2d(
    X: np.ndarray,
    Y: np.ndarray,
    frames: np.ndarray,
    times: np.ndarray | None = None,
    interval: int = 30,
    phase_colored: bool = True,
    cmap: str = "viridis",
    ax=None,
) -> FuncAnimation:
    """Animate a stack of 2D wavefunction/density snapshots via ``imshow``.

    Parameters
    ----------
    X, Y : numpy.ndarray
        Coordinate meshgrid (``indexing='ij'``), e.g.
        ``SplitOperatorSolver2D.X/.Y``.
    frames : numpy.ndarray
        Snapshots, shape ``(n_frames, *X.shape)``. May be complex
        (density and phase both shown via :func:`complex_to_rgb`) or real
        (a plain probability-density heatmap).
    times : numpy.ndarray or None, optional
        Time of each frame, used to label the animation title.
    interval : int, default=30
        Delay between frames, in milliseconds.
    phase_colored : bool, default=True
        Whether to color by phase (only applies if ``frames`` is complex).
    cmap : str, default='viridis'
        Colormap used when not phase-colored.
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on; a new figure/axis is created if omitted.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    extent = [X.min(), X.max(), Y.min(), Y.max()]
    is_complex = phase_colored and np.iscomplexobj(frames)

    if is_complex:
        max_abs = np.abs(frames).max()
        im = ax.imshow(np.transpose(complex_to_rgb(frames[0], max_abs=max_abs), (1, 0, 2)), origin="lower", extent=extent, aspect="auto")
    else:
        density = np.abs(frames) ** 2 if np.iscomplexobj(frames) else frames
        vmax = density.max() * 1.1
        im = ax.imshow(density[0].T, origin="lower", extent=extent, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(i):
        if is_complex:
            im.set_data(np.transpose(complex_to_rgb(frames[i], max_abs=np.abs(frames).max()), (1, 0, 2)))
        else:
            im.set_data(density[i].T)
        title = f"t = {times[i]:.3f}" if times is not None else f"frame {i}"
        ax.set_title(title)
        return (im,)

    return FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)
