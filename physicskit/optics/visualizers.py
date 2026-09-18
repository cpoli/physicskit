"""Plotting helpers: ray traces, Gaussian beam envelopes, diffraction patterns, Wigner surfaces.

Matplotlib is used for 2D plots (ray traces through an optical system, beam
waist envelopes, diffraction-pattern heatmaps), and Plotly for the
interactive 3D Wigner phase-space surface. Every function returns its figure
object rather than calling ``show()``, so it composes cleanly into larger
figures or headless pipelines.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation

from physicskit.optics.wave import angular_spectrum_propagate, intensity

__all__ = [
    "plot_ray_trace",
    "plot_beam_envelope",
    "plot_diffraction_pattern",
    "interactive_wigner_surface",
    "animate_diffraction_propagation",
]


def plot_ray_trace(system, y0, theta0, ax=None):
    """Plot a ray's height through each element of an :class:`~physicskit.optics.ray.OpticalSystem`.

    Parameters
    ----------
    system : physicskit.optics.ray.OpticalSystem
        The optical system to trace the ray through.
    y0 : float
        Initial ray height.
    theta0 : float
        Initial ray angle.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> from physicskit.optics.ray import OpticalElement, OpticalSystem, free_space, thin_lens
    >>> system = OpticalSystem([
    ...     OpticalElement(free_space(1.0), name="d1", length=1.0),
    ...     OpticalElement(thin_lens(1.0), name="lens"),
    ...     OpticalElement(free_space(1.0), name="d2", length=1.0),
    ... ])
    >>> fig, ax = plot_ray_trace(system, y0=0.5, theta0=0.0)
    >>> isinstance(fig, plt.Figure)
    True
    >>> len(ax.lines[0].get_xdata())
    4
    """
    trace = system.trace_ray(y0, theta0)
    positions = np.concatenate(([0.0], np.cumsum([el.length for el in system.elements])))
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(positions, trace[:, 0], marker="o")
    for pos, el in zip(positions[:-1], system.elements):
        if el.length == 0.0 and el.name:
            ax.axvline(pos, color="gray", linewidth=0.8, linestyle=":")
            ax.annotate(
                el.name,
                (pos, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(2, -10),
                textcoords="offset points",
                fontsize=8,
                color="gray",
                rotation=90,
                va="top",
            )
    ax.set_xlabel("position")
    ax.set_ylabel("ray height y")
    ax.axhline(0.0, color="k", linewidth=0.5)
    return fig, ax


def plot_beam_envelope(beam, z_range, n_points=200, ax=None):
    r"""Plot a Gaussian beam's :math:`\pm w(z)` waist envelope over a propagation range.

    Parameters
    ----------
    beam : physicskit.optics.gaussian.GaussianBeam
        The beam to plot.
    z_range : tuple(float, float)
        ``(z_min, z_max)`` propagation-axis range to plot over.
    n_points : int, default=200
        Number of sampled points.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> from physicskit.optics.gaussian import GaussianBeam
    >>> beam = GaussianBeam(wavelength=0.5e-3, w0=0.1, z0=0.0)
    >>> fig, ax = plot_beam_envelope(beam, z_range=(-10, 10))
    >>> isinstance(fig, plt.Figure)
    True
    >>> bool(ax.lines[0].get_ydata().min() >= 0)
    True
    """
    z = np.linspace(z_range[0], z_range[1], n_points)
    w = beam.waist(z)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(z, w, color="C0")
    ax.plot(z, -w, color="C0")
    ax.fill_between(z, -w, w, alpha=0.15, color="C0")
    ax.axvline(beam.z0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("z")
    ax.set_ylabel("beam radius w(z)")
    return fig, ax


def plot_diffraction_pattern(U, dx, ax=None, log_scale=False):
    """Plot the intensity pattern of a complex diffracted field as a 2D heatmap.

    Parameters
    ----------
    U : ndarray of shape (Ny, Nx)
        Complex field amplitude (e.g. from
        :func:`physicskit.optics.wave.angular_spectrum_propagate`).
    dx : float
        Grid spacing, used to set physical axis extents.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.
    log_scale : bool, default=False
        If True, plot ``log10(intensity + eps)`` to reveal faint diffraction
        fringes alongside the bright central peak.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> from physicskit.optics.wave import circular_aperture
    >>> U = circular_aperture((64, 64), dx=1e-3, radius=5e-3).astype(complex)
    >>> fig, ax = plot_diffraction_pattern(U, dx=1e-3)
    >>> isinstance(fig, plt.Figure)
    True
    """
    I = intensity(U)
    if log_scale:
        I = np.log10(I + 1e-12 * I.max())
    ny, nx = I.shape
    extent = (-nx / 2 * dx, nx / 2 * dx, -ny / 2 * dx, ny / 2 * dx)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    im = ax.imshow(I, extent=extent, origin="lower", cmap="inferno")
    fig.colorbar(im, ax=ax, label="log10 intensity" if log_scale else "intensity")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig, ax


def animate_diffraction_propagation(aperture, wavelength, z_values, dx, log_scale=False, interval=100, ax=None):
    """Animate the diffraction pattern developing as propagation distance ``z`` increases.

    Steps :func:`physicskit.optics.wave.angular_spectrum_propagate` over
    each distance in ``z_values`` (treating ``z`` as the animation's "time"
    axis -- the standard way to visualize the Fresnel-to-Fraunhofer
    development of a diffraction pattern), and animates the resulting
    intensity as an imshow heatmap: a double-slit aperture's near-field
    wavefronts visibly evolve into the far-field interference fringes as
    ``z`` grows.

    Parameters
    ----------
    aperture : ndarray of shape (Ny, Nx)
        Input field immediately after the aperture (e.g. from
        :func:`physicskit.optics.wave.double_slit_aperture`).
    wavelength : float
        Wavelength.
    z_values : ndarray
        Sequence of propagation distances to sweep over.
    dx : float
        Grid spacing (same for input and output, since
        :func:`~physicskit.optics.wave.angular_spectrum_propagate` stays on the same grid).
    log_scale : bool, default=False
        If True, plot ``log10(intensity + eps)`` to reveal faint fringes.
    interval : int, default=100
        Delay between frames, in milliseconds.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.optics.wave import double_slit_aperture
    >>> ap = double_slit_aperture((64, 64), dx=1e-3, width=2e-3, separation=1e-2).astype(complex)
    >>> z_values = np.linspace(0.01, 2.0, 5)
    >>> anim = animate_diffraction_propagation(ap, wavelength=0.5e-3, z_values=z_values, dx=1e-3)
    >>> isinstance(anim, FuncAnimation)
    True
    """
    frames = np.stack([intensity(angular_spectrum_propagate(aperture, wavelength, z, dx)) for z in z_values])
    if log_scale:
        frames = np.log10(frames + 1e-12 * frames.max())
    ny, nx = frames.shape[1:]
    extent = (-nx / 2 * dx, nx / 2 * dx, -ny / 2 * dx, ny / 2 * dx)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    im = ax.imshow(frames[0], extent=extent, origin="lower", cmap="inferno", vmin=frames.min(), vmax=frames.max())
    fig.colorbar(im, ax=ax, label="log10 intensity" if log_scale else "intensity")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(i):
        im.set_data(frames[i])
        ax.set_title(f"z = {z_values[i]:.4g}")
        return (im,)

    return FuncAnimation(fig, update, frames=len(z_values), interval=interval, blit=False)


def interactive_wigner_surface(W, x_grid, p_grid, title=None):
    """An interactive 3D Plotly surface plot of a Wigner quasi-probability distribution.

    Parameters
    ----------
    W : ndarray of shape (Nx, Np)
        Wigner function, e.g. from
        :func:`physicskit.optics.quantum_optics.compute_wigner_function`.
    x_grid : ndarray of shape (Nx,)
        Grid of x quadrature values.
    p_grid : ndarray of shape (Np,)
        Grid of p quadrature values.
    title : str, optional
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.optics.quantum_optics import coherent_state, compute_wigner_function
    >>> x = np.linspace(-4, 4, 40)
    >>> p = np.linspace(-4, 4, 40)
    >>> W = compute_wigner_function(coherent_state(1.0, 20), x, p)
    >>> fig = interactive_wigner_surface(W, x, p)
    >>> isinstance(fig, go.Figure)
    True
    """
    fig = go.Figure(data=[go.Surface(x=x_grid, y=p_grid, z=W.T, colorscale="RdBu", cmid=0.0)])
    fig.update_layout(
        title=title,
        scene={"xaxis_title": "x", "yaxis_title": "p", "zaxis_title": "W(x,p)"},
    )
    return fig
