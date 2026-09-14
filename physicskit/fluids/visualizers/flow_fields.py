"""Streamline and vorticity-field plots, the workhorse visualizations of this package."""

from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from physicskit.fluids.systems.instabilities import simulate_kelvin_helmholtz, simulate_rayleigh_taylor
from physicskit.fluids.visualizers import theme

__all__ = ["plot_streamlines", "plot_vorticity_field", "animate_kelvin_helmholtz", "animate_rayleigh_taylor"]


def plot_streamlines(
    X: NDArray[np.float64], Y: NDArray[np.float64], u: NDArray[np.float64], v: NDArray[np.float64], ax: Axes | None = None, **streamplot_kwargs: Any
) -> tuple[Figure, Axes]:
    """Plot velocity-field streamlines.

    Parameters
    ----------
    X, Y : ndarray of float, shape (n, n)
        Real-space coordinate grids.
    u, v : ndarray of float, shape (n, n)
        Velocity components.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.
    **streamplot_kwargs
        Additional keyword arguments forwarded to ``ax.streamplot``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = cast(Figure, ax.figure)
    kwargs: dict[str, Any] = {"color": theme.STRUCTURE, "linewidth": 0.8, "density": 1.2}
    kwargs.update(streamplot_kwargs)
    ax.streamplot(X.T, Y.T, u.T, v.T, **kwargs)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    return fig, ax


def plot_vorticity_field(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    omega: NDArray[np.float64],
    u: NDArray[np.float64] | None = None,
    v: NDArray[np.float64] | None = None,
    ax: Axes | None = None,
    **pcolormesh_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a vorticity field as a heatmap, optionally overlaid with velocity streamlines.

    Parameters
    ----------
    X, Y : ndarray of float, shape (n, n)
        Real-space coordinate grids.
    omega : ndarray of float, shape (n, n)
        Vorticity field.
    u, v : ndarray of float, shape (n, n), optional
        Velocity components; if given, streamlines are overlaid on the heatmap.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.
    **pcolormesh_kwargs
        Additional keyword arguments forwarded to ``ax.pcolormesh``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = cast(Figure, ax.figure)
    vmax = np.max(np.abs(omega)) or 1.0
    kwargs: dict[str, Any] = {"cmap": theme.VORTICITY_CMAP, "shading": "auto", "vmin": -vmax, "vmax": vmax}
    kwargs.update(pcolormesh_kwargs)
    im = ax.pcolormesh(X, Y, omega, **kwargs)
    fig.colorbar(im, ax=ax, label="vorticity")
    if u is not None and v is not None:
        ax.streamplot(X.T, Y.T, u.T, v.T, color=theme.STRUCTURE, linewidth=0.6, density=1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    return fig, ax


def _animate_vorticity_snapshots(
    X: NDArray[np.float64], Y: NDArray[np.float64], snapshots: list[NDArray[np.float64]], title: str, interval: int, label: str = "vorticity"
) -> FuncAnimation:
    """Shared heatmap animator: redraws a vorticity-like (signed, diverging) field over a sequence of snapshots."""
    fig, ax = plt.subplots(figsize=(6, 6))
    vmax = max(float(np.max(np.abs(s))) for s in snapshots) or 1.0
    im = ax.pcolormesh(X, Y, snapshots[0], cmap=theme.VORTICITY_CMAP, shading="auto", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title(title)

    def update(frame):
        im.set_array(snapshots[frame].ravel())
        return (im,)

    return FuncAnimation(fig, update, frames=len(snapshots), interval=interval, blit=True)


def animate_kelvin_helmholtz(
    omega0: NDArray[np.float64], nu: float, dt: float, steps_per_frame: int, n_frames: int, length: float, interval: int = 60
) -> FuncAnimation:
    """Animate a Kelvin-Helmholtz shear layer rolling up into its characteristic vortex row.

    Repeatedly calls
    :func:`physicskit.fluids.systems.instabilities.simulate_kelvin_helmholtz`
    for ``steps_per_frame`` RK4 steps at a time, using each call's final
    vorticity as the next call's initial condition, and animates the
    resulting vorticity-field snapshots.

    Parameters
    ----------
    omega0 : ndarray of float, shape (n, n)
        Initial vorticity field, e.g. from
        :func:`physicskit.fluids.systems.instabilities.kelvin_helmholtz_ic`.
    nu : float
        Kinematic viscosity; must be positive.
    dt : float
        Time step per sub-step.
    steps_per_frame : int
        Number of RK4 steps advanced between animation frames.
    n_frames : int
        Number of animation frames.
    length : float
        Physical domain size.
    interval : int, default=60
        Delay between frames in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    See Also
    --------
    physicskit.fluids.systems.instabilities.simulate_kelvin_helmholtz : Supplies the vorticity snapshots animated here.
    animate_rayleigh_taylor : The companion buoyancy-driven instability animator.

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.fluids.systems.instabilities import kelvin_helmholtz_ic
    >>> n, length = 64, 2 * np.pi
    >>> omega0 = kelvin_helmholtz_ic(n, length, shear_width=0.1, perturbation_amplitude=0.05)
    >>> anim = animate_kelvin_helmholtz(omega0, nu=0.001, dt=0.0025, steps_per_frame=10, n_frames=4, length=length)
    >>> len(list(anim.new_frame_seq()))
    4
    """
    n = omega0.shape[0]
    x = np.linspace(0, length, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    omega = np.asarray(omega0, dtype=float)
    snapshots = [omega.copy()]
    for _ in range(n_frames - 1):
        result = simulate_kelvin_helmholtz(omega, nu, dt, steps_per_frame, length)
        omega = result["omega"]
        snapshots.append(omega.copy())
    return _animate_vorticity_snapshots(X, Y, snapshots, title="Kelvin-Helmholtz instability", interval=interval)


def animate_rayleigh_taylor(
    omega0: NDArray[np.float64],
    buoyancy0: NDArray[np.float64],
    nu: float,
    kappa: float,
    g: float,
    dt: float,
    steps_per_frame: int,
    n_frames: int,
    length: float,
    interval: int = 60,
) -> FuncAnimation:
    """Animate the Rayleigh-Taylor instability's mushroom plumes growing from a rippled density interface.

    Repeatedly calls
    :func:`physicskit.fluids.systems.instabilities.simulate_rayleigh_taylor`
    for ``steps_per_frame`` RK4 steps at a time, using each call's final
    state as the next call's initial condition, and animates the resulting
    buoyancy-field snapshots (the field in which the rising light plumes
    and falling heavy fingers are most visually distinct).

    Parameters
    ----------
    omega0, buoyancy0 : ndarray of float, shape (n, n)
        Initial vorticity and buoyancy fields, e.g. from
        :func:`physicskit.fluids.systems.instabilities.rayleigh_taylor_ic`.
    nu : float
        Kinematic viscosity; must be positive.
    kappa : float
        Buoyancy diffusivity; must be positive.
    g : float
        Gravitational acceleration; must be positive.
    dt : float
        Time step per sub-step.
    steps_per_frame : int
        Number of RK4 steps advanced between animation frames.
    n_frames : int
        Number of animation frames.
    length : float
        Physical domain size.
    interval : int, default=60
        Delay between frames in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    See Also
    --------
    physicskit.fluids.systems.instabilities.simulate_rayleigh_taylor : Supplies the buoyancy snapshots animated here.
    animate_kelvin_helmholtz : The companion shear-driven instability animator.

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.fluids.systems.instabilities import rayleigh_taylor_ic
    >>> n, length = 64, 2 * np.pi
    >>> omega0, buoyancy0 = rayleigh_taylor_ic(n, length, atwood_number=0.3, perturbation_amplitude=0.01)
    >>> anim = animate_rayleigh_taylor(omega0, buoyancy0, nu=0.002, kappa=0.002, g=1.0, dt=0.01, steps_per_frame=10, n_frames=4, length=length)
    >>> len(list(anim.new_frame_seq()))
    4
    """
    n = omega0.shape[0]
    x = np.linspace(0, length, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    omega = np.asarray(omega0, dtype=float)
    buoyancy = np.asarray(buoyancy0, dtype=float)
    snapshots = [buoyancy.copy()]
    for _ in range(n_frames - 1):
        result = simulate_rayleigh_taylor(omega, buoyancy, nu, kappa, g, dt, steps_per_frame, length)
        omega, buoyancy = result["omega"], result["buoyancy"]
        snapshots.append(buoyancy.copy())
    return _animate_vorticity_snapshots(X, Y, snapshots, title="Rayleigh-Taylor instability (buoyancy)", interval=interval, label="buoyancy")
