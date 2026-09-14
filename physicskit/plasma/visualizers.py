"""Plotting helpers for particle orbits, MHD equilibria, wave maps, and kinetic phase space.

Every Matplotlib function returns its figure and axes rather than calling
``show()``; :func:`plot_phase_space_interactive` returns a Plotly figure
for pan/zoom exploration of a PIC phase-space snapshot. The ``animate_*``
functions each return a :class:`matplotlib.animation.FuncAnimation` built
from a sequence of snapshots taken while repeatedly re-invoking the
corresponding time-domain simulation function -- save with, e.g.,
``anim.save(path, writer=PillowWriter(fps=10))``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation

from physicskit.plasma.acceleration import simulate_wakefield_acceleration, wakefield_e_field
from physicskit.plasma.instabilities import simulate_reconnection, simulate_weibel_filamentation
from physicskit.plasma.kinetic import deposit_number_density, pic_simulate
from physicskit.plasma.turbulence import simulate_hasegawa_mima
from physicskit.plasma.waves import ion_acoustic_soliton_evolve, simulate_alfven_wave

__all__ = [
    "plot_particle_orbit_3d",
    "plot_drift_trajectory",
    "plot_flux_surfaces",
    "plot_q_profile",
    "plot_phase_space",
    "plot_field_energy_history",
    "plot_cma_diagram",
    "plot_phase_space_interactive",
    "animate_reconnection",
    "animate_weibel_filamentation",
    "animate_two_stream_phase_space",
    "animate_langmuir_wave",
    "animate_ion_acoustic_soliton",
    "animate_drift_wave_turbulence",
    "animate_alfven_wave",
    "animate_wakefield_acceleration",
]


def plot_particle_orbit_3d(pos_hist: np.ndarray, ax=None):
    """Plot a charged particle's 3D trajectory (e.g. Boris-pusher gyro-orbit).

    Parameters
    ----------
    pos_hist : ndarray, shape (steps, 3)
        Position history, as from :func:`physicskit.plasma.single_particle.boris_integrate`.
    ax : matplotlib.axes.Axes3D, optional
        3D axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes3D

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.plasma.single_particle import boris_integrate, QE, MP
    >>> pos_hist, vel_hist = boris_integrate(
    ...     np.zeros(3), np.array([1e5, 0.0, 0.0]), QE, MP, np.zeros(3), np.array([0.0, 0.0, 1.0]), 1e-10, steps=200
    ... )
    >>> fig, ax = plot_particle_orbit_3d(pos_hist)
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    else:
        fig = ax.figure
    ax.plot(pos_hist[:, 0], pos_hist[:, 1], pos_hist[:, 2])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    return fig, ax


def plot_drift_trajectory(pos_hist: np.ndarray, ax=None):
    """Plot the guiding-center drift path as seen from above (the x-y plane).

    Parameters
    ----------
    pos_hist : ndarray, shape (steps, 3) or (steps, 2)
        Position history; only the first two components are used.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    See Also
    --------
    plot_particle_orbit_3d : The full 3D gyro-orbit this drift path averages over.

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.plasma.single_particle import boris_integrate, QE, MP
    >>> E = np.array([0.0, 1e3, 0.0])
    >>> B = np.array([0.0, 0.0, 1.0])
    >>> pos_hist, vel_hist = boris_integrate(np.zeros(3), np.zeros(3), QE, MP, E, B, 1e-10, steps=500)
    >>> fig, ax = plot_drift_trajectory(pos_hist)
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(pos_hist[:, 0], pos_hist[:, 1])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    return fig, ax


def plot_flux_surfaces(R: np.ndarray, Z: np.ndarray, psi: np.ndarray, ax=None, levels: int = 20):
    """Contour-plot poloidal flux surfaces :math:`\\psi(R, Z)` of a toroidal equilibrium.

    Parameters
    ----------
    R, Z : ndarray, shape (nr,), (nz,)
        Cylindrical coordinate grids.
    psi : ndarray, shape (nr, nz)
        Poloidal flux, as from :func:`physicskit.plasma.mhd.solve_grad_shafranov`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.
    levels : int, default=20
        Number of contour levels.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.plasma.mhd import solve_grad_shafranov
    >>> R = np.linspace(0.5, 1.5, 41)
    >>> Z = np.linspace(-0.5, 0.5, 41)
    >>> psi = solve_grad_shafranov(R, Z, c1=1.0, c2=-2.0)
    >>> fig, ax = plot_flux_surfaces(R, Z, psi)
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    cs = ax.contour(RR, ZZ, psi, levels=levels)
    fig.colorbar(cs, ax=ax, label=r"$\psi$")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    return fig, ax


def plot_q_profile(r: np.ndarray, q: np.ndarray, ax=None):
    """Plot the tokamak safety factor :math:`q(r)` against minor radius.

    Parameters
    ----------
    r : ndarray
        Minor-radius coordinate.
    q : ndarray
        Safety factor, e.g. from
        :func:`physicskit.plasma.mhd.safety_factor_large_aspect_ratio`
        evaluated at each ``r``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.plasma.mhd import safety_factor_large_aspect_ratio
    >>> r = np.linspace(0.05, 0.5, 20)
    >>> q = np.array([safety_factor_large_aspect_ratio(ri, R0=1.0, Bt=2.0, Bp=0.2) for ri in r])
    >>> fig, ax = plot_q_profile(r, q)
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(r, q)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="q=1 (sawtooth)")
    ax.set_xlabel("minor radius r (m)")
    ax.set_ylabel("safety factor q")
    ax.legend()
    return fig, ax


def plot_phase_space(x: np.ndarray, v: np.ndarray, ax=None, bins: int = 64):
    """Heatmap the particle-in-cell phase-space density :math:`f(x, v)` from a particle snapshot.

    Parameters
    ----------
    x, v : ndarray, shape (n_particles,)
        Particle positions and velocities, e.g. from
        :func:`physicskit.plasma.kinetic.pic_simulate`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.
    bins : int, default=64
        Number of histogram bins along each axis.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    See Also
    --------
    plot_phase_space_interactive : An interactive Plotly scatter of the same data.

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.plasma.kinetic import landau_damping_ic
    >>> x, v = landau_damping_ic(2000, L=4 * np.pi, k_mode=0.5, alpha=0.1, v_th=1.0, seed=0)
    >>> fig, ax = plot_phase_space(x, v)
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    h = ax.hist2d(x, v, bins=bins, cmap="inferno")
    fig.colorbar(h[3], ax=ax, label="particle count")
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    return fig, ax


def plot_field_energy_history(t: np.ndarray, field_energy: np.ndarray, ax=None):
    """Semilog plot of electrostatic field energy vs. time, showing Landau damping (or two-stream growth).

    Parameters
    ----------
    t : ndarray
        Time array, as from :func:`physicskit.plasma.kinetic.pic_simulate`.
    field_energy : ndarray
        Field energy at each time, same shape as ``t``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.plasma.kinetic import landau_damping_ic, pic_simulate
    >>> x0, v0 = landau_damping_ic(2000, L=4 * np.pi, k_mode=0.5, alpha=0.05, v_th=1.0, seed=0)
    >>> result = pic_simulate(x0, v0, L=4 * np.pi, ng=32, dt=0.1, steps=30)
    >>> fig, ax = plot_field_energy_history(result["t"], result["field_energy"])
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.semilogy(t, field_energy)
    ax.set_xlabel("t (1/$\\omega_{pe}$)")
    ax.set_ylabel("field energy")
    return fig, ax


def plot_cma_diagram(X: np.ndarray, Y: np.ndarray, ax=None):
    """Scatter a set of plasma states on log-log Clemmow-Mullaly-Allis (CMA) diagram axes.

    Parameters
    ----------
    X, Y : ndarray
        CMA coordinates, as from :func:`physicskit.plasma.waves.cma_coordinates`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.plasma.waves import cma_coordinates
    >>> omega = np.linspace(0.1, 5.0, 30)
    >>> X, Y = cma_coordinates(omega, wpe=2.0, wce=3.0)
    >>> fig, ax = plot_cma_diagram(X, Y)
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.loglog(X, Y, "o-")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$X = \omega_{pe}^2/\omega^2$")
    ax.set_ylabel(r"$Y = \omega_{ce}/\omega$")
    return fig, ax


def plot_phase_space_interactive(x: np.ndarray, v: np.ndarray):
    """Interactive Plotly scatter of a particle-in-cell phase-space snapshot.

    Unlike :func:`plot_phase_space`, points remain individually
    identifiable under pan and zoom -- useful for inspecting fine
    structure like phase-space vortices in a developed two-stream
    instability.

    Parameters
    ----------
    x, v : ndarray, shape (n_particles,)
        Particle positions and velocities.

    Returns
    -------
    plotly.graph_objects.Figure

    See Also
    --------
    plot_phase_space : The static Matplotlib heatmap equivalent.

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.plasma.kinetic import two_stream_ic
    >>> x, v = two_stream_ic(500, L=10.0, v_drift=3.0, v_th=0.5, seed=0)
    >>> fig = plot_phase_space_interactive(x, v)
    >>> isinstance(fig, go.Figure)
    True
    """
    fig = go.Figure(data=go.Scattergl(x=x, y=v, mode="markers", marker={"size": 3, "opacity": 0.5}))
    fig.update_layout(xaxis_title="x", yaxis_title="v")
    return fig


def _animate_1d_snapshots(x: np.ndarray, snapshots: list, xlabel: str, ylabel: str, title: str | None = None, interval: int = 60) -> FuncAnimation:
    """Shared line-plot animator: redraws a single ``y(x)`` curve over a sequence of snapshots."""
    fig, ax = plt.subplots()
    (line,) = ax.plot(x, snapshots[0])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    stacked = np.stack(snapshots)
    lo, hi = stacked.min(), stacked.max()
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    ax.set_ylim(lo - pad, hi + pad)

    def update(frame):
        line.set_ydata(snapshots[frame])
        return (line,)

    return FuncAnimation(fig, update, frames=len(snapshots), interval=interval, blit=True)


def _animate_2d_snapshots(
    snapshots: list, extent: tuple, xlabel: str = "x", ylabel: str = "y", cmap: str = "RdBu_r", title: str | None = None, interval: int = 60
) -> FuncAnimation:
    """Shared imshow animator: redraws a single 2D scalar field over a sequence of snapshots."""
    fig, ax = plt.subplots()
    vmax = max(np.max(np.abs(s)) for s in snapshots) or 1.0
    im = ax.imshow(snapshots[0].T, origin="lower", extent=extent, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    def update(frame):
        im.set_data(snapshots[frame].T)
        return (im,)

    return FuncAnimation(fig, update, frames=len(snapshots), interval=interval, blit=True)


def animate_reconnection(
    psi0: np.ndarray, eta: float, v0: float, dt: float, steps_per_frame: int, n_frames: int, Lx: float, Ly: float, interval: int = 60
) -> FuncAnimation:
    """Animate the flux function :math:`\\psi(x,y,t)` reconnecting at the X-point of a resistive current sheet.

    Repeatedly calls :func:`physicskit.plasma.instabilities.simulate_reconnection`
    for ``steps_per_frame`` steps at a time, using each call's final state as
    the next call's initial condition, and shows the accumulated flux
    snapshots as an imshow animation -- the antiparallel field lines above
    and below the sheet visibly merge into a single reconnected topology at
    the X-point, and the squeezed-out reconnected flux forms the outflow
    "jets" along the sheet.

    Each snapshot has its :math:`x`-mean subtracted, :math:`\\psi(x,y,t) -
    \\langle\\psi\\rangle_x(y,t)`, before display. The unperturbed Harris
    profile :math:`-L\\ln\\cosh(y/L)` grows without bound away from the
    sheet, so on a fixed color scale it dwarfs the localized island/X-point
    structure that is actually reconnecting -- left in, the animation reads
    as visually static even though the underlying field is evolving.
    Removing the (x-independent) background isolates exactly the
    x-varying perturbation that breaks and reconnects, which is what makes
    the merging visible frame to frame.

    Parameters
    ----------
    psi0 : ndarray, shape (nx, ny)
        Initial flux function, e.g. from
        :func:`physicskit.plasma.instabilities.reconnection_harris_ic`.
    eta : float
        Resistivity.
    v0 : float
        Inflow speed.
    dt : float
        Time step per :func:`~physicskit.plasma.instabilities.simulate_reconnection` sub-step.
    steps_per_frame : int
        Number of sub-steps advanced between animation frames.
    n_frames : int
        Number of animation frames.
    Lx, Ly : float
        Domain size.
    interval : int, default=60
        Delay between frames in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> from physicskit.plasma.instabilities import reconnection_harris_ic
    >>> psi0 = reconnection_harris_ic(32, 32, Lx=20.0, Ly=20.0, sheet_width=1.0, perturbation_amplitude=0.2)
    >>> anim = animate_reconnection(psi0, eta=0.02, v0=0.05, dt=0.02, steps_per_frame=5, n_frames=4, Lx=20.0, Ly=20.0)
    >>> len(list(anim.new_frame_seq()))
    4
    """
    psi = psi0
    snapshots = [psi0]
    for _ in range(n_frames - 1):
        result = simulate_reconnection(psi, eta, v0, dt, steps_per_frame, Lx, Ly)
        psi = result["psi"]
        snapshots.append(psi)
    deviations = [s - s.mean(axis=0, keepdims=True) for s in snapshots]
    return _animate_2d_snapshots(
        deviations,
        extent=(0.0, Lx, -Ly / 2.0, Ly / 2.0),
        xlabel="x",
        ylabel="y",
        title=r"Reconnection: perturbed flux $\psi-\langle\psi\rangle_x$",
        interval=interval,
    )


def animate_weibel_filamentation(
    x: np.ndarray, t: np.ndarray, wpe: float, temperature_anisotropy: float, n_modes: int = 12, seed: int = 0, interval: int = 60
) -> FuncAnimation:
    """Animate the transverse current filaments growing under the reduced quasi-linear Weibel model.

    Parameters
    ----------
    x : ndarray, shape (nx,)
        Spatial grid.
    t : ndarray, shape (nt,)
        Animation frame times.
    wpe : float
        Electron plasma frequency.
    temperature_anisotropy : float
        The ratio :math:`T_\\perp/T_\\parallel`; see
        :func:`physicskit.plasma.instabilities.weibel_growth_rate`.
    n_modes : int, default=12
        Number of seeded Fourier modes.
    seed : int, default=0
        Random seed.
    interval : int, default=60
        Delay between frames in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    See Also
    --------
    physicskit.plasma.instabilities.simulate_weibel_filamentation : Supplies the current-density snapshots animated here.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 20.0, 128, endpoint=False)
    >>> t = np.linspace(0, 5.0, 6)
    >>> anim = animate_weibel_filamentation(x, t, wpe=1.0, temperature_anisotropy=4.0, n_modes=6)
    >>> len(list(anim.new_frame_seq()))
    6
    """
    J = simulate_weibel_filamentation(x, t, wpe, temperature_anisotropy, n_modes=n_modes, seed=seed)
    snapshots = [J[i] for i in range(J.shape[0])]
    return _animate_1d_snapshots(
        x, snapshots, xlabel="x", ylabel=r"current density $J_y$ (normalized)", title="Weibel/filamentation instability", interval=interval
    )


def animate_two_stream_phase_space(
    x0: np.ndarray, v0: np.ndarray, L: float, ng: int, dt: float, steps_per_frame: int, n_frames: int, interval: int = 60
) -> FuncAnimation:
    """Animate two-stream-instability phase space :math:`(x, v)` developing its characteristic vortex.

    Repeatedly calls :func:`physicskit.plasma.kinetic.pic_simulate` for
    ``steps_per_frame`` leapfrog steps at a time, redrawing a scatter of
    every particle's position and velocity each frame -- the two initially
    separate beams of :func:`physicskit.plasma.kinetic.two_stream_ic`
    visibly wrap around each other into a single phase-space "hole" as the
    instability saturates.

    Parameters
    ----------
    x0, v0 : ndarray, shape (n_particles,)
        Initial particle positions and velocities, e.g. from
        :func:`physicskit.plasma.kinetic.two_stream_ic`.
    L : float
        Domain length.
    ng : int
        Number of grid points for the PIC field solve.
    dt : float
        Time step per leapfrog sub-step.
    steps_per_frame : int
        Number of PIC steps advanced between animation frames.
    n_frames : int
        Number of animation frames.
    interval : int, default=60
        Delay between frames in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> from physicskit.plasma.kinetic import two_stream_ic
    >>> x0, v0 = two_stream_ic(2000, L=10.0, v_drift=3.0, v_th=0.5, seed=0)
    >>> anim = animate_two_stream_phase_space(x0, v0, L=10.0, ng=32, dt=0.05, steps_per_frame=5, n_frames=4)
    >>> len(list(anim.new_frame_seq()))
    4
    """
    x, v = np.asarray(x0, dtype=float), np.asarray(v0, dtype=float)
    x_snaps, v_snaps = [x.copy()], [v.copy()]
    for _ in range(n_frames - 1):
        result = pic_simulate(x, v, L, ng, dt, steps_per_frame)
        x, v = result["x"], result["v"]
        x_snaps.append(x.copy())
        v_snaps.append(v.copy())

    fig, ax = plt.subplots()
    scatter = ax.scatter(x_snaps[0], v_snaps[0], s=2, alpha=0.5)
    ax.set_xlim(0, L)
    all_v = np.concatenate(v_snaps)
    vpad = 0.1 * (all_v.max() - all_v.min())
    ax.set_ylim(all_v.min() - vpad, all_v.max() + vpad)
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.set_title("Two-stream instability phase space")

    def update(frame):
        scatter.set_offsets(np.column_stack([x_snaps[frame], v_snaps[frame]]))
        return (scatter,)

    return FuncAnimation(fig, update, frames=len(x_snaps), interval=interval, blit=True)


def animate_langmuir_wave(
    x0: np.ndarray, v0: np.ndarray, L: float, ng: int, dt: float, steps_per_frame: int, n_frames: int, interval: int = 60
) -> FuncAnimation:
    """Animate a Langmuir wave's electron density oscillating in place at (approximately) the plasma frequency.

    Repeatedly calls :func:`physicskit.plasma.kinetic.pic_simulate` for
    ``steps_per_frame`` steps at a time and, each frame, deposits the
    current particle positions onto the grid with
    :func:`physicskit.plasma.kinetic.deposit_number_density` -- reusing the
    exact charge-assignment kernel the PIC field solve itself uses, so the
    density shown is precisely what
    :func:`physicskit.plasma.kinetic.pic_step` sees when it solves Poisson's
    equation for the field driving the next sub-step.

    Parameters
    ----------
    x0, v0 : ndarray, shape (n_particles,)
        Initial particle positions and velocities, e.g. from
        :func:`physicskit.plasma.kinetic.langmuir_wave_ic`.
    L : float
        Domain length.
    ng : int
        Number of grid points.
    dt : float
        Time step per leapfrog sub-step.
    steps_per_frame : int
        Number of PIC steps advanced between animation frames.
    n_frames : int
        Number of animation frames.
    interval : int, default=60
        Delay between frames in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.plasma.kinetic import langmuir_wave_ic
    >>> k = 2 * np.pi / 4.0
    >>> x0, v0 = langmuir_wave_ic(4000, L=4.0, k_mode=k, alpha=0.05, v_th=0.05, seed=0)
    >>> anim = animate_langmuir_wave(x0, v0, L=4.0, ng=32, dt=0.05, steps_per_frame=4, n_frames=4)
    >>> len(list(anim.new_frame_seq()))
    4
    """
    x, v = np.asarray(x0, dtype=float), np.asarray(v0, dtype=float)
    x_grid = np.linspace(0.0, L, ng, endpoint=False)
    snapshots = [deposit_number_density(x, L, ng)]
    for _ in range(n_frames - 1):
        result = pic_simulate(x, v, L, ng, dt, steps_per_frame)
        x, v = result["x"], result["v"]
        snapshots.append(deposit_number_density(x, L, ng))
    return _animate_1d_snapshots(x_grid, snapshots, xlabel="x", ylabel="electron density", title="Langmuir wave", interval=interval)


def animate_ion_acoustic_soliton(u0: np.ndarray, x: np.ndarray, dt: float, steps_per_frame: int, n_frames: int, interval: int = 60) -> FuncAnimation:
    """Animate an ion-acoustic soliton propagating without change of shape.

    Repeatedly calls :func:`physicskit.plasma.waves.ion_acoustic_soliton_evolve`
    for ``steps_per_frame`` steps at a time and animates the resulting
    density-pulse snapshots.

    Parameters
    ----------
    u0 : ndarray
        Initial soliton profile, e.g. from
        :func:`physicskit.plasma.waves.ion_acoustic_soliton_profile`.
    x : ndarray
        Uniformly spaced periodic spatial grid.
    dt : float
        Time step per sub-step.
    steps_per_frame : int
        Number of steps advanced between animation frames.
    n_frames : int
        Number of animation frames.
    interval : int, default=60
        Delay between frames in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.plasma.waves import ion_acoustic_soliton_profile
    >>> x = np.linspace(-30, 30, 256, endpoint=False)
    >>> u0 = ion_acoustic_soliton_profile(x, speed=4.0, x0=-15.0)
    >>> anim = animate_ion_acoustic_soliton(u0, x, dt=0.0005, steps_per_frame=200, n_frames=4)
    >>> len(list(anim.new_frame_seq()))
    4
    """
    u = np.asarray(u0, dtype=float)
    snapshots = [u.copy()]
    for _ in range(n_frames - 1):
        u = ion_acoustic_soliton_evolve(u, x, dt, steps_per_frame)
        snapshots.append(u.copy())
    return _animate_1d_snapshots(x, snapshots, xlabel="x", ylabel="density perturbation", title="Ion-acoustic soliton", interval=interval)


def animate_drift_wave_turbulence(
    phi0: np.ndarray, dt: float, steps_per_frame: int, n_frames: int, length: float, nu: float = 0.03, interval: int = 60
) -> FuncAnimation:
    """Animate a Hasegawa-Mima potential field developing turbulent structure from small-amplitude noise.

    Repeatedly calls
    :func:`physicskit.plasma.turbulence.simulate_hasegawa_mima` for
    ``steps_per_frame`` steps at a time and animates the resulting
    potential-field snapshots.

    Parameters
    ----------
    phi0 : ndarray, shape (n, n)
        Initial potential field, e.g. from
        :func:`physicskit.plasma.turbulence.drift_wave_noise_ic`.
    dt : float
        Time step per sub-step.
    steps_per_frame : int
        Number of RK4 steps advanced between animation frames.
    n_frames : int
        Number of animation frames.
    length : float
        Physical domain size.
    nu : float, default=0.03
        Dissipation coefficient (see :func:`physicskit.plasma.turbulence.simulate_hasegawa_mima`).
    interval : int, default=60
        Delay between frames in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.plasma.turbulence import drift_wave_noise_ic
    >>> phi0 = drift_wave_noise_ic(48, 2 * np.pi, amplitude=0.05, seed=0)
    >>> anim = animate_drift_wave_turbulence(phi0, dt=0.02, steps_per_frame=10, n_frames=4, length=2 * np.pi)
    >>> len(list(anim.new_frame_seq()))
    4
    """
    phi = np.asarray(phi0, dtype=float)
    snapshots = [phi.copy()]
    for _ in range(n_frames - 1):
        result = simulate_hasegawa_mima(phi, dt, steps_per_frame, length, nu=nu)
        phi = result["phi"]
        snapshots.append(phi.copy())
    return _animate_2d_snapshots(snapshots, extent=(0.0, length, 0.0, length), title="Drift-wave turbulence (Hasegawa-Mima)", interval=interval)


def animate_alfven_wave(
    By0: np.ndarray,
    vy0: np.ndarray,
    x: np.ndarray,
    dt: float,
    steps_per_frame: int,
    n_frames: int,
    B0: float,
    rho0: float,
    mu0: float = 1.0,
    interval: int = 60,
) -> FuncAnimation:
    """Animate a transverse Alfven-wave pulse propagating (and splitting) along the background field.

    Repeatedly calls :func:`physicskit.plasma.waves.simulate_alfven_wave`
    for ``steps_per_frame`` steps at a time and animates the resulting
    transverse-field snapshots.

    Parameters
    ----------
    By0, vy0 : ndarray
        Initial transverse field and velocity perturbations, e.g. from
        :func:`physicskit.plasma.waves.alfven_wave_pulse_ic`.
    x : ndarray
        Uniformly spaced periodic spatial grid.
    dt : float
        Time step per sub-step.
    steps_per_frame : int
        Number of RK4 steps advanced between animation frames.
    n_frames : int
        Number of animation frames.
    B0 : float
        Background field magnitude.
    rho0 : float
        Background mass density.
    mu0 : float, default=1.0
        Vacuum permeability (normalized units by default; see
        :func:`physicskit.plasma.waves.simulate_alfven_wave`).
    interval : int, default=60
        Delay between frames in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.plasma.waves import alfven_wave_pulse_ic
    >>> x = np.linspace(-20, 20, 256, endpoint=False)
    >>> By0, vy0 = alfven_wave_pulse_ic(x, x0=0.0, width=1.0, amplitude=0.1)
    >>> anim = animate_alfven_wave(By0, vy0, x, dt=0.002, steps_per_frame=200, n_frames=4, B0=1.0, rho0=1.0)
    >>> len(list(anim.new_frame_seq()))
    4
    """
    By, vy = np.asarray(By0, dtype=float), np.asarray(vy0, dtype=float)
    snapshots = [By.copy()]
    for _ in range(n_frames - 1):
        result = simulate_alfven_wave(By, vy, x, dt, steps_per_frame, B0, rho0, mu0=mu0)
        By, vy = result["By"], result["vy"]
        snapshots.append(By.copy())
    return _animate_1d_snapshots(x, snapshots, xlabel="x", ylabel=r"$B_y$ perturbation", title="Alfven wave", interval=interval)


def animate_wakefield_acceleration(
    x0: float, v0: float, q: float, m: float, E0: float, k: float, v_phase: float, dt: float, steps: int, frame_stride: int = 20, interval: int = 60
) -> FuncAnimation:
    """Animate a test charge surfing a prescribed traveling wakefield, with an energy-gain trace inset.

    Runs :func:`physicskit.plasma.acceleration.simulate_wakefield_acceleration`
    once for the full trajectory, then animates a snapshot of the wakefield
    :math:`E_z(x, t)` with the particle's position marked on it, alongside
    a running plot of its kinetic energy -- showing both the spatial
    "surfing" picture and the resulting energy gain simultaneously.

    Parameters
    ----------
    x0, v0 : float
        Initial particle position and velocity.
    q, m : float
        Particle charge and mass.
    E0, k, v_phase : float
        Wakefield amplitude, wavenumber, and phase velocity, as in
        :func:`physicskit.plasma.acceleration.wakefield_e_field`.
    dt : float
        Time step.
    steps : int
        Total number of Boris-pusher steps to integrate.
    frame_stride : int, default=20
        Number of integration steps between animation frames.
    interval : int, default=60
        Delay between frames in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> anim = animate_wakefield_acceleration(
    ...     x0=0.0, v0=0.9, q=1.0, m=1.0, E0=0.05, k=1.0, v_phase=1.0, dt=0.01, steps=400, frame_stride=40
    ... )
    >>> len(list(anim.new_frame_seq()))
    11
    """
    result = simulate_wakefield_acceleration(x0, v0, q, m, E0, k, v_phase, dt, steps)
    t_hist, x_hist, kinetic_energy = result["t"], result["x"], result["kinetic_energy"]
    frame_idx = np.arange(0, len(t_hist), frame_stride)

    x_span = max(abs(x_hist.max() - x_hist.min()), 1.0)
    x_field = np.linspace(x_hist.min() - 0.2 * x_span, x_hist.max() + 0.2 * x_span, 400)

    fig, (ax_field, ax_energy) = plt.subplots(2, 1, figsize=(6, 6))
    (field_line,) = ax_field.plot(x_field, wakefield_e_field(x_field, t_hist[0], E0, k, v_phase))
    (particle_marker,) = ax_field.plot([x_hist[0]], [wakefield_e_field(x_hist[0], t_hist[0], E0, k, v_phase)], "o", color="red", markersize=8)
    ax_field.set_xlabel("x")
    ax_field.set_ylabel(r"$E_z$")
    ax_field.set_title("Wakefield acceleration")

    (energy_line,) = ax_energy.plot([], [])
    ax_energy.set_xlim(t_hist[0], t_hist[-1])
    ax_energy.set_ylim(0.0, 1.1 * kinetic_energy.max())
    ax_energy.set_xlabel("t")
    ax_energy.set_ylabel("kinetic energy")

    def update(i):
        idx = frame_idx[i]
        field_line.set_ydata(wakefield_e_field(x_field, t_hist[idx], E0, k, v_phase))
        particle_marker.set_data([x_hist[idx]], [wakefield_e_field(x_hist[idx], t_hist[idx], E0, k, v_phase)])
        energy_line.set_data(t_hist[: idx + 1], kinetic_energy[: idx + 1])
        return field_line, particle_marker, energy_line

    return FuncAnimation(fig, update, frames=len(frame_idx), interval=interval, blit=True)
