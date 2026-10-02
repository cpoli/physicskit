"""Linear acoustics: sound waves in pipes and rooms, and their standing modes.

Small disturbances of a fluid at rest obey the linearized Euler equations

.. math::

    \\frac{\\partial p}{\\partial t} = -\\rho_0 c^2\\,\\nabla\\cdot\\mathbf u, \\qquad
    \\rho_0\\frac{\\partial \\mathbf u}{\\partial t} = -\\nabla p,

which combine into the wave equation :math:`\\partial_t^2 p = c^2\\nabla^2 p`
for the acoustic pressure :math:`p` (Rayleigh, *The Theory of Sound*,
1877-1878; Kinsler et al., *Fundamentals of Acoustics*, 4th ed., ch. 5).
:func:`acoustic_wave_1d` and :func:`acoustic_wave_2d` integrate the
first-order pair on a staggered grid (pressure at cell centers, velocity
on cell faces) with leapfrog time stepping, the acoustic analogue of Yee's
FDTD scheme. It is second order, conserves a discrete acoustic energy, and
is stable for :math:`c\\,\\Delta t\\sqrt{\\sum_a \\Delta x_a^{-2}} \\le 1`.

A rigid wall (a closed pipe end) forces :math:`u_n = 0` and is a pressure
antinode. An open pipe end is idealized as :math:`p = 0`, a pressure node
(ignoring the end correction of about :math:`0.6` radii). The standing
modes of a pipe of length :math:`L` then have frequencies

.. math::

    f_n = \\frac{nc}{2L} \\;\\;\\text{(both ends open or both closed)}, \\qquad
    f_n = \\frac{(2n-1)c}{4L} \\;\\;\\text{(one closed, one open)},

given by :func:`pipe_mode_frequencies`.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from physicskit.fluids.exceptions import InvalidParameterError

__all__ = [
    "ideal_gas_sound_speed",
    "pipe_mode_frequencies",
    "pipe_mode_shape",
    "rectangular_room_mode_frequencies",
    "acoustic_wave_1d",
    "acoustic_wave_2d",
]

_ENDS = ("closed", "open")


def ideal_gas_sound_speed(gamma: float, pressure: float, density: float) -> float:
    """Adiabatic sound speed of an ideal gas, :math:`c = \\sqrt{\\gamma p/\\rho}`.

    Laplace's (1816) correction to Newton's isothermal :math:`\\sqrt{p/\\rho}`,
    which had underestimated the measured speed by about 15%.

    Parameters
    ----------
    gamma : float
        Ratio of specific heats (1.4 for air).
    pressure : float
        Ambient pressure in Pa.
    density : float
        Ambient density in kg/m^3.

    Returns
    -------
    float
        Sound speed in m/s.

    Examples
    --------
    >>> round(ideal_gas_sound_speed(1.4, 101325.0, 1.204), 1)  # air at 20 C
    343.2
    """
    if gamma <= 0 or pressure <= 0 or density <= 0:
        raise InvalidParameterError("gamma, pressure and density must be positive")
    return float(np.sqrt(gamma * pressure / density))


def _check_ends(ends) -> tuple[str, str]:
    left, right = ends
    if left not in _ENDS or right not in _ENDS:
        raise InvalidParameterError(f"each end must be 'closed' or 'open', got {ends}")
    return left, right


def pipe_mode_frequencies(length: float, c: float, n_modes: int = 5, ends=("open", "open")) -> NDArray[np.float64]:
    """Standing-wave frequencies of a pipe with closed (rigid) or open (pressure-release) ends.

    Parameters
    ----------
    length : float
        Pipe length :math:`L`.
    c : float
        Sound speed.
    n_modes : int, default=5
        Number of modes, lowest first.
    ends : tuple of {"closed", "open"}, default=("open", "open")
        Boundary condition at each end.

    Returns
    -------
    ndarray, shape (n_modes,)
        :math:`f_n = nc/2L` for like ends and :math:`(2n-1)c/4L` for unlike
        ends. (A pipe closed at both ends also has a trivial :math:`f = 0`
        uniform-pressure mode, which is not listed.)

    Examples
    --------
    >>> pipe_mode_frequencies(1.0, 340.0, 3, ends=("closed", "open")).tolist()
    [85.0, 255.0, 425.0]
    """
    left, right = _check_ends(ends)
    if length <= 0 or c <= 0:
        raise InvalidParameterError("length and c must be positive")
    n = np.arange(1, n_modes + 1, dtype=np.float64)
    if left == right:
        return n * c / (2.0 * length)
    return (2 * n - 1) * c / (4.0 * length)


def pipe_mode_shape(x: ArrayLike, length: float, mode: int, ends=("open", "open")) -> NDArray[np.float64]:
    """Pressure profile of a pipe's ``mode``-th standing wave (unit amplitude).

    Parameters
    ----------
    x : array_like
        Position along the pipe, ``0 <= x <= length``.
    length : float
        Pipe length.
    mode : int
        Mode number, starting at 1 (the order of :func:`pipe_mode_frequencies`).
    ends : tuple of {"closed", "open"}, default=("open", "open")
        Boundary condition at each end.

    Returns
    -------
    ndarray
        :math:`\\sin(n\\pi x/L)` (open-open), :math:`\\cos(n\\pi x/L)`
        (closed-closed), :math:`\\cos((2n-1)\\pi x/2L)` (closed-open), or
        :math:`\\sin((2n-1)\\pi x/2L)` (open-closed).

    Examples
    --------
    >>> pipe_mode_shape([0.0, 0.5, 1.0], 1.0, 1, ends=("closed", "closed")).round(12).tolist()
    [1.0, 0.0, -1.0]
    """
    left, right = _check_ends(ends)
    x = np.asarray(x, dtype=np.float64)
    if left == right:
        k = mode * np.pi / length
        return np.cos(k * x) if left == "closed" else np.sin(k * x)
    k = (2 * mode - 1) * np.pi / (2.0 * length)
    return np.cos(k * x) if left == "closed" else np.sin(k * x)


def rectangular_room_mode_frequencies(Lx: float, Ly: float, c: float, n_max: int = 3) -> NDArray[np.float64]:
    """Mode frequencies of a rigid-walled rectangular room (2D).

    .. math::

        f_{mn} = \\frac{c}{2}\\sqrt{\\left(\\frac{m}{L_x}\\right)^2 + \\left(\\frac{n}{L_y}\\right)^2}

    with pressure :math:`\\cos(m\\pi x/L_x)\\cos(n\\pi y/L_y)` (Kinsler et al.,
    §9.2).

    Parameters
    ----------
    Lx, Ly : float
        Room dimensions.
    c : float
        Sound speed.
    n_max : int, default=3
        Largest mode index along each axis.

    Returns
    -------
    ndarray, shape (k, 3)
        Rows ``(m, n, f_mn)`` sorted by frequency, excluding ``(0, 0)``.

    Examples
    --------
    >>> rectangular_room_mode_frequencies(4.0, 3.0, 340.0, n_max=1)[:2].tolist()
    [[1.0, 0.0, 42.5], [0.0, 1.0, 56.666666666666664]]
    """
    rows = []
    for m in range(n_max + 1):
        for n in range(n_max + 1):
            if m == 0 and n == 0:
                continue
            rows.append((m, n, 0.5 * c * np.hypot(m / Lx, n / Ly)))
    out = np.array(rows, dtype=np.float64)
    return out[np.argsort(out[:, 2], kind="stable")]


@njit(cache=True)
def _leapfrog_1d(p, u, n_steps, kp, ku, open_left, open_right, record_every, probe):
    n = p.shape[0]
    n_rec = n_steps // record_every
    frames = np.empty((n_rec + 1, n))
    frames[0] = p
    trace = np.empty(n_steps + 1)
    trace[0] = p[probe]
    k = 1
    for step in range(n_steps):
        # velocity on interior faces
        for i in range(1, n):
            u[i] -= ku * (p[i] - p[i - 1])
        # boundary faces: rigid (u = 0) or pressure release (p = 0 at the face,
        # via a mirrored ghost cell p_ghost = -p_edge)
        if open_left:
            u[0] -= ku * (p[0] + p[0])
        else:
            u[0] = 0.0
        if open_right:
            u[n] -= ku * (-p[n - 1] - p[n - 1])
        else:
            u[n] = 0.0
        for i in range(n):
            p[i] -= kp * (u[i + 1] - u[i])
        trace[step + 1] = p[probe]
        if (step + 1) % record_every == 0:
            frames[k] = p
            k += 1
    return frames, trace


def acoustic_wave_1d(
    p0: ArrayLike,
    length: float,
    c: float,
    t_max: float,
    dt: float | None = None,
    ends=("closed", "closed"),
    rho0: float = 1.0,
    u0: ArrayLike | None = None,
    n_frames: int = 100,
    probe: int | None = None,
):
    """Sound in a 1D pipe: staggered-grid leapfrog for the linearized Euler equations.

    The pipe :math:`0 \\le x \\le L` is split into ``n = len(p0)`` cells;
    pressure lives at cell centers :math:`x_i = (i + \\tfrac12)\\Delta x` and
    velocity on the ``n + 1`` faces. Each step updates

    .. math::

        u_{i}^{n+1/2} = u_i^{n-1/2} - \\frac{\\Delta t}{\\rho_0\\Delta x}(p_i^n - p_{i-1}^n),
        \\qquad
        p_i^{n+1} = p_i^n - \\frac{\\rho_0 c^2\\Delta t}{\\Delta x}(u_{i+1}^{n+1/2} - u_i^{n+1/2}).

    Parameters
    ----------
    p0 : array_like, shape (n,)
        Initial acoustic pressure at the cell centers.
    length : float
        Pipe length.
    c : float
        Sound speed.
    t_max : float
        Final time, rounded to a whole number of steps.
    dt : float, optional
        Time step; defaults to ``0.5 * dx / c``. Must satisfy
        ``c * dt <= dx``.
    ends : tuple of {"closed", "open"}, default=("closed", "closed")
        Rigid (``u = 0``) or pressure-release (``p = 0``) at each end.
    rho0 : float, default=1.0
        Ambient density.
    u0 : array_like, shape (n + 1,), optional
        Initial face velocities (at :math:`t = -\\Delta t/2`); zero by default.
    n_frames : int, default=100
        Approximate number of recorded pressure frames.
    probe : int, optional
        Cell whose pressure is recorded at every step (for spectra);
        defaults to cell 0.

    Returns
    -------
    times : ndarray, shape (k + 1,)
        Frame times.
    frames : ndarray, shape (k + 1, n)
        Pressure at each frame.
    trace : ndarray, shape (n_steps + 1,)
        Pressure at ``probe`` after every step, sampled every ``dt``.
    dt : float
        The time step used.

    Examples
    --------
    The fundamental of a closed-closed pipe returns to itself after one
    period :math:`2L/c`:

    >>> x = (np.arange(200) + 0.5) / 200
    >>> t, P, trace, dt = acoustic_wave_1d(np.cos(np.pi * x), 1.0, 1.0, t_max=2.0, dt=0.0025, n_frames=1)
    >>> bool(np.max(np.abs(P[-1] - P[0])) < 1e-3)
    True
    """
    left, right = _check_ends(ends)
    p = np.array(p0, dtype=np.float64)
    n = p.size
    if length <= 0 or c <= 0 or rho0 <= 0:
        raise InvalidParameterError("length, c and rho0 must be positive")
    dx = length / n
    if dt is None:
        dt = 0.5 * dx / c
    if c * dt > dx * (1 + 1e-12):
        raise InvalidParameterError(f"Courant number c*dt/dx = {c * dt / dx:.3g} exceeds 1")
    u = np.zeros(n + 1) if u0 is None else np.array(u0, dtype=np.float64)
    n_steps = int(round(t_max / dt))
    record_every = max(1, n_steps // max(n_frames, 1))
    frames, trace = _leapfrog_1d(
        p, u, n_steps, rho0 * c**2 * dt / dx, dt / (rho0 * dx), left == "open", right == "open", record_every, 0 if probe is None else int(probe)
    )
    times = np.arange(frames.shape[0]) * record_every * dt
    return times, frames, trace, dt


@njit(cache=True)
def _leapfrog_2d(p, ux, uy, n_steps, kp_x, kp_y, ku_x, ku_y, record_every, px, py):
    nx, ny = p.shape
    n_rec = n_steps // record_every
    frames = np.empty((n_rec + 1, nx, ny))
    frames[0] = p
    trace = np.empty(n_steps + 1)
    trace[0] = p[px, py]
    k = 1
    for step in range(n_steps):
        for i in range(1, nx):
            for j in range(ny):
                ux[i, j] -= ku_x * (p[i, j] - p[i - 1, j])
        for i in range(nx):
            for j in range(1, ny):
                uy[i, j] -= ku_y * (p[i, j] - p[i, j - 1])
        for i in range(nx):
            for j in range(ny):
                p[i, j] -= kp_x * (ux[i + 1, j] - ux[i, j]) + kp_y * (uy[i, j + 1] - uy[i, j])
        trace[step + 1] = p[px, py]
        if (step + 1) % record_every == 0:
            frames[k] = p
            k += 1
    return frames, trace


def acoustic_wave_2d(
    p0: ArrayLike,
    size,
    c: float,
    t_max: float,
    dt: float | None = None,
    rho0: float = 1.0,
    n_frames: int = 100,
    probe=(0, 0),
):
    """Sound in a rigid-walled rectangular room: 2D staggered-grid leapfrog.

    The 2D version of :func:`acoustic_wave_1d`, with pressure at cell
    centers, :math:`u_x` on the x-faces and :math:`u_y` on the y-faces, and
    zero normal velocity on all four walls. Its standing modes are those of
    :func:`rectangular_room_mode_frequencies`.

    Parameters
    ----------
    p0 : array_like, shape (nx, ny)
        Initial pressure at cell centers.
    size : tuple of float
        Room dimensions ``(Lx, Ly)``.
    c : float
        Sound speed.
    t_max : float
        Final time.
    dt : float, optional
        Time step; defaults to half the stability limit
        ``1 / (c * sqrt(1/dx^2 + 1/dy^2))``.
    rho0 : float, default=1.0
        Ambient density.
    n_frames : int, default=100
        Approximate number of recorded frames.
    probe : tuple of int, default=(0, 0)
        Cell whose pressure is recorded at every step.

    Returns
    -------
    times : ndarray
    frames : ndarray, shape (k + 1, nx, ny)
    trace : ndarray, shape (n_steps + 1,)
    dt : float

    Examples
    --------
    >>> p0 = np.zeros((40, 30))
    >>> p0[20, 15] = 1.0
    >>> t, P, trace, dt = acoustic_wave_2d(p0, (4.0, 3.0), 340.0, t_max=0.01, n_frames=5)
    >>> P.shape[1:], bool(abs(P[-1].sum() - 1.0) < 1e-9)  # rigid walls conserve the mean pressure
    ((40, 30), True)
    """
    p = np.array(p0, dtype=np.float64)
    nx, ny = p.shape
    Lx, Ly = size
    if Lx <= 0 or Ly <= 0 or c <= 0 or rho0 <= 0:
        raise InvalidParameterError("size, c and rho0 must be positive")
    dx, dy = Lx / nx, Ly / ny
    limit = 1.0 / (c * np.sqrt(1.0 / dx**2 + 1.0 / dy**2))
    if dt is None:
        dt = 0.5 * limit
    if dt > limit * (1 + 1e-12):
        raise InvalidParameterError(f"dt = {dt:.3g} exceeds the stability limit {limit:.3g}")
    ux = np.zeros((nx + 1, ny))
    uy = np.zeros((nx, ny + 1))
    n_steps = int(round(t_max / dt))
    record_every = max(1, n_steps // max(n_frames, 1))
    k = rho0 * c**2 * dt
    frames, trace = _leapfrog_2d(p, ux, uy, n_steps, k / dx, k / dy, dt / (rho0 * dx), dt / (rho0 * dy), record_every, int(probe[0]), int(probe[1]))
    return np.arange(frames.shape[0]) * record_every * dt, frames, trace, dt
