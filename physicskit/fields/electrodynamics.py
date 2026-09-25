"""Finite-difference time-domain (FDTD) solutions of Maxwell's equations on a Yee grid.

Implements the staggered-grid (Yee 1966) leapfrog update for 1D plane waves
and 2D TMz-mode propagation, plus a graded-conductivity absorbing boundary
(a simplified, non-split-field approximation to Berenger's Perfectly
Matched Layer) that damps outgoing waves instead of reflecting them off a
hard electric wall.

Units are SI throughout; :data:`EPS0`, :data:`MU0`, :data:`C0` are the
vacuum permittivity, permeability, and speed of light.
"""

from __future__ import annotations

import numpy as np

from physicskit import constants as _const

__all__ = [
    "EPS0",
    "MU0",
    "C0",
    "courant_limit_1d",
    "courant_limit_2d",
    "fdtd_1d",
    "fdtd_2d_tmz",
    "fdtd_2d_tmz_evolve",
    "pml_conductivity_profile",
    "pml_conductivity_profile_2d",
    "oscillating_dipole_source",
    "dielectric_slab",
    "tmz_cavity_mode",
    "poynting_vector_tmz",
    "flux_tube_field_1d",
    "flux_tube_energy_density_2d",
]

EPS0 = _const.VACUUM_PERMITTIVITY
MU0 = _const.VACUUM_PERMEABILITY
C0 = _const.C


def courant_limit_1d(dx: float) -> float:
    """Maximum stable time step for the 1D FDTD update.

    Parameters
    ----------
    dx : float
        Spatial grid spacing in meters.

    Returns
    -------
    float
        The Courant stability limit :math:`\\Delta t = \\Delta x / c_0`.

    Examples
    --------
    >>> round(float(courant_limit_1d(1e-3) * C0 / 1e-3), 6)
    1.0
    """
    return dx / C0


def courant_limit_2d(dx: float, dy: float) -> float:
    """Maximum stable time step for the 2D FDTD update.

    Parameters
    ----------
    dx, dy : float
        Spatial grid spacing in meters along each axis.

    Returns
    -------
    float
        The Courant stability limit
        :math:`\\Delta t = 1 / (c_0 \\sqrt{1/\\Delta x^2 + 1/\\Delta y^2})`.

    Examples
    --------
    >>> dt = courant_limit_2d(1e-3, 1e-3)
    >>> round(float(dt * C0 / 1e-3), 6)
    0.707107
    """
    return 1.0 / (C0 * np.sqrt(1.0 / dx**2 + 1.0 / dy**2))


def pml_conductivity_profile(n_cells: int, pml_width: int, dx: float, order: int = 3, sigma_max: float | None = None) -> np.ndarray:
    """Build a polynomial-graded electric conductivity profile absorbing at both ends of a 1D grid.

    A simplified (non-split-field) approximation to Berenger's Perfectly
    Matched Layer: rather than solving auxiliary split-field equations,
    this grades the ordinary medium's electric conductivity smoothly up
    from zero over ``pml_width`` cells at each boundary, damping outgoing
    waves via the standard lossy-dielectric FDTD update
    (see ``sigma`` in :func:`fdtd_1d`) instead of reflecting them off a
    hard wall.

    Parameters
    ----------
    n_cells : int
        Total number of grid points.
    pml_width : int
        Number of cells over which the conductivity ramps from 0 to ``sigma_max``.
    dx : float
        Spatial grid spacing in meters, used to scale the default ``sigma_max``.
    order : int, default=3
        Polynomial grading order (Sadiku's rule of thumb uses a cubic ramp).
    sigma_max : float, optional
        Peak conductivity at the outermost cell. Defaults to the standard
        empirical optimum ``(order + 1) / (150 * pi * dx)``.

    Returns
    -------
    ndarray, shape (n_cells,)
        Conductivity profile, zero in the interior and ramping up at both edges.

    Examples
    --------
    >>> sigma = pml_conductivity_profile(200, pml_width=20, dx=1e-3)
    >>> bool(np.all(sigma[20:-20] == 0))
    True
    >>> bool(sigma[0] > sigma[10] > 0)
    True
    """
    if sigma_max is None:
        sigma_max = (order + 1) / (150 * np.pi * dx)
    sigma = np.zeros(n_cells)
    ramp = (np.arange(pml_width) / pml_width) ** order * sigma_max
    sigma[:pml_width] = ramp[::-1]
    sigma[-pml_width:] = ramp
    return sigma


def pml_conductivity_profile_2d(shape: tuple, pml_width: int, dx: float, dy: float, order: int = 3, sigma_max: float | None = None) -> np.ndarray:
    """Build a 2D graded-conductivity absorbing boundary by combining two 1D PML profiles.

    Reuses :func:`pml_conductivity_profile` along each axis and takes the
    pointwise maximum, so a cell absorbs at whichever rate is larger --
    correct in the corners, where both the ``x`` and ``y`` ramps are active.

    Parameters
    ----------
    shape : tuple(int, int)
        ``(Nx, Ny)`` grid shape.
    pml_width : int
        Number of cells over which each axis's conductivity ramps up.
    dx, dy : float
        Spatial grid spacing along each axis.
    order : int, default=3
        Polynomial grading order, passed to :func:`pml_conductivity_profile`.
    sigma_max : float, optional
        Peak conductivity; defaults (per axis) to the standard empirical optimum.

    Returns
    -------
    ndarray, shape (Nx, Ny)
        Conductivity map, zero in the interior and ramping up at all four edges.

    Examples
    --------
    >>> sigma = pml_conductivity_profile_2d((100, 80), pml_width=10, dx=1e-3, dy=1e-3)
    >>> bool(np.all(sigma[10:-10, 10:-10] == 0))
    True
    >>> bool(sigma[0, 0] > 0)
    True
    """
    Nx, Ny = shape
    sx = pml_conductivity_profile(Nx, pml_width, dx, order, sigma_max)
    sy = pml_conductivity_profile(Ny, pml_width, dy, order, sigma_max)
    return np.maximum(sx[:, None], sy[None, :])


def fdtd_1d(Ez0: np.ndarray, Hy0: np.ndarray, eps_r: np.ndarray, mu_r: np.ndarray, steps: int, dt: float, dx: float, sigma: np.ndarray | None = None) -> tuple:
    """Evolve a 1D plane wave (Ez, Hy) on a Yee grid using leapfrog FDTD.

    Discretizes Maxwell's curl equations for fields varying only along
    ``x``, :math:`\\partial_t H_y = \\partial_x E_z/\\mu` and
    :math:`\\partial_t E_z = \\partial_x H_y/\\epsilon` (the same signs as
    :func:`fdtd_2d_tmz`), so a wave moving toward ``+x`` has
    :math:`H_y = -E_z/\\eta` (Poynting vector :math:`-E_zH_y>0`).

    Parameters
    ----------
    Ez0 : ndarray, shape (N,)
        Initial electric field.
    Hy0 : ndarray, shape (N-1,)
        Initial magnetic field, staggered half a cell ahead of ``Ez0``.
    eps_r, mu_r : ndarray, shape (N,)
        Relative permittivity and permeability at each ``Ez`` grid point.
    steps : int
        Number of time steps to advance.
    dt : float
        Time step in seconds; must satisfy :func:`courant_limit_1d`.
    dx : float
        Spatial grid spacing in meters.
    sigma : ndarray, shape (N,), optional
        Electric conductivity (S/m) at each grid point, e.g. from
        :func:`pml_conductivity_profile`, for a lossy/absorbing boundary.
        Defaults to a lossless grid with hard (PEC) walls at both ends.

    Returns
    -------
    Ez, Hy : ndarray
        Fields at the final time step, same shapes as ``Ez0``, ``Hy0``.

    See Also
    --------
    fdtd_2d_tmz : The 2D (TMz-mode) analog of this solver.

    Examples
    --------
    A smooth Gaussian pulse, launched with the impedance-matched initial
    condition of a purely right-moving wave (:math:`H_y=-E_z/\\eta_0`),
    propagates at exactly the vacuum speed of light:

    >>> import numpy as np
    >>> N = 800
    >>> dx = 1e-3
    >>> dt = 0.99 * courant_limit_1d(dx)
    >>> eta0 = np.sqrt(MU0 / EPS0)
    >>> x0, sigma_pulse = 100, 25
    >>> Ez0 = np.exp(-((np.arange(N) - x0) ** 2) / (2 * sigma_pulse ** 2))
    >>> xh = np.arange(N - 1) + 0.5
    >>> Hy0 = -np.exp(-((xh - x0) ** 2) / (2 * sigma_pulse ** 2)) / eta0
    >>> eps_r, mu_r = np.ones(N), np.ones(N)
    >>> Ez, Hy = fdtd_1d(Ez0, Hy0, eps_r, mu_r, steps=360, dt=dt, dx=dx)
    >>> peak = np.argmax(Ez)
    >>> measured_speed = (peak - x0) * dx / (360 * dt)
    >>> round(float(measured_speed / C0), 3)
    1.002
    """
    Ez = np.array(Ez0, dtype=float, copy=True)
    Hy = np.array(Hy0, dtype=float, copy=True)
    if sigma is None:
        Ca = np.ones_like(Ez)
        Cb = dt / (EPS0 * eps_r * dx)
    else:
        loss = sigma * dt / (2 * EPS0 * eps_r)
        Ca = (1 - loss) / (1 + loss)
        Cb = (dt / (EPS0 * eps_r * dx)) / (1 + loss)
    for _ in range(steps):
        Hy += dt / (MU0 * mu_r[:-1] * dx) * (Ez[1:] - Ez[:-1])
        Ez[1:-1] = Ca[1:-1] * Ez[1:-1] + Cb[1:-1] * (Hy[1:] - Hy[:-1])
        Ez[0] = 0.0
        Ez[-1] = 0.0
    return Ez, Hy


def fdtd_2d_tmz(Ez0: np.ndarray, Hx0: np.ndarray, Hy0: np.ndarray, eps_r: np.ndarray, mu_r: np.ndarray, steps: int, dt: float, dx: float, dy: float) -> tuple:
    """Evolve a 2D TMz-mode field (Ez, Hx, Hy) on a Yee grid using leapfrog FDTD.

    Parameters
    ----------
    Ez0 : ndarray, shape (Nx, Ny)
        Initial (out-of-plane) electric field.
    Hx0, Hy0 : ndarray, shape (Nx, Ny)
        Initial in-plane magnetic field components.
    eps_r, mu_r : ndarray, shape (Nx, Ny)
        Relative permittivity and permeability maps.
    steps : int
        Number of time steps to advance.
    dt : float
        Time step in seconds; must satisfy :func:`courant_limit_2d`.
    dx, dy : float
        Spatial grid spacing in meters along each axis.

    Returns
    -------
    Ez, Hx, Hy : ndarray, shape (Nx, Ny)
        Fields at the final time step.

    See Also
    --------
    fdtd_1d : The 1D analog of this solver.
    poynting_vector_tmz : Energy flux from the evolved fields.

    Examples
    --------
    >>> import numpy as np
    >>> Nx, Ny = 50, 50
    >>> Ez0 = np.zeros((Nx, Ny))
    >>> Hx0 = np.zeros((Nx, Ny))
    >>> Hy0 = np.zeros((Nx, Ny))
    >>> eps_r, mu_r = np.ones((Nx, Ny)), np.ones((Nx, Ny))
    >>> dx = dy = 1e-3
    >>> dt = 0.5 * courant_limit_2d(dx, dy)
    >>> Ez0[Nx // 2, Ny // 2] = 1.0
    >>> Ez, Hx, Hy = fdtd_2d_tmz(Ez0, Hx0, Hy0, eps_r, mu_r, steps=10, dt=dt, dx=dx, dy=dy)
    >>> Ez.shape
    (50, 50)
    """
    Ez = np.array(Ez0, dtype=float, copy=True)
    Hx = np.array(Hx0, dtype=float, copy=True)
    Hy = np.array(Hy0, dtype=float, copy=True)
    for _ in range(steps):
        _fdtd_2d_tmz_step(Ez, Hx, Hy, eps_r, mu_r, dt, dx, dy)
    return Ez, Hx, Hy


def _fdtd_2d_tmz_step(Ez, Hx, Hy, eps_r, mu_r, dt, dx, dy, Ca=None, Cb=None):
    """One leapfrog TMz update, in place; the shared inner loop of :func:`fdtd_2d_tmz` and :func:`fdtd_2d_tmz_evolve`.

    With ``Ca``/``Cb`` left as ``None`` this is exactly the lossless,
    hard-wall (PEC) update. Passing loss coefficients (as built from a
    conductivity map by :func:`fdtd_2d_tmz_evolve`) instead applies the
    same lossy-dielectric update used by :func:`fdtd_1d`, generalized to 2D.
    """
    Hx[:, :-1] += -dt / (MU0 * mu_r[:, :-1] * dy) * (Ez[:, 1:] - Ez[:, :-1])
    Hy[:-1, :] += dt / (MU0 * mu_r[:-1, :] * dx) * (Ez[1:, :] - Ez[:-1, :])
    curl_h = (Hy[1:-1, 1:-1] - Hy[:-2, 1:-1]) / dx - (Hx[1:-1, 1:-1] - Hx[1:-1, :-2]) / dy
    if Ca is None:
        Ez[1:-1, 1:-1] += dt / (EPS0 * eps_r[1:-1, 1:-1]) * curl_h
    else:
        Ez[1:-1, 1:-1] = Ca[1:-1, 1:-1] * Ez[1:-1, 1:-1] + Cb[1:-1, 1:-1] * curl_h
    Ez[0, :] = 0.0
    Ez[-1, :] = 0.0
    Ez[:, 0] = 0.0
    Ez[:, -1] = 0.0


def fdtd_2d_tmz_evolve(
    Ez0: np.ndarray,
    Hx0: np.ndarray,
    Hy0: np.ndarray,
    eps_r: np.ndarray,
    mu_r: np.ndarray,
    steps: int,
    dt: float,
    dx: float,
    dy: float,
    sigma: np.ndarray | None = None,
    source=None,
    snapshot_stride: int = 1,
) -> tuple:
    """Time-step the 2D TMz FDTD update while recording ``Ez`` snapshots, for animation.

    Unlike :func:`fdtd_2d_tmz`, which returns only the field at the final
    step, this driver records ``Ez`` every ``snapshot_stride`` steps and
    supports two features layered on top of the same base update: a lossy
    absorbing boundary (``sigma``, e.g. from :func:`pml_conductivity_profile_2d`,
    so an outgoing wave is damped instead of reflecting off the hard PEC
    walls) and a soft source callback injected into ``Ez`` after every
    field update (e.g. :func:`oscillating_dipole_source`).

    Parameters
    ----------
    Ez0, Hx0, Hy0, eps_r, mu_r, steps, dt, dx, dy
        As in :func:`fdtd_2d_tmz`.
    sigma : ndarray, shape (Nx, Ny), optional
        Electric conductivity map. ``None`` (default) reproduces
        :func:`fdtd_2d_tmz`'s lossless, hard-wall behavior exactly.
    source : callable, optional
        ``source(Ez, t)``, called after each field update and expected to
        mutate ``Ez`` in place (a soft source: added to, not overwriting,
        the field already there).
    snapshot_stride : int, default=1
        Record ``Ez`` every this many steps (plus the initial condition).

    Returns
    -------
    snapshots : ndarray, shape (n_recorded, Nx, Ny)
        ``Ez`` at ``t=0`` and after every recorded step.
    times : ndarray, shape (n_recorded,)
        Time of each recorded snapshot.

    See Also
    --------
    fdtd_2d_tmz : The single-shot (final-state-only) solver this wraps.

    Examples
    --------
    >>> import numpy as np
    >>> Nx, Ny = 40, 40
    >>> Ez0 = np.zeros((Nx, Ny))
    >>> Hx0 = Hy0 = np.zeros((Nx, Ny))
    >>> eps_r = mu_r = np.ones((Nx, Ny))
    >>> dx = dy = 1e-3
    >>> dt = 0.5 * courant_limit_2d(dx, dy)
    >>> source = oscillating_dipole_source(Nx // 2, Ny // 2, amplitude=1.0, freq=5e10)
    >>> snaps, times = fdtd_2d_tmz_evolve(Ez0, Hx0, Hy0, eps_r, mu_r, steps=20, dt=dt, dx=dx, dy=dy, source=source, snapshot_stride=5)
    >>> snaps.shape
    (5, 40, 40)
    """
    Ez = np.array(Ez0, dtype=float, copy=True)
    Hx = np.array(Hx0, dtype=float, copy=True)
    Hy = np.array(Hy0, dtype=float, copy=True)
    if sigma is None:
        Ca = Cb = None
    else:
        loss = sigma * dt / (2 * EPS0 * eps_r)
        Ca = (1 - loss) / (1 + loss)
        Cb = (dt / (EPS0 * eps_r)) / (1 + loss)
    n_snap = steps // snapshot_stride + 1
    snapshots = np.empty((n_snap, *Ez.shape))
    times = np.empty(n_snap)
    snapshots[0] = Ez
    times[0] = 0.0
    idx = 1
    for step in range(1, steps + 1):
        _fdtd_2d_tmz_step(Ez, Hx, Hy, eps_r, mu_r, dt, dx, dy, Ca, Cb)
        t = step * dt
        if source is not None:
            source(Ez, t)
        if step % snapshot_stride == 0:
            snapshots[idx] = Ez
            times[idx] = t
            idx += 1
    return snapshots[:idx], times[:idx]


def oscillating_dipole_source(i0: int, j0: int, amplitude: float, freq: float):
    """Build a soft, sinusoidally-oscillating point source for :func:`fdtd_2d_tmz_evolve`.

    Models a simple oscillating dipole antenna: a source current
    concentrated at a single Yee cell, injected as a *soft* source (added
    to the existing field rather than overwriting it, so outgoing waves
    already at that cell are not clobbered).

    Parameters
    ----------
    i0, j0 : int
        Grid indices of the source cell.
    amplitude : float
        Peak field amplitude injected each step.
    freq : float
        Oscillation frequency in Hz.

    Returns
    -------
    callable
        ``source(Ez, t)``, suitable for :func:`fdtd_2d_tmz_evolve`'s
        ``source`` argument: adds ``amplitude * sin(2*pi*freq*t)`` to
        ``Ez[i0, j0]`` in place.

    Examples
    --------
    >>> import numpy as np
    >>> Ez = np.zeros((10, 10))
    >>> source = oscillating_dipole_source(5, 5, amplitude=2.0, freq=1.0)
    >>> source(Ez, t=0.25)
    >>> round(float(Ez[5, 5]), 6)
    2.0
    """

    def _source(Ez, t):
        Ez[i0, j0] += amplitude * np.sin(2.0 * np.pi * freq * t)

    return _source


def dielectric_slab(shape: tuple, i_start: int, i_end: int, eps_r_slab: float = 4.0) -> np.ndarray:
    """A planar dielectric slab: ``eps_r_slab`` between grid columns ``i_start`` and ``i_end``, vacuum elsewhere.

    A minimal demo permittivity map for :func:`fdtd_2d_tmz` / :func:`fdtd_2d_tmz_evolve`:
    a wave launched from one side crossing ``i_start`` partially reflects
    and partially refracts/transmits into the slab, and again at ``i_end``
    on the way out, illustrating propagation through a medium interface.

    Parameters
    ----------
    shape : tuple(int, int)
        ``(Nx, Ny)`` grid shape.
    i_start, i_end : int
        Grid-index bounds of the slab along the ``x`` axis (rows ``i_start:i_end``).
    eps_r_slab : float, default=4.0
        Relative permittivity inside the slab.

    Returns
    -------
    ndarray, shape (Nx, Ny)
        ``eps_r`` map: ``eps_r_slab`` inside the slab, ``1.0`` outside.

    Examples
    --------
    >>> eps_r = dielectric_slab((40, 20), i_start=15, i_end=25, eps_r_slab=4.0)
    >>> float(eps_r[10, 5]), float(eps_r[20, 5])
    (1.0, 4.0)
    """
    Nx, Ny = shape
    eps_r = np.ones((Nx, Ny))
    eps_r[i_start:i_end, :] = eps_r_slab
    return eps_r


def tmz_cavity_mode(shape: tuple, dx: float, dy: float, m: int, n: int) -> tuple:
    """The analytic :math:`TM_{mn}` standing-wave mode of an ideal rectangular PEC cavity.

    :func:`fdtd_2d_tmz` (and :func:`fdtd_2d_tmz_evolve` without ``sigma``)
    already enforce ``Ez=0`` on all four edges every step -- exactly a
    perfectly-conducting (PEC) cavity wall, no absorbing boundary needed.
    Seeding the grid with this analytic mode shape (which already vanishes
    on the boundary) launches a standing wave that should oscillate in
    place at ``omega_mn`` rather than propagate.

    Parameters
    ----------
    shape : tuple(int, int)
        ``(Nx, Ny)`` grid shape.
    dx, dy : float
        Grid spacing; the cavity spans ``Lx=(Nx-1)*dx`` by ``Ly=(Ny-1)*dy``.
    m, n : int
        Mode indices (number of half-wavelengths along each axis).

    Returns
    -------
    Ez0 : ndarray, shape (Nx, Ny)
        :math:`\\sin(m\\pi x/L_x)\\sin(n\\pi y/L_y)`.
    omega_mn : float
        The mode's angular frequency, :math:`c_0\\pi\\sqrt{(m/L_x)^2+(n/L_y)^2}`.

    Examples
    --------
    >>> Ez0, omega = tmz_cavity_mode((41, 41), dx=1e-3, dy=1e-3, m=1, n=1)
    >>> bool(np.all(Ez0[0, :] == 0) and np.all(Ez0[:, 0] == 0))
    True
    >>> bool(omega > 0)
    True
    """
    Nx, Ny = shape
    Lx, Ly = (Nx - 1) * dx, (Ny - 1) * dy
    x = np.arange(Nx) * dx
    y = np.arange(Ny) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    Ez0 = np.sin(m * np.pi * X / Lx) * np.sin(n * np.pi * Y / Ly)
    # sin(m*pi) / sin(n*pi) at the boundary is 0 only up to floating-point roundoff
    # (e.g. ~1e-16, not exactly 0) -- clamp it exactly, since it must vanish there by construction.
    Ez0[0, :] = 0.0
    Ez0[-1, :] = 0.0
    Ez0[:, 0] = 0.0
    Ez0[:, -1] = 0.0
    omega_mn = C0 * np.pi * np.sqrt((m / Lx) ** 2 + (n / Ly) ** 2)
    return Ez0, omega_mn


def poynting_vector_tmz(Ez: np.ndarray, Hx: np.ndarray, Hy: np.ndarray) -> tuple:
    """Poynting vector :math:`\\mathbf{S} = \\mathbf{E} \\times \\mathbf{H}` for a TMz-mode field.

    Parameters
    ----------
    Ez : ndarray
        Out-of-plane electric field.
    Hx, Hy : ndarray
        In-plane magnetic field components, same shape as ``Ez``.

    Returns
    -------
    Sx, Sy : ndarray
        In-plane energy flux density components, :math:`S_x = -E_z H_y`,
        :math:`S_y = E_z H_x`.

    Examples
    --------
    >>> import numpy as np
    >>> Ez = np.array([[2.0]])
    >>> Hx = np.array([[3.0]])
    >>> Hy = np.array([[5.0]])
    >>> poynting_vector_tmz(Ez, Hx, Hy)
    (array([[-10.]]), array([[6.]]))
    """
    return -Ez * Hy, Ez * Hx


# -- Toy confinement ("flux tube") model -------------------------------------
#
# NOT a lattice-QCD or first-principles gauge-theory calculation. This is the
# textbook illustrative picture of confinement (the "dual superconductor" /
# bag-model flux tube): unlike an ordinary Coulomb field, which spreads out
# in all directions and whose energy is independent of source separation,
# confinement is modeled here simply by *forbidding* the field from
# spreading transversely -- it is squeezed into a tube of fixed
# cross-section connecting the two charges. Gauss's law then forces the
# field strength inside the tube to stay at the fixed value set by the
# enclosed charge, independent of how far apart the charges are, so the
# field energy density is uniform along the whole tube and the total field
# energy grows *linearly* with separation -- the defining signature of
# confinement, reproduced here by direct construction rather than derived
# from an underlying gauge-invariant Lagrangian.


def flux_tube_field_1d(x: np.ndarray, separation: float, flux_quantum: float = 1.0, wall_width: float | None = None) -> np.ndarray:
    """Static 1D confined flux profile between two opposite point charges (toy confinement model).

    Solves the 1D Gauss law :math:`d(\\text{flux})/dx = \\rho(x)` for a field
    confined to a fixed-cross-section tube connecting two opposite charges
    at :math:`x=\\pm\\text{separation}/2`, each smoothed into a Gaussian of
    width ``wall_width`` (a numerical regularization of the point charge,
    not a physical effect). The field is computed by direct numerical
    integration (cumulative trapezoidal sum) of this charge density, so it
    is genuinely "solved for" at each separation rather than assumed.

    Parameters
    ----------
    x : ndarray
        1D grid the charges live on.
    separation : float
        Distance between the two charges.
    flux_quantum : float, default=1.0
        Charge magnitude (equivalently, the plateau field strength inside the tube).
    wall_width : float, optional
        Smoothing width of each point charge. Defaults to ``3 * (x[1] - x[0])``,
        i.e. a few grid cells -- small enough to look like a point charge
        but resolved on the grid.

    Returns
    -------
    ndarray, same shape as ``x``
        The confined field, approximately ``+flux_quantum`` between the
        charges and ``0`` outside, with smooth transitions of width ``wall_width``.

    See Also
    --------
    flux_tube_energy_density_2d : The corresponding 2D energy-density map, for animation.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-20, 20, 2000)
    >>> field = flux_tube_field_1d(x, separation=10.0, flux_quantum=1.0)
    >>> bool(abs(field[np.argmin(np.abs(x))] - 1.0) < 0.05)
    True
    >>> bool(abs(field[0]) < 0.05)
    True
    """
    dx = x[1] - x[0]
    if wall_width is None:
        wall_width = 3.0 * dx
    x_left, x_right = -separation / 2.0, separation / 2.0
    norm = 1.0 / (wall_width * np.sqrt(2.0 * np.pi))
    rho = flux_quantum * norm * (np.exp(-0.5 * ((x - x_left) / wall_width) ** 2) - np.exp(-0.5 * ((x - x_right) / wall_width) ** 2))
    field = np.concatenate(([0.0], np.cumsum((rho[1:] + rho[:-1]) * 0.5 * dx)))
    return field


def flux_tube_energy_density_2d(
    shape: tuple, dx: float, dy: float, separation: float, flux_quantum: float = 1.0, tube_width: float | None = None
) -> np.ndarray:
    """2D energy-density map of the confined flux tube, for visualizing it as a stretching heatmap.

    Extrudes :func:`flux_tube_field_1d` (evaluated along the axis joining
    the two charges) into 2D by giving it a fixed-width Gaussian transverse
    profile -- the "fixed cross-section" that produces confinement -- and
    forms the field energy density :math:`\\tfrac{1}{2}\\,\\text{field}(x)^2\\times\\text{transverse}(y)^2`.

    Parameters
    ----------
    shape : tuple(int, int)
        ``(Nx, Ny)`` grid shape.
    dx, dy : float
        Grid spacing; the charges are placed on the centered grid at
        :math:`x=\\pm\\text{separation}/2`, :math:`y=0`.
    separation : float
        Distance between the two charges.
    flux_quantum : float, default=1.0
        Charge magnitude, as in :func:`flux_tube_field_1d`.
    tube_width : float, optional
        Transverse (``y``) width of the flux tube. Defaults to ``5 * dy``.

    Returns
    -------
    ndarray, shape (Nx, Ny)
        Energy density, largest along the tube connecting the charges and
        decaying away from its axis.

    Examples
    --------
    >>> e = flux_tube_energy_density_2d((200, 40), dx=0.2, dy=0.2, separation=10.0)
    >>> bool(e.sum() > 0)
    True
    """
    Nx, Ny = shape
    if tube_width is None:
        tube_width = 5.0 * dy
    x = (np.arange(Nx) - Nx // 2) * dx
    y = (np.arange(Ny) - Ny // 2) * dy
    field_x = flux_tube_field_1d(x, separation, flux_quantum)
    transverse = np.exp(-0.5 * (y / tube_width) ** 2)
    return 0.5 * (field_x[:, None] ** 2) * (transverse[None, :] ** 2)
