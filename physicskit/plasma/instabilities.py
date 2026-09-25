"""Time-evolving current-sheet instabilities: magnetic reconnection and Weibel filamentation.

Both problems in this module involve a current-carrying layer that breaks up
under a collective instability, but at opposite ends of the fluid-kinetic
spectrum. Magnetic reconnection (:func:`reconnection_harris_ic`,
:func:`simulate_reconnection`) is treated here as a *kinematic resistive-MHD*
problem: the induction equation for the flux function :math:`\\psi(x,y,t)` is
evolved with a prescribed inflow velocity rather than a self-consistently
solved momentum equation, which is the standard first simplification made
before attempting full two-fluid or MHD reconnection (the prescribed-flow
approximation underlying, e.g., the original Sweet-Parker and Petschek
estimates in :mod:`physicskit.plasma.mhd`). The Weibel/filamentation
instability (:func:`weibel_growth_rate`, :func:`simulate_weibel_filamentation`)
is treated as a *reduced linear/quasi-linear* problem: rather than extending
the 1D-space/1D-velocity PIC machinery of :mod:`physicskit.plasma.kinetic` to
a full 2D-in-velocity, electromagnetic PIC code (a much larger undertaking),
a spectrum of transverse-current Fourier modes is seeded with random phases
and each mode's amplitude is evolved analytically according to the known
linear Weibel growth rate -- exact for as long as the linear approximation
holds, and sufficient to show the characteristic real-space current
filaments forming and growing.
"""

from __future__ import annotations

import numpy as np

from physicskit import constants as _const

__all__ = [
    "reconnection_harris_ic",
    "reconnection_inflow_velocity",
    "reconnection_field_from_flux",
    "simulate_reconnection",
    "weibel_growth_rate",
    "weibel_fastest_growing_mode",
    "simulate_weibel_filamentation",
]


def reconnection_harris_ic(nx: int, ny: int, Lx: float, Ly: float, sheet_width: float, perturbation_amplitude: float, k_modes: int = 1) -> np.ndarray:
    """Flux-function initial condition: a Harris current sheet with an X-point-seeding ripple.

    The unperturbed Harris sheet :math:`\\psi_0(y) = -B_0 L\\ln\\cosh(y/L)`
    gives an antiparallel reconnecting field
    :math:`B_x = \\partial_y\\psi_0 = -B_0\\tanh(y/L)` (with the
    :func:`reconnection_field_from_flux` convention :math:`B_x=\\partial_y\\psi`) that reverses sign
    across :math:`y=0` -- the classic current-sheet configuration in which
    reconnection is normally studied. Adding a small ripple
    :math:`\\psi_1 = \\epsilon\\cos(k x)\\,\\mathrm{sech}^2(y/L)`, localized to
    the sheet and periodic in :math:`x`, is the standard tearing-mode-like
    seed: it breaks the translational symmetry along the sheet just enough
    to create one X-point/O-point pair per wavelength, without which the
    sheet would stay a stationary equilibrium forever regardless of
    resistivity.

    Parameters
    ----------
    nx, ny : int
        Grid points along `x` (periodic, along the sheet) and `y`
        (across the sheet).
    Lx, Ly : float
        Domain size in `x` and `y`; the domain spans
        ``[0, Lx) x [-Ly/2, Ly/2)``.
    sheet_width : float
        Current-sheet half-thickness :math:`L` (with :math:`B_0=1` in these
        normalized units).
    perturbation_amplitude : float
        Amplitude :math:`\\epsilon` of the seeding ripple.
    k_modes : int, default=1
        Number of full wavelengths of the ripple fit into `Lx`.

    Returns
    -------
    ndarray, shape (nx, ny)
        Flux function :math:`\\psi(x, y)`.

    See Also
    --------
    simulate_reconnection : Evolves this initial condition forward in time.

    Examples
    --------
    >>> psi0 = reconnection_harris_ic(64, 64, Lx=20.0, Ly=20.0, sheet_width=1.0, perturbation_amplitude=0.1)
    >>> psi0.shape
    (64, 64)
    """
    x = np.linspace(0.0, Lx, nx, endpoint=False)
    y = np.linspace(-Ly / 2.0, Ly / 2.0, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    psi0 = -sheet_width * np.log(np.cosh(Y / sheet_width))
    k = 2.0 * np.pi * k_modes / Lx
    psi1 = perturbation_amplitude * np.cos(k * X) / np.cosh(Y / sheet_width) ** 2
    return psi0 + psi1


def reconnection_inflow_velocity(X: np.ndarray, Y: np.ndarray, v0: float, Lx: float, Ly: float) -> tuple:
    """Prescribed stagnation-point inflow/outflow velocity field around the reconnection X-point.

    A simple incompressible stagnation flow, :math:`v_x=+v_0 x'/L_x`,
    :math:`v_y=-v_0 y'/L_y` (with :math:`x', y'` measured from the nearest
    X-point at the domain center), that converges onto the X-point from
    above and below and squirts back out sideways as reconnection outflow
    "jets" -- exactly the inflow/outflow pattern Sweet-Parker and Petschek
    reconnection assume, here imposed directly rather than derived from a
    momentum equation (see the module docstring).

    Parameters
    ----------
    X, Y : ndarray
        Coordinate grids, as built by :func:`reconnection_harris_ic`.
    v0 : float
        Characteristic inflow speed.
    Lx, Ly : float
        Domain size in `x` and `y`, used to center the stagnation point.

    Returns
    -------
    vx, vy : ndarray
        Velocity components, same shape as `X`.

    Examples
    --------
    >>> import numpy as np
    >>> X, Y = np.meshgrid(np.linspace(0, 20, 8, endpoint=False), np.linspace(-10, 10, 8, endpoint=False), indexing="ij")
    >>> vx, vy = reconnection_inflow_velocity(X, Y, v0=0.1, Lx=20.0, Ly=20.0)
    >>> vx.shape
    (8, 8)
    """
    Xc = np.mod(X - Lx / 2.0 + Lx / 2.0, Lx) - Lx / 2.0
    vx = v0 * Xc / (Lx / 2.0)
    vy = -v0 * Y / (Ly / 2.0)
    return vx, vy


def reconnection_field_from_flux(psi: np.ndarray, dx: float, dy: float) -> tuple:
    """Recover the in-plane magnetic field :math:`(B_x, B_y) = (\\partial_y\\psi, -\\partial_x\\psi)` from the flux function.

    Parameters
    ----------
    psi : ndarray, shape (nx, ny)
        Flux function, periodic in both directions.
    dx, dy : float
        Grid spacing along `x` and `y`.

    Returns
    -------
    Bx, By : ndarray, shape (nx, ny)
        In-plane magnetic field components.

    Examples
    --------
    >>> import numpy as np
    >>> psi = reconnection_harris_ic(32, 32, Lx=20.0, Ly=20.0, sheet_width=1.0, perturbation_amplitude=0.0)
    >>> Bx, By = reconnection_field_from_flux(psi, dx=20.0 / 32, dy=20.0 / 32)
    >>> bool(np.all(Bx[:, 16] * Bx[:, 15] <= 0))
    True
    """
    dpsi_dy = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2.0 * dy)
    dpsi_dx = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2.0 * dx)
    return dpsi_dy, -dpsi_dx


def _laplacian_periodic(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    d2x = (np.roll(f, -1, axis=0) - 2.0 * f + np.roll(f, 1, axis=0)) / dx**2
    d2y = (np.roll(f, -1, axis=1) - 2.0 * f + np.roll(f, 1, axis=1)) / dy**2
    return d2x + d2y


def simulate_reconnection(psi0: np.ndarray, eta: float, v0: float, dt: float, steps: int, Lx: float, Ly: float) -> dict:
    """Time-step the kinematic resistive induction equation :math:`\\partial_t\\psi = \\eta\\nabla^2\\psi - \\mathbf{v}\\cdot\\nabla\\psi`.

    A doubly periodic, second-order central-difference, explicit (forward
    Euler in time) finite-difference solve. With a prescribed velocity
    field rather than one obtained from solving the momentum equation
    (see the module docstring), this reduces reconnection to pure flux
    transport-and-diffusion: the inflow (:func:`reconnection_inflow_velocity`)
    advects oppositely directed flux into the X-point, where resistive
    diffusion (the :math:`\\eta\\nabla^2\\psi` term) is the only mechanism
    that can actually break and reconnect field lines -- exactly Faraday's
    law with an Ohmic (rather than ideal) Ohm's law, restricted to the
    kinematic (fixed-flow) limit.

    Explicit forward-Euler time-stepping is only conditionally stable:
    the diffusive term requires :math:`\\eta\\,dt \\lesssim \\tfrac{1}{4}\\min(dx,dy)^2`
    and the advective term requires the Courant condition
    :math:`v_0\\,dt \\lesssim \\min(dx, dy)`. Both are the caller's
    responsibility to satisfy by choosing `dt` appropriately; unstable
    combinations manifest as exponentially growing grid-scale noise.

    Parameters
    ----------
    psi0 : ndarray, shape (nx, ny)
        Initial flux function, e.g. from :func:`reconnection_harris_ic`.
    eta : float
        Resistivity (magnetic diffusivity).
    v0 : float
        Characteristic inflow speed for :func:`reconnection_inflow_velocity`.
    dt : float
        Time step.
    steps : int
        Number of forward-Euler steps to advance.
    Lx, Ly : float
        Domain size in `x` and `y`.

    Returns
    -------
    dict
        ``{"psi": final flux function, "Bx": ..., "By": ...}``.

    See Also
    --------
    reconnection_harris_ic : Builds the initial condition consumed here.
    physicskit.plasma.mhd.sweet_parker_rate : The steady-state reconnection
        rate this kinematic model is a simplified, time-dependent analogue of.

    Examples
    --------
    >>> import numpy as np
    >>> psi0 = reconnection_harris_ic(48, 48, Lx=20.0, Ly=20.0, sheet_width=1.0, perturbation_amplitude=0.2)
    >>> result = simulate_reconnection(psi0, eta=0.02, v0=0.05, dt=0.02, steps=50, Lx=20.0, Ly=20.0)
    >>> result["psi"].shape
    (48, 48)
    >>> bool(np.all(np.isfinite(result["psi"])))
    True
    """
    nx, ny = psi0.shape
    dx, dy = Lx / nx, Ly / ny
    x = np.linspace(0.0, Lx, nx, endpoint=False)
    y = np.linspace(-Ly / 2.0, Ly / 2.0, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    vx, vy = reconnection_inflow_velocity(X, Y, v0, Lx, Ly)

    psi = np.array(psi0, dtype=float, copy=True)
    for _ in range(steps):
        dpsi_dx = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2.0 * dx)
        dpsi_dy = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2.0 * dy)
        rhs = eta * _laplacian_periodic(psi, dx, dy) - (vx * dpsi_dx + vy * dpsi_dy)
        psi = psi + dt * rhs

    Bx, By = reconnection_field_from_flux(psi, dx, dy)
    return {"psi": psi, "Bx": Bx, "By": By}


def weibel_growth_rate(k: np.ndarray, wpe: float, temperature_anisotropy: float, c: float = _const.C) -> np.ndarray:
    """Linear growth rate of the (non-relativistic, cold-parallel-limit) Weibel/filamentation instability.

    For an electron distribution anisotropic between the direction of
    wave propagation (temperature :math:`T_\\parallel`, taken cold here)
    and the transverse direction (temperature :math:`T_\\perp`), the
    electromagnetic mode with wavevector :math:`\\mathbf{k}` transverse to
    the anisotropy axis is purely growing (Weibel, 1959) with

    .. math::

       \\gamma(k)^2 = \\omega_{pe}^2\\left(\\frac{T_\\perp}{T_\\parallel}-1\\right) - k^2 c^2,

    unstable for :math:`T_\\perp > T_\\parallel` and only up to the cutoff
    :math:`k_{max} = (\\omega_{pe}/c)\\sqrt{T_\\perp/T_\\parallel - 1}` --
    beyond which the field's magnetic tension (the :math:`k^2c^2` term)
    overcomes the free energy the anisotropy supplies. Growing transverse
    magnetic perturbations pinch counter-streaming electron sub-populations
    into spatially separated current channels: the filaments this
    instability is named for.

    Parameters
    ----------
    k : ndarray or float
        Wavenumber(s), transverse to the anisotropy axis.
    wpe : float
        Electron plasma frequency.
    temperature_anisotropy : float
        The ratio :math:`T_\\perp/T_\\parallel`; must exceed 1 for any
        instability.
    c : float, default=physicskit.constants.C
        Speed of light, in the same units `k` and `wpe` combine to give
        (SI by default).

    Returns
    -------
    ndarray or float
        Growth rate :math:`\\gamma(k)`, clipped to zero for stable (real
        or imaginary-:math:`\\gamma^2<0`) wavenumbers.

    See Also
    --------
    weibel_fastest_growing_mode : The wavenumber maximizing this growth rate.
    simulate_weibel_filamentation : Superposes modes growing at this rate.

    Examples
    --------
    >>> import numpy as np
    >>> round(float(weibel_growth_rate(k=0.0, wpe=1.0, temperature_anisotropy=4.0)), 6)
    1.732051
    >>> bool(weibel_growth_rate(k=1e10, wpe=1.0, temperature_anisotropy=4.0) == 0.0)
    True
    """
    k = np.asarray(k, dtype=float)
    gamma_sq = wpe**2 * (temperature_anisotropy - 1.0) - (k * c) ** 2
    return np.sqrt(np.clip(gamma_sq, 0.0, None))


def weibel_fastest_growing_mode(wpe: float, temperature_anisotropy: float, c: float = _const.C) -> tuple:
    """Wavenumber and growth rate of the fastest-growing Weibel mode.

    For the growth-rate law of :func:`weibel_growth_rate`,
    :math:`\\gamma(k)^2` is maximized at :math:`k=0` (long wavelength,
    where the stabilizing field-tension term vanishes), giving
    :math:`\\gamma_{max}=\\omega_{pe}\\sqrt{T_\\perp/T_\\parallel-1}` -- the
    growth rate used to set the overall filament-growth timescale in
    :func:`simulate_weibel_filamentation`, even though the *dominant
    filament spacing* in a real (finite-seed) simulation is set by which
    finite-:math:`k` modes happen to be seeded with the largest initial
    amplitude, since every :math:`k` below cutoff grows, just more slowly
    farther from :math:`k=0`.

    Parameters
    ----------
    wpe : float
        Electron plasma frequency.
    temperature_anisotropy : float
        The ratio :math:`T_\\perp/T_\\parallel`. At or below 1 the plasma
        is Weibel-stable and ``(0.0, 0.0)`` is returned.
    c : float, default=physicskit.constants.C
        Speed of light. Unused, since the maximum sits at :math:`k=0` where
        the :math:`k^2c^2` term vanishes; kept for signature symmetry with
        :func:`weibel_growth_rate`.

    Returns
    -------
    k_max_growth : float
        Wavenumber of maximum growth (always 0.0).
    gamma_max : float
        The corresponding growth rate :math:`\\gamma_{max}`.

    Examples
    --------
    >>> k0, gamma_max = weibel_fastest_growing_mode(wpe=1.0, temperature_anisotropy=4.0)
    >>> k0
    0.0
    >>> round(gamma_max, 6)
    1.732051
    """
    if temperature_anisotropy <= 1.0:
        return 0.0, 0.0
    gamma_max = wpe * np.sqrt(temperature_anisotropy - 1.0)
    return 0.0, float(gamma_max)


def simulate_weibel_filamentation(
    x: np.ndarray, t: np.ndarray, wpe: float, temperature_anisotropy: float, n_modes: int = 12, k_max_factor: float = 0.9, seed: int = 0
) -> np.ndarray:
    """Reduced quasi-linear Weibel filamentation model: a spectrum of independently growing current modes.

    Seeds `n_modes` transverse-current Fourier modes with small random
    amplitudes and phases, spanning wavenumbers up to
    ``k_max_factor`` times the linear cutoff :math:`k_{max}` of
    :func:`weibel_growth_rate`, and evolves each mode's amplitude
    independently as :math:`a_k(t) = a_k(0)\\,e^{\\gamma(k)t}` -- exact
    linear theory, valid until the fastest-growing filaments approach
    nonlinear saturation (order-unity current perturbation), at which
    point real filamentation departs from this superposition (mode
    coupling, filament merging) that this reduced model does not capture.
    This stands in for a full 2D-in-velocity electromagnetic PIC
    simulation; see the module docstring for why.

    Parameters
    ----------
    x : ndarray, shape (nx,)
        Spatial grid (periodic).
    t : ndarray, shape (nt,)
        Times at which to evaluate the current pattern.
    wpe : float
        Electron plasma frequency.
    temperature_anisotropy : float
        The ratio :math:`T_\\perp/T_\\parallel`; must exceed 1.
    n_modes : int, default=12
        Number of Fourier modes to seed.
    k_max_factor : float, default=0.9
        Fraction of the linear cutoff wavenumber up to which modes are
        seeded (kept below 1 so every seeded mode is genuinely unstable).
    seed : int, default=0
        Random seed for the initial mode amplitudes and phases.

    Returns
    -------
    ndarray, shape (nt, nx)
        Transverse current density :math:`J_y(x, t)`, in units where the
        initial per-mode amplitude is order unity (i.e. a normalized,
        not absolute, current).

    See Also
    --------
    weibel_growth_rate : The per-mode growth rate used here.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 20.0, 128, endpoint=False)
    >>> t = np.linspace(0, 5.0, 10)
    >>> J = simulate_weibel_filamentation(x, t, wpe=1.0, temperature_anisotropy=4.0, n_modes=6, seed=0)
    >>> J.shape
    (10, 128)
    >>> bool(np.std(J[-1]) > np.std(J[0]))
    True
    """
    rng = np.random.default_rng(seed)
    c = _const.C
    k_max = (wpe / c) * np.sqrt(max(temperature_anisotropy - 1.0, 1e-12))
    k_modes = np.linspace(k_max / n_modes, k_max_factor * k_max, n_modes)
    amplitudes0 = rng.uniform(0.5, 1.0, n_modes)
    phases = rng.uniform(0.0, 2.0 * np.pi, n_modes)
    gammas = weibel_growth_rate(k_modes, wpe, temperature_anisotropy, c=c)

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    growth = np.exp(np.outer(t, gammas))
    amplitudes = amplitudes0[None, :] * growth
    phase_arg = k_modes[None, None, :] * x[None, :, None] + phases[None, None, :]
    J = np.sum(amplitudes[:, None, :] * np.cos(phase_arg), axis=2)
    return J
