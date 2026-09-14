"""Shear and buoyancy-driven instabilities: Kelvin-Helmholtz and Rayleigh-Taylor.

Both instabilities turn a smooth base state (a shear layer, a stratified
interface) into vigorous mixing given only an infinitesimal perturbation --
the base state is *linearly unstable*, not just occasionally disturbed into
instability. This module pairs each initial condition with the linear
growth-rate law that predicts how fast its seeding ripple should grow, and
with a full nonlinear simulation (built on
:mod:`physicskit.fluids.systems.navier_stokes`) that grows it well past the
point where the linear theory stops applying, into the rolled-up vortices
(Kelvin-Helmholtz) or mushroom-shaped plumes (Rayleigh-Taylor) these
instabilities are named for.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from physicskit.fluids.core.timestepping import integrate_boussinesq, integrate_vorticity_streamfunction
from physicskit.fluids.exceptions import InvalidParameterError

__all__ = [
    "kelvin_helmholtz_ic",
    "kelvin_helmholtz_growth_rate",
    "simulate_kelvin_helmholtz",
    "rayleigh_taylor_ic",
    "rayleigh_taylor_growth_rate",
    "simulate_rayleigh_taylor",
]


def kelvin_helmholtz_ic(n: int, length: float, shear_width: float = 0.1, perturbation_amplitude: float = 0.05) -> NDArray[np.float64]:
    """Vorticity initial condition for the Kelvin-Helmholtz shear-layer instability.

    A thin vortex sheet at mid-domain, with a small sinusoidal ripple that
    seeds the instability: under
    :class:`physicskit.fluids.systems.navier_stokes.NavierStokes2D`, the
    ripple's amplitude grows -- initially at the rate predicted by
    :func:`kelvin_helmholtz_growth_rate` -- as the shear layer rolls up into
    a row of discrete "cat's eye" vortices.

    Parameters
    ----------
    n : int
        Number of grid points along each axis.
    length : float
        Physical domain size.
    shear_width : float, default 0.1
        Thickness of the vortex sheet.
    perturbation_amplitude : float, default 0.05
        Amplitude of the seeding sinusoidal ripple.

    Returns
    -------
    ndarray of float, shape (n, n)
        Vorticity field.

    Examples
    --------
    >>> omega0 = kelvin_helmholtz_ic(64, 2 * 3.141592653589793)
    >>> omega0.shape
    (64, 64)
    >>> bool(omega0.max() > 0)
    True
    """
    x = np.linspace(0, length, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    y0 = length / 2
    base = (1.0 / shear_width) / np.cosh((Y - y0) / shear_width) ** 2
    ripple = perturbation_amplitude * np.sin(2 * np.pi * X / length) * np.exp(-(((Y - y0) / (length / 4)) ** 2))
    return base + ripple


def kelvin_helmholtz_growth_rate(k: float, delta_u: float) -> float:
    """Inviscid linear growth rate of a Kelvin-Helmholtz-unstable vortex sheet.

    For a vortex sheet (an infinitesimally thin shear layer with a velocity
    jump `delta_u` across it), linear stability analysis of the Euler
    equations gives a perturbation of wavenumber `k` growing as
    :math:`e^{\\sigma t}` with

    .. math::

        \\sigma(k) = \\frac{k\\,\\Delta u}{2},

    unstable at *every* wavenumber with no threshold velocity -- unlike,
    say, the Rayleigh-Taylor instability below, which is stabilized at short
    wavelength by surface tension or diffusion. A shear layer of finite
    thickness `shear_width` (as built by :func:`kelvin_helmholtz_ic`)
    instead has a fastest-growing wavelength comparable to the layer
    thickness, with growth cut off entirely for :math:`k\\,(\\text{shear
    width})` large (Michalke, 1964); this vortex-sheet limit is the
    thin-layer, small-``k`` approximation to that fuller theory, and is what
    the shear-layer roll-up in :func:`kelvin_helmholtz_ic` grows at
    initially, before finite-thickness and nonlinear effects take over.

    Parameters
    ----------
    k : float
        Perturbation wavenumber (angular, :math:`2\\pi/\\lambda`).
    delta_u : float
        Velocity jump across the shear layer.

    Returns
    -------
    float
        Growth rate :math:`\\sigma`, in units of inverse time.

    Examples
    --------
    >>> round(kelvin_helmholtz_growth_rate(k=1.0, delta_u=2.0), 6)
    1.0
    """
    return 0.5 * k * delta_u


def simulate_kelvin_helmholtz(omega0: NDArray[np.float64], nu: float, dt: float, steps: int, length: float) -> dict[str, NDArray[np.float64]]:
    """Time-step the 2D incompressible Kelvin-Helmholtz shear layer with RK4.

    A thin wrapper around
    :func:`physicskit.fluids.core.timestepping.integrate_vorticity_streamfunction`
    -- the same pseudo-spectral vorticity-streamfunction Navier-Stokes
    engine :func:`simulate_rayleigh_taylor` builds its Boussinesq extension
    on -- evolving :func:`kelvin_helmholtz_ic` through the roll-up of its
    seeded ripple into the characteristic "cat's-eye" vortex row, well past
    the point :func:`kelvin_helmholtz_growth_rate`'s linear-theory
    exponential growth stops applying.

    Parameters
    ----------
    omega0 : ndarray of float, shape (n, n)
        Initial vorticity field, as from :func:`kelvin_helmholtz_ic`.
    nu : float
        Kinematic viscosity; must be positive.
    dt : float
        Time step.
    steps : int
        Number of RK4 steps to advance.
    length : float
        Physical domain size.

    Returns
    -------
    dict
        ``{"omega": ..., "psi": ..., "u": ..., "v": ...}`` at the final time.

    Raises
    ------
    InvalidParameterError
        If `nu` is not positive.

    See Also
    --------
    kelvin_helmholtz_ic : Builds the initial condition consumed here.
    kelvin_helmholtz_growth_rate : The linear-theory prediction this simulation exceeds nonlinearly.
    simulate_rayleigh_taylor : The companion buoyancy-driven instability, built on the same engine.

    Examples
    --------
    >>> import numpy as np
    >>> n, length = 96, 2 * np.pi
    >>> omega0 = kelvin_helmholtz_ic(n, length, shear_width=0.1, perturbation_amplitude=0.05)
    >>> result = simulate_kelvin_helmholtz(omega0, nu=0.001, dt=0.0025, steps=50, length=length)
    >>> result["omega"].shape
    (96, 96)
    """
    if nu <= 0:
        raise InvalidParameterError(f"nu (kinematic viscosity) must be positive, got {nu}")
    return integrate_vorticity_streamfunction(omega0, nu, dt, steps, length)


def rayleigh_taylor_ic(
    n: int, length: float, atwood_number: float, perturbation_amplitude: float = 0.02, interface_position: float | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Vorticity and buoyancy initial condition for the Rayleigh-Taylor instability.

    A denser fluid layer sits atop a lighter one (heavy-on-top is exactly
    the unstable arrangement), separated by an interface at mid-domain with
    a small sinusoidal ripple, evolved through
    :func:`simulate_rayleigh_taylor`. Vorticity starts at exactly zero: in
    the Boussinesq system, all vorticity generation comes from the
    baroclinic torque acting on the rippled density interface once gravity
    is switched on (see :func:`physicskit.fluids.core.timestepping.buoyant_vorticity_rhs`).

    Parameters
    ----------
    n : int
        Number of grid points along each axis.
    length : float
        Physical domain size.
    atwood_number : float
        The Atwood number :math:`A=(\\rho_{heavy}-\\rho_{light})/(\\rho_{heavy}+\\rho_{light})`,
        :math:`0 < A < 1`, setting the buoyancy jump across the interface.
    perturbation_amplitude : float, default 0.02
        Amplitude of the seeding sinusoidal ripple on the interface.
    interface_position : float, optional
        Vertical position of the unperturbed interface; defaults to
        ``length / 2``.

    Returns
    -------
    omega0 : ndarray of float, shape (n, n)
        Initial vorticity field (identically zero).
    buoyancy0 : ndarray of float, shape (n, n)
        Initial buoyancy field: :math:`+A` below the interface (light fluid),
        :math:`-A` above it (heavy fluid), smoothed over one grid cell so the
        spectral derivatives it feeds resolve without ringing.

    Raises
    ------
    InvalidParameterError
        If `atwood_number` is not in ``(0, 1)``.

    Examples
    --------
    >>> omega0, buoyancy0 = rayleigh_taylor_ic(64, 2 * 3.141592653589793, atwood_number=0.3)
    >>> bool(np.all(omega0 == 0.0))
    True
    >>> bool(buoyancy0.max() > 0 > buoyancy0.min())
    True
    """
    if not 0.0 < atwood_number < 1.0:
        raise InvalidParameterError(f"atwood_number must be in (0, 1), got {atwood_number}")
    x = np.linspace(0, length, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    y0 = length / 2 if interface_position is None else interface_position
    smoothing = length / n
    ripple = perturbation_amplitude * np.cos(2 * np.pi * X / length)
    buoyancy0 = -atwood_number * np.tanh((Y - y0 - ripple) / smoothing)
    omega0 = np.zeros_like(X)
    return omega0, buoyancy0


def rayleigh_taylor_growth_rate(k: float, atwood_number: float, g: float = 1.0) -> float:
    """Linear growth rate of the Rayleigh-Taylor instability.

    For two inviscid, semi-infinite fluid layers (heavy density
    :math:`\\rho_2` on top of light density :math:`\\rho_1`, gravity `g`
    pointing from the heavy into the light layer), a perturbation of
    wavenumber `k` on the interface grows as :math:`e^{\\sigma t}` with

    .. math::

        \\sigma(k) = \\sqrt{A\\,g\\,k}, \\qquad
        A = \\frac{\\rho_2-\\rho_1}{\\rho_2+\\rho_1}

    (Rayleigh, 1883; Taylor, 1950), unstable at every wavenumber in the
    idealized inviscid limit -- real fluids are cut off at short wavelength
    by viscosity and surface tension, which :func:`simulate_rayleigh_taylor`
    supplies dynamically via its buoyancy diffusivity.

    Parameters
    ----------
    k : float
        Perturbation wavenumber (angular, :math:`2\\pi/\\lambda`).
    atwood_number : float
        The Atwood number :math:`A`, :math:`0 < A < 1`.
    g : float, default 1.0
        Gravitational acceleration, pointing from the heavy layer into the light one.

    Returns
    -------
    float
        Growth rate :math:`\\sigma`, in units of inverse time.

    Raises
    ------
    InvalidParameterError
        If `atwood_number` is not in ``(0, 1)`` or `k` is negative.

    Examples
    --------
    >>> round(rayleigh_taylor_growth_rate(k=2.0, atwood_number=0.5, g=1.0), 6)
    1.0
    """
    if not 0.0 < atwood_number < 1.0:
        raise InvalidParameterError(f"atwood_number must be in (0, 1), got {atwood_number}")
    if k < 0:
        raise InvalidParameterError(f"k must be non-negative, got {k}")
    return float(np.sqrt(atwood_number * g * k))


def simulate_rayleigh_taylor(
    omega0: NDArray[np.float64],
    buoyancy0: NDArray[np.float64],
    nu: float,
    kappa: float,
    g: float,
    dt: float,
    steps: int,
    length: float,
) -> dict[str, NDArray[np.float64]]:
    """Time-step the Boussinesq Rayleigh-Taylor system with RK4.

    Evolves the coupled vorticity-buoyancy equations of
    :func:`physicskit.fluids.core.timestepping.buoyant_vorticity_rhs`
    (scaled by the gravitational acceleration `g`) starting from
    :func:`rayleigh_taylor_ic`, through the roll-up of the rippled interface
    into the characteristic Rayleigh-Taylor mushroom plumes.

    Parameters
    ----------
    omega0 : ndarray of float, shape (n, n)
        Initial vorticity field, as from :func:`rayleigh_taylor_ic`.
    buoyancy0 : ndarray of float, shape (n, n)
        Initial buoyancy field, as from :func:`rayleigh_taylor_ic`.
    nu : float
        Kinematic viscosity; must be positive.
    kappa : float
        Buoyancy (density) diffusivity; must be positive.
    g : float
        Gravitational acceleration; must be positive.
    dt : float
        Time step.
    steps : int
        Number of RK4 steps to advance.
    length : float
        Physical domain size.

    Returns
    -------
    dict
        ``{"omega": ..., "buoyancy": ..., "psi": ..., "u": ..., "v": ...}`` at
        the final time.

    Raises
    ------
    InvalidParameterError
        If `nu`, `kappa`, or `g` is not positive.

    See Also
    --------
    rayleigh_taylor_ic : Builds the initial condition consumed here.
    rayleigh_taylor_growth_rate : The linear-theory prediction this simulation exceeds nonlinearly.
    """
    if nu <= 0:
        raise InvalidParameterError(f"nu (kinematic viscosity) must be positive, got {nu}")
    if kappa <= 0:
        raise InvalidParameterError(f"kappa (buoyancy diffusivity) must be positive, got {kappa}")
    if g <= 0:
        raise InvalidParameterError(f"g (gravitational acceleration) must be positive, got {g}")
    return integrate_boussinesq(omega0, buoyancy0, nu, kappa, g, dt, steps, length)
