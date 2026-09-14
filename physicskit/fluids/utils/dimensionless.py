"""Dimensionless numbers: Reynolds, Froude, Mach, Strouhal, and Weber.

Fluid mechanics is governed by ratios, not absolute scales: two flows with
the same dimensionless numbers behave identically regardless of their size,
speed, or fluid, which is what makes wind-tunnel testing of a model
airplane meaningful for a full-size one. Each function here computes one
such ratio, and each result's magnitude is what determines which regime
(and which module of this package) actually applies -- a low Reynolds
number licenses :func:`physicskit.fluids.systems.viscous_flow.stokes_drag`,
a high one calls for
:class:`physicskit.fluids.systems.navier_stokes.NavierStokes2D` or the
turbulence tools in :mod:`physicskit.fluids.utils.spectral_analysis`.
"""

from __future__ import annotations

from physicskit.fluids.exceptions import InvalidParameterError

__all__ = [
    "reynolds_number",
    "froude_number",
    "mach_number",
    "strouhal_number",
    "weber_number",
]


def reynolds_number(velocity: float, length: float, nu: float) -> float:
    """The Reynolds number: inertial forces over viscous forces.

    .. math::

        Re = \\frac{U L}{\\nu}

    Osborne Reynolds' 1883 pipe-flow experiments identified this ratio as
    the single parameter controlling the transition from smooth (laminar)
    to chaotic (turbulent) flow, regardless of the pipe's size or the
    fluid's identity -- the founding result of dimensional-similarity
    reasoning in fluid mechanics.

    Parameters
    ----------
    velocity : float
        Characteristic flow speed `U`.
    length : float
        Characteristic length scale `L`.
    nu : float
        Kinematic viscosity.

    Returns
    -------
    float
        Reynolds number `Re`.

    Raises
    ------
    InvalidParameterError
        If `length` or `nu` is not positive.

    Examples
    --------
    >>> reynolds_number(velocity=2.0, length=0.5, nu=1e-6)
    1000000.0
    """
    if length <= 0:
        raise InvalidParameterError(f"length must be positive, got {length}")
    if nu <= 0:
        raise InvalidParameterError(f"nu must be positive, got {nu}")
    return velocity * length / nu


def froude_number(velocity: float, length: float, g: float = 9.81) -> float:
    """The Froude number: inertial forces over gravitational forces.

    .. math::

        Fr = \\frac{U}{\\sqrt{gL}}

    Governs free-surface (gravity wave) flows the way the Reynolds number
    governs viscous ones: a ship model and its full-size counterpart make
    the same waves, relative to their length, only if `Fr` matches. William
    Froude's 1868 ship-hull towing-tank experiments established exactly
    this scaling law.

    Parameters
    ----------
    velocity : float
        Characteristic flow speed `U`.
    length : float
        Characteristic length scale `L`.
    g : float, default 9.81
        Gravitational acceleration.

    Returns
    -------
    float
        Froude number `Fr`.

    Raises
    ------
    InvalidParameterError
        If `length` or `g` is not positive.

    Examples
    --------
    >>> round(froude_number(velocity=3.0, length=9.81, g=9.81), 4)
    0.3058
    """
    if length <= 0:
        raise InvalidParameterError(f"length must be positive, got {length}")
    if g <= 0:
        raise InvalidParameterError(f"g must be positive, got {g}")
    return velocity / (g * length) ** 0.5


def mach_number(velocity: float, speed_of_sound: float) -> float:
    """The Mach number: flow speed over the local speed of sound.

    .. math::

        M = \\frac{U}{c}

    The single number that decides whether compressibility can be ignored
    (:math:`M \\ll 1`, the regime of every other module in this package
    except :mod:`physicskit.fluids.systems.compressible_flow`) or dominates
    the flow (:math:`M \\gtrsim 1`, where :func:`~physicskit.fluids.systems.compressible_flow.normal_shock_relations`
    applies).

    Parameters
    ----------
    velocity : float
        Flow speed `U`.
    speed_of_sound : float
        Local speed of sound `c`.

    Returns
    -------
    float
        Mach number `M`.

    Raises
    ------
    InvalidParameterError
        If `speed_of_sound` is not positive.

    Examples
    --------
    >>> mach_number(velocity=340.0, speed_of_sound=340.0)
    1.0
    """
    if speed_of_sound <= 0:
        raise InvalidParameterError(f"speed_of_sound must be positive, got {speed_of_sound}")
    return velocity / speed_of_sound


def strouhal_number(frequency: float, length: float, velocity: float) -> float:
    """The Strouhal number: a dimensionless vortex-shedding frequency.

    .. math::

        St = \\frac{f L}{U}

    For flow past a bluff body shedding a
    :func:`~physicskit.fluids.systems.vortex_dynamics.von_karman_vortex_street`
    at frequency `f`, `St` stays remarkably constant (near 0.2 for a
    circular cylinder) over a wide range of Reynolds number, letting a
    single scaled model predict the shedding frequency at full scale.

    Parameters
    ----------
    frequency : float
        Vortex-shedding frequency `f`.
    length : float
        Characteristic length scale `L` (e.g. the body's cross-stream width).
    velocity : float
        Characteristic flow speed `U`.

    Returns
    -------
    float
        Strouhal number `St`.

    Raises
    ------
    InvalidParameterError
        If `velocity` is not positive.

    Examples
    --------
    >>> round(strouhal_number(frequency=2.0, length=0.1, velocity=1.0), 4)
    0.2
    """
    if velocity <= 0:
        raise InvalidParameterError(f"velocity must be positive, got {velocity}")
    return frequency * length / velocity


def weber_number(rho: float, velocity: float, length: float, surface_tension: float) -> float:
    """The Weber number: inertial forces over surface-tension forces.

    .. math::

        We = \\frac{\\rho U^2 L}{\\sigma}

    Governs whether a free liquid surface (a droplet, a jet, a bubble)
    deforms and breaks up under its own inertia (:math:`We \\gg 1`) or is
    held together by surface tension (:math:`We \\lesssim 1`) -- the
    parameter that decides, for instance, whether a jet of liquid breaks
    into droplets.

    Parameters
    ----------
    rho : float
        Fluid density.
    velocity : float
        Characteristic flow speed `U`.
    length : float
        Characteristic length scale `L` (e.g. a droplet diameter).
    surface_tension : float
        Surface tension coefficient :math:`\\sigma`.

    Returns
    -------
    float
        Weber number `We`.

    Raises
    ------
    InvalidParameterError
        If `surface_tension` is not positive.

    Examples
    --------
    >>> round(weber_number(rho=1000.0, velocity=1.0, length=0.001, surface_tension=0.072), 4)
    13.8889
    """
    if surface_tension <= 0:
        raise InvalidParameterError(f"surface_tension must be positive, got {surface_tension}")
    return rho * velocity**2 * length / surface_tension
