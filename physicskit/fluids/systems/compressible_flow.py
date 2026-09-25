"""1D compressible flow: the Euler equations, normal shocks, and the Sod shock tube.

Where potential and viscous flow assume incompressibility, this module
drops that assumption entirely: the 1D Euler equations for an inviscid,
compressible, ideal gas support genuinely discontinuous solutions (shocks
and contact discontinuities) that no continuous velocity field can produce.
:func:`rankine_hugoniot_jump_conditions` and :func:`normal_shock_relations`
give the exact algebraic jump a shock must satisfy; :func:`sod_shock_tube`
resolves that jump -- alongside a contact discontinuity and an expansion
fan -- dynamically, by integrating the full nonlinear equations through a
classic Riemann problem.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

from physicskit.fluids.exceptions import InvalidParameterError

__all__ = [
    "rankine_hugoniot_jump_conditions",
    "normal_shock_relations",
    "sod_shock_tube",
]


def rankine_hugoniot_jump_conditions(rho1: float, u1: float, p1: float, rho2: float, u2: float, p2: float, gamma: float = 1.4) -> dict[str, float]:
    """Residuals of the Rankine-Hugoniot jump conditions across a stationary discontinuity.

    A steady discontinuity in a 1D inviscid compressible flow (states 1
    upstream, 2 downstream, in the frame where the discontinuity itself is
    at rest) must conserve mass, momentum, and energy flux exactly:

    .. math::

        \\rho_1 u_1 = \\rho_2 u_2, \\qquad
        p_1 + \\rho_1 u_1^2 = p_2 + \\rho_2 u_2^2, \\qquad
        h_1 + \\tfrac{1}{2}u_1^2 = h_2 + \\tfrac{1}{2}u_2^2,

    where :math:`h=\\gamma p/((\\gamma-1)\\rho)` is specific enthalpy (an
    ideal gas is assumed only for the energy residual). Rankine (1870) first
    wrote the mass and momentum conditions; Hugoniot (1887, 1889) added the
    energy condition and the resulting relation between the shock's pressure
    and density ratios now called the Hugoniot curve. This function returns
    each condition's residual, primarily to *verify* a candidate jump (e.g.
    the output of :func:`normal_shock_relations`) rather than to solve for
    one directly.

    Parameters
    ----------
    rho1, u1, p1 : float
        Upstream density, velocity, and pressure.
    rho2, u2, p2 : float
        Downstream density, velocity, and pressure.
    gamma : float, default=1.4
        Ratio of specific heats, used only in the enthalpy of the energy
        residual; must match the one used to generate a candidate jump
        (e.g. :func:`normal_shock_relations`'s `gamma`).

    Returns
    -------
    dict
        ``{"mass": ..., "momentum": ..., "energy": ...}`` residuals; all
        zero for an exactly satisfied jump.

    Examples
    --------
    >>> jump = normal_shock_relations(M1=2.0)
    >>> residuals = rankine_hugoniot_jump_conditions(1.0, 2.0, 1.0 / 1.4, jump["rho2_rho1"], 2.0 / jump["rho2_rho1"], jump["p2_p1"] / 1.4)
    >>> bool(max(abs(v) for v in residuals.values()) < 1e-10)
    True
    """
    mass = rho1 * u1 - rho2 * u2
    momentum = (p1 + rho1 * u1**2) - (p2 + rho2 * u2**2)
    h1 = gamma * p1 / ((gamma - 1.0) * rho1)
    h2 = gamma * p2 / ((gamma - 1.0) * rho2)
    energy = (h1 + 0.5 * u1**2) - (h2 + 0.5 * u2**2)
    return {"mass": mass, "momentum": momentum, "energy": energy}


def normal_shock_relations(M1: float, gamma: float = 1.4) -> dict[str, float]:
    """Ideal-gas normal shock relations as a function of upstream Mach number.

    Solving the Rankine-Hugoniot conditions
    (:func:`rankine_hugoniot_jump_conditions`) for an ideal gas gives the
    downstream state entirely in terms of the upstream Mach number
    :math:`M_1 = u_1/c_1`:

    .. math::

        \\frac{p_2}{p_1} = 1 + \\frac{2\\gamma}{\\gamma+1}(M_1^2-1), \\qquad
        \\frac{\\rho_2}{\\rho_1} = \\frac{(\\gamma+1)M_1^2}{(\\gamma-1)M_1^2+2}, \\qquad
        M_2^2 = \\frac{1+\\tfrac{\\gamma-1}{2}M_1^2}{\\gamma M_1^2-\\tfrac{\\gamma-1}{2}}.

    Only :math:`M_1 \\geq 1` gives a physically admissible (entropy-increasing)
    shock; a supersonic upstream flow needs a mechanism, such as this jump,
    to return to subsonic downstream, and no analogous jump exists in reverse.

    Parameters
    ----------
    M1 : float
        Upstream Mach number; must be at least 1.
    gamma : float, default 1.4
        Ratio of specific heats (1.4 for a diatomic ideal gas, e.g. air).

    Returns
    -------
    dict
        ``{"p2_p1": ..., "rho2_rho1": ..., "T2_T1": ..., "M2": ...}``.

    Raises
    ------
    InvalidParameterError
        If `M1` is less than 1.

    Examples
    --------
    >>> jump = normal_shock_relations(M1=1.0)
    >>> [round(v, 6) for v in (jump["p2_p1"], jump["rho2_rho1"], jump["M2"])]
    [1.0, 1.0, 1.0]
    >>> jump = normal_shock_relations(M1=2.0)
    >>> round(jump["p2_p1"], 3), round(jump["rho2_rho1"], 3), round(jump["M2"], 3)
    (4.5, 2.667, 0.577)
    """
    if M1 < 1.0:
        raise InvalidParameterError(f"M1 must be >= 1 for an admissible (entropy-increasing) shock, got {M1}")
    p2_p1 = 1.0 + (2.0 * gamma / (gamma + 1.0)) * (M1**2 - 1.0)
    rho2_rho1 = ((gamma + 1.0) * M1**2) / ((gamma - 1.0) * M1**2 + 2.0)
    T2_T1 = p2_p1 / rho2_rho1
    M2 = np.sqrt((1.0 + 0.5 * (gamma - 1.0) * M1**2) / (gamma * M1**2 - 0.5 * (gamma - 1.0)))
    return {"p2_p1": float(p2_p1), "rho2_rho1": float(rho2_rho1), "T2_T1": float(T2_T1), "M2": float(M2)}


@njit(cache=True)
def _euler_flux(U: NDArray[np.float64], gamma: float) -> NDArray[np.float64]:
    """Physical flux of the 1D Euler equations in conservative variables.

    Parameters
    ----------
    U : ndarray of float, shape (3,)
        Conservative state ``(rho, rho*u, E)``.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    ndarray of float, shape (3,)
        Flux ``(rho*u, rho*u^2 + p, u*(E + p))``.
    """
    rho, mom, E = U[0], U[1], U[2]
    u = mom / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * u * u)
    flux = np.empty(3)
    flux[0] = mom
    flux[1] = mom * u + p
    flux[2] = u * (E + p)
    return flux


@njit(cache=True)
def _lax_friedrichs_evolve(U0: NDArray[np.float64], dx: float, dt: float, gamma: float, steps: int) -> NDArray[np.float64]:
    """Evolve the 1D Euler equations with the Lax-Friedrichs finite-volume scheme.

    Parameters
    ----------
    U0 : ndarray of float, shape (nx, 3)
        Initial conservative state at each cell.
    dx : float
        Cell width.
    dt : float
        Time step (assumed to already satisfy the CFL condition).
    gamma : float
        Ratio of specific heats.
    steps : int
        Number of time steps to advance.

    Returns
    -------
    ndarray of float, shape (nx, 3)
        Conservative state after `steps` steps. The two boundary cells are
        held fixed at their initial (far-field) values, approximating an
        open/transmissive boundary over the short times these simulations run.
    """
    nx = U0.shape[0]
    U = U0.copy()
    lam = dt / dx
    for _ in range(steps):
        U_new = U.copy()
        for i in range(1, nx - 1):
            fL = _euler_flux(U[i - 1], gamma)
            fR = _euler_flux(U[i + 1], gamma)
            U_new[i] = 0.5 * (U[i - 1] + U[i + 1]) - 0.5 * lam * (fR - fL)
        U = U_new
    return U


def sod_shock_tube(nx: int = 400, x0: float = 0.5, t_final: float = 0.2, gamma: float = 1.4, cfl: float = 0.5) -> dict[str, NDArray[np.float64]]:
    """Solve the classic Sod shock-tube problem with a Lax-Friedrichs finite-volume scheme.

    The Sod (1978) problem is the standard test Riemann problem for a
    compressible-flow solver: a diaphragm at `x0` initially separates high
    pressure/density gas (left, at rest) from low pressure/density gas
    (right, at rest); removing the diaphragm at :math:`t=0` produces, for
    :math:`t>0`, exactly three simple waves fanning out from `x0` -- a
    left-running rarefaction (expansion) fan, a right-running contact
    discontinuity (a density jump with continuous pressure and velocity),
    and a right-running shock satisfying
    :func:`rankine_hugoniot_jump_conditions` -- against which any
    finite-volume scheme's numerical diffusion and shock-capturing behavior
    can be judged. The domain is ``[0, 1]``; the classic Sod initial
    condition is ``(rho, u, p) = (1, 0, 1)`` for ``x < x0`` and
    ``(0.125, 0, 0.1)`` for ``x >= x0``.

    Parameters
    ----------
    nx : int, default 400
        Number of finite-volume cells.
    x0 : float, default 0.5
        Initial diaphragm location.
    t_final : float, default 0.2
        Time to integrate to.
    gamma : float, default 1.4
        Ratio of specific heats.
    cfl : float, default 0.5
        Courant number; the time step is chosen adaptively each macro-chunk
        of the integration as ``cfl * dx / max(|u| + c)``, and must satisfy
        ``0 < cfl <= 1`` for the explicit scheme to be stable.

    Returns
    -------
    dict
        ``{"x": ..., "rho": ..., "u": ..., "p": ...}`` at `t_final`, each an
        array of length `nx`.

    Raises
    ------
    InvalidParameterError
        If `nx` is smaller than 4, `t_final` is not positive, or `cfl` is
        not in ``(0, 1]``.

    Examples
    --------
    >>> result = sod_shock_tube(nx=200, t_final=0.15)
    >>> result["rho"].shape
    (200,)
    >>> bool(result["rho"][0] > result["rho"][-1] > 0)  # left state denser than right
    True
    """
    if nx < 4:
        raise InvalidParameterError(f"nx must be at least 4, got {nx}")
    if t_final <= 0:
        raise InvalidParameterError(f"t_final must be positive, got {t_final}")
    if not 0.0 < cfl <= 1.0:
        raise InvalidParameterError(f"cfl must be in (0, 1], got {cfl}")

    dx = 1.0 / nx
    x = (np.arange(nx) + 0.5) * dx
    rho = np.where(x < x0, 1.0, 0.125)
    u = np.zeros(nx)
    p = np.where(x < x0, 1.0, 0.1)

    U = np.empty((nx, 3))
    U[:, 0] = rho
    U[:, 1] = rho * u
    U[:, 2] = p / (gamma - 1.0) + 0.5 * rho * u**2

    t = 0.0
    while t < t_final:
        rho_now = U[:, 0]
        u_now = U[:, 1] / rho_now
        p_now = (gamma - 1.0) * (U[:, 2] - 0.5 * rho_now * u_now**2)
        c_now = np.sqrt(gamma * p_now / rho_now)
        dt = cfl * dx / np.max(np.abs(u_now) + c_now)
        dt = min(dt, t_final - t)
        chunk_steps = 1
        U = _lax_friedrichs_evolve(U, dx, dt, gamma, chunk_steps)
        t += dt

    rho = U[:, 0]
    u = U[:, 1] / rho
    p = (gamma - 1.0) * (U[:, 2] - 0.5 * rho * u**2)
    return {"x": x, "rho": rho, "u": u, "p": p}
