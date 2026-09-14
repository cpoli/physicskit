"""Classic exact solutions of the viscous (Navier-Stokes) equations.

Four textbook problems simple enough to solve in closed form (or, for
Blasius, by reducing an entire boundary layer to a single ordinary
differential equation): plane Couette flow, plane Poiseuille flow, Stokes
drag on a sphere, and the Blasius laminar boundary layer. Each is a limit
where the full nonlinear Navier-Stokes equations collapse to something
tractable -- steady unidirectional shear flow (Couette, Poiseuille),
vanishing Reynolds number (Stokes), or a self-similar boundary layer
(Blasius) -- and each remains a standard benchmark for validating a general
solver, since it is one of the few cases where "the answer" is known exactly.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from physicskit.fluids.exceptions import InvalidParameterError

__all__ = [
    "couette_flow_velocity",
    "poiseuille_flow_velocity",
    "poiseuille_flow_rate",
    "stokes_drag",
    "blasius_solve",
    "blasius_boundary_layer_thickness",
    "blasius_skin_friction_coefficient",
]


def couette_flow_velocity(y: ArrayLike, U_wall: float, h: float) -> NDArray[np.float64]:
    """Velocity profile of steady plane Couette flow.

    The flow between two infinite parallel plates, the lower one at rest
    and the upper one (at ``y = h``) moving at speed `U_wall`, with no
    imposed pressure gradient. The steady Navier-Stokes equations reduce to
    :math:`\\mu\\, d^2u/dy^2 = 0`, whose solution subject to no-slip at both
    walls is the linear profile below -- shear stress
    :math:`\\tau = \\mu\\,U_{wall}/h` is exactly uniform across the gap.

    Parameters
    ----------
    y : array_like of float
        Height above the lower (stationary) plate, ``0 <= y <= h``.
    U_wall : float
        Speed of the upper, moving plate.
    h : float
        Gap between the plates.

    Returns
    -------
    ndarray of float
        Velocity :math:`u(y) = U_{wall}\\,y/h`.

    Raises
    ------
    InvalidParameterError
        If `h` is not positive.

    Examples
    --------
    >>> couette_flow_velocity(y=[0.0, 0.5, 1.0], U_wall=2.0, h=1.0)
    array([0., 1., 2.])
    """
    if h <= 0:
        raise InvalidParameterError(f"h must be positive, got {h}")
    y = np.asarray(y, dtype=np.float64)
    return U_wall * y / h


def poiseuille_flow_velocity(y: ArrayLike, dpdx: float, mu: float, h: float) -> NDArray[np.float64]:
    """Velocity profile of steady plane Poiseuille flow.

    The flow between two stationary infinite parallel plates driven by a
    constant imposed pressure gradient `dpdx`. The steady Navier-Stokes
    equations reduce to :math:`\\mu\\,d^2u/dy^2 = dp/dx`, whose solution
    subject to no-slip at both walls (``y=0`` and ``y=h``) is the parabolic
    profile below, maximal at the channel centerline.

    Parameters
    ----------
    y : array_like of float
        Height above the lower plate, ``0 <= y <= h``.
    dpdx : float
        Imposed pressure gradient (negative drives flow in ``+x``).
    mu : float
        Dynamic viscosity.
    h : float
        Gap between the plates.

    Returns
    -------
    ndarray of float
        Velocity :math:`u(y) = -\\frac{1}{2\\mu}\\frac{dp}{dx}\\,y\\,(h-y)`.

    Raises
    ------
    InvalidParameterError
        If `mu` or `h` is not positive.

    See Also
    --------
    poiseuille_flow_rate : The volumetric flow rate driven by this profile.

    Examples
    --------
    >>> y = np.linspace(0, 1, 5)
    >>> u = poiseuille_flow_velocity(y, dpdx=-8.0, mu=1.0, h=1.0)
    >>> round(float(u[2]), 4)  # peak at centerline
    1.0
    >>> bool(u[0] == 0.0 and u[-1] == 0.0)  # no-slip at both walls
    True
    """
    if mu <= 0:
        raise InvalidParameterError(f"mu must be positive, got {mu}")
    if h <= 0:
        raise InvalidParameterError(f"h must be positive, got {h}")
    y = np.asarray(y, dtype=np.float64)
    return -(1.0 / (2.0 * mu)) * dpdx * y * (h - y)


def poiseuille_flow_rate(dpdx: float, mu: float, h: float) -> float:
    """Volumetric flow rate per unit depth of plane Poiseuille flow.

    The integral of :func:`poiseuille_flow_velocity` across the channel:

    .. math::

        Q = -\\frac{h^3}{12\\mu}\\frac{dp}{dx}

    Parameters
    ----------
    dpdx : float
        Imposed pressure gradient.
    mu : float
        Dynamic viscosity.
    h : float
        Gap between the plates.

    Returns
    -------
    float
        Volumetric flow rate per unit depth, `Q`.

    Raises
    ------
    InvalidParameterError
        If `mu` or `h` is not positive.

    Examples
    --------
    >>> round(poiseuille_flow_rate(dpdx=-12.0, mu=1.0, h=1.0), 4)
    1.0
    """
    if mu <= 0:
        raise InvalidParameterError(f"mu must be positive, got {mu}")
    if h <= 0:
        raise InvalidParameterError(f"h must be positive, got {h}")
    return -(h**3 / (12.0 * mu)) * dpdx


def stokes_drag(mu: float, radius: float, velocity: float) -> float:
    """Stokes' law: drag force on a sphere at low Reynolds number.

    .. math::

        F_D = 6\\pi\\mu R v

    Valid in the creeping-flow limit :math:`Re = 2R|v|/\\nu \\ll 1`, where
    the nonlinear advection term in Navier-Stokes is negligible next to
    viscous diffusion, leaving the linear Stokes equations
    :math:`\\mu\\nabla^2\\mathbf{u}=\\nabla p` -- George Stokes' 1851 exact
    solution for uniform flow past a sphere.

    Parameters
    ----------
    mu : float
        Dynamic viscosity.
    radius : float
        Sphere radius.
    velocity : float
        Relative speed between the sphere and the fluid far away.

    Returns
    -------
    float
        Drag force magnitude :math:`F_D`.

    Raises
    ------
    InvalidParameterError
        If `mu` or `radius` is not positive.

    Examples
    --------
    >>> round(stokes_drag(mu=1.0, radius=1.0, velocity=1.0), 4)
    18.8496
    """
    if mu <= 0:
        raise InvalidParameterError(f"mu must be positive, got {mu}")
    if radius <= 0:
        raise InvalidParameterError(f"radius must be positive, got {radius}")
    return 6.0 * np.pi * mu * radius * velocity


@njit(cache=True)
def _blasius_rhs(state: NDArray[np.float64]) -> NDArray[np.float64]:
    """Blasius ODE as a first-order system: ``(f, f', f'')' = (f', f'', -f f''/2)``.

    Parameters
    ----------
    state : ndarray of float, shape (3,)
        ``(f, f', f'')`` at the current similarity coordinate `eta`.

    Returns
    -------
    ndarray of float, shape (3,)
        Derivative of `state` with respect to `eta`.
    """
    f, fp, fpp = state[0], state[1], state[2]
    out = np.empty(3)
    out[0] = fp
    out[1] = fpp
    out[2] = -0.5 * f * fpp
    return out


@njit(cache=True)
def _blasius_integrate(fpp0: float, eta_max: float, n_points: int) -> NDArray[np.float64]:
    """Integrate the Blasius ODE with RK4 from ``eta=0`` given a guessed ``f''(0)``.

    Parameters
    ----------
    fpp0 : float
        Guessed shooting parameter :math:`f''(0)`.
    eta_max : float
        Similarity coordinate at which to stop integrating.
    n_points : int
        Number of output points (including ``eta=0``).

    Returns
    -------
    ndarray of float, shape (n_points, 3)
        ``(f, f', f'')`` at each of `n_points` equally spaced `eta` in
        ``[0, eta_max]``.
    """
    states = np.empty((n_points, 3))
    state = np.array([0.0, 0.0, fpp0])
    states[0] = state
    deta = eta_max / (n_points - 1)
    for i in range(n_points - 1):
        k1 = _blasius_rhs(state)
        k2 = _blasius_rhs(state + 0.5 * deta * k1)
        k3 = _blasius_rhs(state + 0.5 * deta * k2)
        k4 = _blasius_rhs(state + deta * k3)
        state = state + (deta / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        states[i + 1] = state
    return states


def blasius_solve(eta_max: float = 10.0, n_points: int = 2001, tol: float = 1e-10, max_iter: int = 100) -> dict[str, NDArray[np.float64]]:
    """Solve the Blasius laminar boundary-layer equation by shooting.

    The Blasius similarity reduction of the steady, 2D, zero-pressure-gradient
    boundary-layer equations collapses them to the single third-order ODE

    .. math::

        f''' + \\tfrac{1}{2} f f'' = 0, \\qquad
        f(0) = f'(0) = 0, \\qquad f'(\\infty) = 1,

    for the dimensionless streamfunction :math:`f(\\eta)` of the similarity
    variable :math:`\\eta = y\\sqrt{U_\\infty/(\\nu x)}`; the streamwise
    velocity is :math:`u/U_\\infty = f'(\\eta)`. Since :math:`f''(0)` (rather
    than the far-field condition) is the free parameter needed to start an
    initial-value integration, this is a two-point boundary value problem
    solved here by shooting: bisect on :math:`f''(0)` until the resulting
    :func:`_blasius_integrate` solution satisfies :math:`f'(\\eta_{max})
    \\approx 1`.

    Parameters
    ----------
    eta_max : float, default 10.0
        Similarity coordinate treated as "infinity"; large enough that
        :math:`f'` has converged to 1 to within `tol`.
    n_points : int, default 2001
        Number of output points between ``eta=0`` and `eta_max`.
    tol : float, default 1e-10
        Bisection tolerance on :math:`f'(\\eta_{max}) - 1`.
    max_iter : int, default 100
        Maximum number of bisection iterations.

    Returns
    -------
    dict
        ``{"eta": ..., "f": ..., "fp": ..., "fpp": ...}``, each an array of
        length `n_points`. ``fpp[0]`` is the classic Blasius constant,
        :math:`f''(0) \\approx 0.332`.

    Examples
    --------
    >>> result = blasius_solve()
    >>> round(float(result["fpp"][0]), 3)
    0.332
    >>> round(float(result["fp"][-1]), 4)
    1.0
    """
    lo, hi = 0.1, 1.0
    solution = _blasius_integrate(hi, eta_max, n_points)
    while solution[-1, 1] < 1.0:
        hi *= 2.0
        solution = _blasius_integrate(hi, eta_max, n_points)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        solution = _blasius_integrate(mid, eta_max, n_points)
        residual = solution[-1, 1] - 1.0
        if abs(residual) < tol:
            break
        if residual > 0:
            hi = mid
        else:
            lo = mid

    eta = np.linspace(0.0, eta_max, n_points)
    return {"eta": eta, "f": solution[:, 0], "fp": solution[:, 1], "fpp": solution[:, 2]}


def blasius_boundary_layer_thickness(x: ArrayLike, U_inf: float, nu: float, eta_99: float = 4.91) -> NDArray[np.float64]:
    """Blasius laminar boundary-layer thickness :math:`\\delta_{99}(x)`.

    The height at which :math:`u/U_\\infty` first reaches 0.99 occurs, in
    similarity coordinates, at :math:`\\eta \\approx 4.91` (a property of
    the Blasius profile from :func:`blasius_solve`, not re-derived here), so

    .. math::

        \\delta_{99}(x) = \\eta_{99}\\sqrt{\\frac{\\nu x}{U_\\infty}}
            \\propto \\sqrt{x}.

    The boundary layer's parabolic-looking growth with downstream distance
    -- much slower than the layer's own streamwise extent -- is exactly the
    "thin layer" assumption Ludwig Prandtl's 1904 boundary-layer theory
    used to simplify Navier-Stokes into the boundary-layer equations Blasius
    then solved.

    Parameters
    ----------
    x : array_like of float
        Downstream distance from the leading edge; must be positive.
    U_inf : float
        Free-stream speed.
    nu : float
        Kinematic viscosity.
    eta_99 : float, default 4.91
        Similarity coordinate at which :math:`f'(\\eta)=0.99`.

    Returns
    -------
    ndarray of float
        Boundary-layer thickness :math:`\\delta_{99}(x)`.

    Raises
    ------
    InvalidParameterError
        If `U_inf` or `nu` is not positive, or any entry of `x` is not positive.

    Examples
    --------
    >>> round(float(blasius_boundary_layer_thickness(x=1.0, U_inf=1.0, nu=1e-4)), 4)
    0.0491
    """
    if U_inf <= 0:
        raise InvalidParameterError(f"U_inf must be positive, got {U_inf}")
    if nu <= 0:
        raise InvalidParameterError(f"nu must be positive, got {nu}")
    x = np.asarray(x, dtype=np.float64)
    if np.any(x <= 0):
        raise InvalidParameterError("x must be positive (measured from the leading edge)")
    return eta_99 * np.sqrt(nu * x / U_inf)


def blasius_skin_friction_coefficient(reynolds_x: ArrayLike, fpp0: float = 0.33206) -> NDArray[np.float64]:
    """Local skin-friction coefficient of the Blasius boundary layer.

    The wall shear stress :math:`\\tau_w = \\mu U_\\infty f''(0)\\sqrt{U_\\infty/(\\nu x)}`
    gives a local skin-friction coefficient

    .. math::

        c_f(x) = \\frac{\\tau_w}{\\tfrac{1}{2}\\rho U_\\infty^2}
            = \\frac{2 f''(0)}{\\sqrt{Re_x}}, \\qquad
        Re_x = \\frac{U_\\infty x}{\\nu},

    the classic :math:`c_f \\propto Re_x^{-1/2}` scaling of a laminar
    boundary layer.

    Parameters
    ----------
    reynolds_x : array_like of float
        Local Reynolds number :math:`Re_x = U_\\infty x/\\nu`; must be positive.
    fpp0 : float, default 0.33206
        The Blasius constant :math:`f''(0)`, as returned by :func:`blasius_solve`.

    Returns
    -------
    ndarray of float
        Local skin-friction coefficient :math:`c_f(x)`.

    Raises
    ------
    InvalidParameterError
        If any entry of `reynolds_x` is not positive.

    Examples
    --------
    >>> round(float(blasius_skin_friction_coefficient(reynolds_x=1e4)), 5)
    0.00664
    """
    reynolds_x = np.asarray(reynolds_x, dtype=np.float64)
    if np.any(reynolds_x <= 0):
        raise InvalidParameterError("reynolds_x must be positive")
    return 2.0 * fpp0 / np.sqrt(reynolds_x)
