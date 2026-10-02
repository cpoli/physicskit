"""The shallow-water (Saint-Venant) equations: dam breaks and geostrophic adjustment.

When the horizontal scale of a flow is much larger than its depth
:math:`h`, vertical accelerations are negligible, the pressure is
hydrostatic, and depth-averaging the Euler equations gives (Saint-Venant,
C. R. Acad. Sci. 73, 147 (1871))

.. math::

    \\partial_t h + \\partial_x(hu) = 0, \\qquad
    \\partial_t(hu) + \\partial_x\\left(hu^2 + \\tfrac12 g h^2\\right) = 0,

a hyperbolic system with wave speed :math:`c = \\sqrt{gh}`, isomorphic to
1D isentropic gas dynamics with :math:`\\gamma = 2`. A dam break, the
Riemann problem of these equations, is solved exactly by a rarefaction
fan and a bore (Ritter 1892 for a dry bed, Stoker 1957 for a wet one).

On a rotating planet the linearized equations about rest depth :math:`H`
pick up the Coriolis terms,

.. math::

    \\partial_t u - fv = -g\\partial_x\\eta, \\qquad \\partial_t v + fu = 0,
    \\qquad \\partial_t\\eta + H\\partial_x u = 0,

and an initial step in the surface height :math:`\\eta` does not spread
out completely, as it would without rotation. Instead it adjusts to a
geostrophically balanced front of width the Rossby radius
:math:`L_d = \\sqrt{gH}/f`, fixed by conservation of potential vorticity
:math:`\\partial_x v - f\\eta/H` (Rossby, J. Mar. Res. 1, 239 (1938);
Gill, *Atmosphere-Ocean Dynamics*, 1982, sec. 7.2).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from physicskit.fluids.exceptions import InvalidParameterError

__all__ = [
    "dam_break_exact",
    "geostrophic_adjustment_steady",
    "rotating_shallow_water_1d",
    "shallow_water_1d",
]


def _hll_flux(hL, qL, hR, qR, g):
    uL = np.where(hL > 0, qL / np.maximum(hL, 1e-300), 0.0)
    uR = np.where(hR > 0, qR / np.maximum(hR, 1e-300), 0.0)
    cL, cR = np.sqrt(g * hL), np.sqrt(g * hR)
    sL = np.minimum(uL - cL, uR - cR)
    sR = np.maximum(uL + cL, uR + cR)
    FL = np.array([qL, qL * uL + 0.5 * g * hL**2])
    FR = np.array([qR, qR * uR + 0.5 * g * hR**2])
    UL = np.array([hL, qL])
    UR = np.array([hR, qR])
    denom = np.where(sR - sL > 0, sR - sL, 1.0)
    F_hll = (sR * FL - sL * FR + sL * sR * (UR - UL)) / denom
    return np.where(sL >= 0, FL, np.where(sR <= 0, FR, F_hll)), np.maximum(np.abs(sL), np.abs(sR)).max()


def shallow_water_1d(
    h0: NDArray[np.float64],
    u0: NDArray[np.float64],
    x: NDArray[np.float64],
    t_max: float,
    g: float = 9.81,
    cfl: float = 0.45,
) -> dict[str, NDArray[np.float64]]:
    """Solve the 1D shallow-water equations with a finite-volume HLL scheme.

    Second order in space (minmod-limited MUSCL reconstruction of
    :math:`h` and :math:`hu`) and time (Heun), with transmissive boundaries.
    Dry cells (:math:`h = 0`) are allowed.

    Parameters
    ----------
    h0, u0 : ndarray of float, shape (n,)
        Initial depth and velocity at the cell centres ``x``.
    x : ndarray of float, shape (n,)
        Uniformly spaced cell centres.
    t_max : float
        Final time.
    g : float, default 9.81
    cfl : float, default 0.45
        Courant number.

    Returns
    -------
    dict
        ``"h"`` and ``"u"`` at ``t_max``, and the number of ``"steps"``.

    Examples
    --------
    >>> x = np.linspace(-1, 1, 200)
    >>> out = shallow_water_1d(np.where(x < 0, 1.0, 0.5), np.zeros_like(x), x, t_max=0.1)
    >>> bool(abs(out["h"].mean() - 0.75) < 1e-2)  # mass is conserved away from the ends
    True
    """
    h = np.asarray(h0, dtype=np.float64).copy()
    q = h * np.asarray(u0, dtype=np.float64)
    if np.any(h < 0):
        raise InvalidParameterError("depth must be non-negative")
    dx = x[1] - x[0]

    def minmod(a, b):
        return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)

    def rhs(h, q):
        hp = np.concatenate(([h[0], h[0]], h, [h[-1], h[-1]]))
        qp = np.concatenate(([q[0], q[0]], q, [q[-1], q[-1]]))
        sh = minmod(hp[1:-1] - hp[:-2], hp[2:] - hp[1:-1])
        sq = minmod(qp[1:-1] - qp[:-2], qp[2:] - qp[1:-1])
        # left and right states at each of the n + 1 interfaces
        hL = hp[1:-2] + 0.5 * sh[:-1]
        hR = hp[2:-1] - 0.5 * sh[1:]
        qL = qp[1:-2] + 0.5 * sq[:-1]
        qR = qp[2:-1] - 0.5 * sq[1:]
        hL, hR = np.maximum(hL, 0.0), np.maximum(hR, 0.0)
        qL, qR = np.where(hL > 1e-12, qL, 0.0), np.where(hR > 1e-12, qR, 0.0)
        F, smax = _hll_flux(hL, qL, hR, qR, g)
        return -(F[0, 1:] - F[0, :-1]) / dx, -(F[1, 1:] - F[1, :-1]) / dx, smax

    t = 0.0
    steps = 0
    while t < t_max:
        dh1, dq1, smax = rhs(h, q)
        dt = min(cfl * dx / max(smax, 1e-12), t_max - t)
        h1 = np.maximum(h + dt * dh1, 0.0)
        q1 = q + dt * dq1
        dh2, dq2, _ = rhs(h1, q1)
        h = np.maximum(0.5 * (h + h1 + dt * dh2), 0.0)
        q = np.where(h > 1e-12, 0.5 * (q + q1 + dt * dq2), 0.0)
        t += dt
        steps += 1
    u = np.where(h > 1e-12, q / np.maximum(h, 1e-300), 0.0)
    return {"h": h, "u": u, "steps": np.array(steps)}


def dam_break_exact(x: NDArray[np.float64], t: float, h_left: float, h_right: float = 0.0, g: float = 9.81) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Exact solution of the dam break at :math:`x = 0`, released at :math:`t = 0`.

    A rarefaction fan :math:`h = (2c_L - x/t)^2/(9g)`,
    :math:`u = \\tfrac23(c_L + x/t)` connects the reservoir to a uniform
    state :math:`(h_m, u_m)`, which is joined to the tailwater by a bore of
    speed :math:`s = h_m u_m/(h_m - h_R)`. :math:`h_m` solves

    .. math::

        2(c_L - c_m) = (h_m - h_R)\\sqrt{\\frac{g(h_m + h_R)}{2h_mh_R}}

    (Stoker, *Water Waves*, 1957). For a dry bed, :math:`h_R = 0`, the fan
    reaches the wet front :math:`x = 2c_Lt` (Ritter 1892).

    Parameters
    ----------
    x : ndarray of float
        Positions.
    t : float
        Time after the release, positive.
    h_left : float
        Reservoir depth.
    h_right : float, default 0.0
        Tailwater depth, ``0 <= h_right < h_left``.
    g : float, default 9.81

    Returns
    -------
    h, u : ndarray of float

    Examples
    --------
    >>> h, u = dam_break_exact(np.array([0.0]), t=1.0, h_left=1.0)
    >>> round(float(h[0]), 6)  # Ritter: h = 4 h_L / 9 at the dam site
    0.444444
    """
    if not 0.0 <= h_right < h_left:
        raise InvalidParameterError("need 0 <= h_right < h_left")
    x = np.asarray(x, dtype=np.float64)
    cL = np.sqrt(g * h_left)
    xi = x / t
    h = np.full_like(xi, h_left)
    u = np.zeros_like(xi)
    fan = (xi > -cL) & (xi < 2 * cL)
    h_fan = (2 * cL - xi) ** 2 / (9 * g)
    u_fan = 2.0 / 3.0 * (cL + xi)
    if h_right == 0.0:
        h[fan] = h_fan[fan]
        u[fan] = u_fan[fan]
        h[xi >= 2 * cL] = 0.0
        return h, u

    def mismatch(hm):
        return 2 * (cL - np.sqrt(g * hm)) - (hm - h_right) * np.sqrt(g * (hm + h_right) / (2 * hm * h_right))

    hm = brentq(mismatch, h_right, h_left)
    cm = np.sqrt(g * hm)
    um = 2 * (cL - cm)
    s = hm * um / (hm - h_right)
    in_fan = (xi > -cL) & (xi < um - cm)
    h[in_fan] = h_fan[in_fan]
    u[in_fan] = u_fan[in_fan]
    middle = (xi >= um - cm) & (xi < s)
    h[middle] = hm
    u[middle] = um
    h[xi >= s] = h_right
    return h, u


def rotating_shallow_water_1d(
    eta0: NDArray[np.float64],
    x: NDArray[np.float64],
    times: NDArray[np.float64],
    g: float = 9.81,
    H: float = 1.0,
    f: float = 1.0,
) -> dict[str, NDArray[np.float64]]:
    """Exact evolution of the linear rotating shallow-water equations on a periodic line.

    Each Fourier mode of :math:`(u, v, \\eta)`, starting from rest, splits
    into a steady geostrophic part and inertia-gravity (Poincaré) waves of
    frequency :math:`\\omega_k = \\sqrt{f^2 + gHk^2}`, whose evolution is
    computed exactly:

    .. math::

        \\hat\\eta_k(t) = \\hat\\eta_k(0)\\,\\frac{f^2 + gHk^2\\cos\\omega_k t}{\\omega_k^2},\\quad
        \\hat u_k = -\\frac{igk\\hat\\eta_k(0)}{\\omega_k}\\sin\\omega_k t,\\quad
        \\hat v_k = \\frac{igkf\\hat\\eta_k(0)}{\\omega_k^2}(1 - \\cos\\omega_k t).

    Parameters
    ----------
    eta0 : ndarray of float, shape (n,)
        Initial surface displacement, fluid at rest.
    x : ndarray of float, shape (n,)
        Uniform periodic grid.
    times : array_like
        Output times.
    g, H, f : float
        Gravity, mean depth and Coriolis parameter.

    Returns
    -------
    dict
        ``"eta"``, ``"u"``, ``"v"``, each of shape ``(len(times), n)``.

    Examples
    --------
    >>> x = np.linspace(0, 10, 64, endpoint=False)
    >>> out = rotating_shallow_water_1d(np.cos(2 * np.pi * x / 10), x, [0.0], g=1.0)
    >>> bool(np.allclose(out["eta"][0], np.cos(2 * np.pi * x / 10)))
    True
    """
    n = x.size
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    eh = np.fft.fft(eta0)
    w = np.sqrt(f**2 + g * H * k**2)
    times = np.atleast_1d(np.asarray(times, dtype=np.float64))[:, None]
    c, s = np.cos(w * times), np.sin(w * times)
    eta = np.real(np.fft.ifft(eh * (f**2 + g * H * k**2 * c) / w**2, axis=1))
    u = np.real(np.fft.ifft(-1j * g * k * eh * s / w, axis=1))
    v = np.real(np.fft.ifft(1j * g * k * f * eh * (1 - c) / w**2, axis=1))
    return {"eta": eta, "u": u, "v": v}


def geostrophic_adjustment_steady(
    eta0: NDArray[np.float64], x: NDArray[np.float64], g: float = 9.81, H: float = 1.0, f: float = 1.0
) -> dict[str, NDArray[np.float64]]:
    """Balanced end state of Rossby's adjustment problem, from potential-vorticity conservation.

    The linear potential vorticity :math:`\\partial_x v - f\\eta/H` is
    conserved, and the final state is in geostrophic balance, :math:`fv =
    g\\partial_x\\eta`, :math:`u = 0`, so

    .. math::

        \\eta - L_d^2\\,\\partial_x^2\\eta = \\eta_0, \\qquad L_d = \\sqrt{gH}/f.

    For a step of height :math:`2\\eta_0` this gives the front
    :math:`\\eta = \\eta_0\\,\\mathrm{sgn}(x)(1 - e^{-|x|/L_d})`.

    Parameters
    ----------
    eta0 : ndarray of float, shape (n,)
        Initial displacement on the periodic grid ``x``.
    x : ndarray of float, shape (n,)
    g, H, f : float

    Returns
    -------
    dict
        ``"eta"`` and ``"v"`` of the adjusted state, and the Rossby radius
        ``"L_d"``.

    Examples
    --------
    >>> x = np.linspace(0, 100, 512, endpoint=False)
    >>> out = geostrophic_adjustment_steady(np.cos(2 * np.pi * x / 100), x, g=1.0)
    >>> round(float(out["eta"].max()), 4)  # 1 / (1 + (L_d k)^2)
    0.9961
    """
    k = 2 * np.pi * np.fft.fftfreq(x.size, d=x[1] - x[0])
    L_d = np.sqrt(g * H) / f
    eh = np.fft.fft(eta0) / (1 + (L_d * k) ** 2)
    eta = np.real(np.fft.ifft(eh))
    v = np.real(np.fft.ifft(1j * k * eh)) * g / f
    return {"eta": eta, "v": v, "L_d": np.array(L_d)}
