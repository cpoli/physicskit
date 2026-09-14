"""A numerical differential geometry engine for arbitrary 4D spacetimes.

Rather than requiring a symbolic algebra system, this module computes
Christoffel symbols and curvature tensors **numerically**, by finite
differencing a metric function :math:`g_{\\mu\\nu}(x)` supplied as a plain
Python callable. This makes it trivial to plug in any metric -- Schwarzschild,
Kerr, Reissner-Nordstrom, FLRW, even an exotic metric like the Alcubierre
warp drive -- and get its curvature out, at the cost of some numerical noise
relative to an exact symbolic computation (typically negligible at the
default step size for smooth metrics away from coordinate singularities).

A metric function has the signature ``g(coords, **params) -> ndarray`` of
shape ``(4, 4)``, where ``coords`` is a length-4 array of the four spacetime
coordinates at which to evaluate the metric.

Every function in this module follows the :math:`(-,+,+,+)` signature
convention.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "alcubierre_metric",
    "christoffel_symbols",
    "einstein_tensor",
    "flrw_metric",
    "inverse_metric",
    "kerr_metric_bl",
    "reissner_nordstrom_metric",
    "ricci_scalar",
    "ricci_tensor",
    "riemann_tensor",
    "schwarzschild_metric",
]


def inverse_metric(metric_func, coords, params=None):
    """Inverse metric :math:`g^{\\mu\\nu}` at a point.

    Parameters
    ----------
    metric_func : callable
        A metric function ``g(coords, **params) -> ndarray`` of shape (4, 4).
    coords : array_like of shape (4,)
        Coordinates at which to evaluate the metric.
    params : dict, optional
        Keyword parameters forwarded to ``metric_func`` (e.g. ``{"M": 1.0}``).

    Returns
    -------
    ndarray of shape (4, 4)
        The inverse metric :math:`g^{\\mu\\nu}`.
    """
    params = params or {}
    g = metric_func(coords, **params)
    return np.linalg.inv(g)


def christoffel_symbols(metric_func, coords, params=None, h=1e-6):
    """Christoffel symbols of the second kind, :math:`\\Gamma^\\mu_{\\alpha\\beta}`.

    .. math::

        \\Gamma^\\mu_{\\alpha\\beta} = \\frac{1}{2} g^{\\mu\\nu}
            \\left(\\partial_\\alpha g_{\\nu\\beta} + \\partial_\\beta g_{\\nu\\alpha}
            - \\partial_\\nu g_{\\alpha\\beta}\\right)

    Metric derivatives are estimated with a central finite difference.

    Parameters
    ----------
    metric_func : callable
        A metric function ``g(coords, **params) -> ndarray`` of shape (4, 4).
    coords : array_like of shape (4,)
        Coordinates at which to evaluate the Christoffel symbols.
    params : dict, optional
        Keyword parameters forwarded to ``metric_func``.
    h : float, default=1e-6
        Finite-difference step size.

    Returns
    -------
    ndarray of shape (4, 4, 4)
        ``Gamma[mu, alpha, beta]`` :math:`= \\Gamma^\\mu_{\\alpha\\beta}`.

    Examples
    --------
    >>> coords = np.array([0.0, 10.0, np.pi / 2, 0.0])
    >>> Gamma = christoffel_symbols(schwarzschild_metric, coords, {"M": 1.0})
    >>> Gamma.shape
    (4, 4, 4)
    """
    params = params or {}
    coords = np.asarray(coords, dtype=np.float64)
    g_inv = inverse_metric(metric_func, coords, params)

    dg = np.empty((4, 4, 4))  # dg[alpha, mu, nu] = d_alpha g_{mu nu}
    for alpha in range(4):
        shift = np.zeros(4)
        shift[alpha] = h
        g_plus = metric_func(coords + shift, **params)
        g_minus = metric_func(coords - shift, **params)
        dg[alpha] = (g_plus - g_minus) / (2.0 * h)

    Gamma = np.zeros((4, 4, 4))
    for mu in range(4):
        for alpha in range(4):
            for beta in range(4):
                s = 0.0
                for nu in range(4):
                    s += g_inv[mu, nu] * (dg[alpha, nu, beta] + dg[beta, nu, alpha] - dg[nu, alpha, beta])
                Gamma[mu, alpha, beta] = 0.5 * s
    return Gamma


def riemann_tensor(metric_func, coords, params=None, h=1e-4):
    """Riemann curvature tensor :math:`R^\\rho{}_{\\sigma\\mu\\nu}`.

    .. math::

        R^\\rho{}_{\\sigma\\mu\\nu} = \\partial_\\mu \\Gamma^\\rho_{\\nu\\sigma}
            - \\partial_\\nu \\Gamma^\\rho_{\\mu\\sigma}
            + \\Gamma^\\rho_{\\mu\\lambda}\\Gamma^\\lambda_{\\nu\\sigma}
            - \\Gamma^\\rho_{\\nu\\lambda}\\Gamma^\\lambda_{\\mu\\sigma}

    Christoffel derivatives are estimated with a central finite difference of
    :func:`christoffel_symbols` itself, so this involves an effective
    second derivative of the metric; a coarser default step size than
    :func:`christoffel_symbols` is used to balance truncation and
    round-off error.

    Parameters
    ----------
    metric_func : callable
        A metric function ``g(coords, **params) -> ndarray`` of shape (4, 4).
    coords : array_like of shape (4,)
        Coordinates at which to evaluate the Riemann tensor.
    params : dict, optional
        Keyword parameters forwarded to ``metric_func``.
    h : float, default=1e-4
        Finite-difference step size.

    Returns
    -------
    ndarray of shape (4, 4, 4, 4)
        ``R[rho, sigma, mu, nu]``.
    """
    params = params or {}
    coords = np.asarray(coords, dtype=np.float64)

    dGamma = np.empty((4, 4, 4, 4))  # dGamma[mu, rho, nu, sigma] = d_mu Gamma^rho_{nu sigma}
    for mu in range(4):
        shift = np.zeros(4)
        shift[mu] = h
        G_plus = christoffel_symbols(metric_func, coords + shift, params, h=h * 1e-2)
        G_minus = christoffel_symbols(metric_func, coords - shift, params, h=h * 1e-2)
        dGamma[mu] = (G_plus - G_minus) / (2.0 * h)

    Gamma = christoffel_symbols(metric_func, coords, params, h=h * 1e-2)

    R = np.zeros((4, 4, 4, 4))
    for rho in range(4):
        for sigma in range(4):
            for mu in range(4):
                for nu in range(4):
                    term = dGamma[mu, rho, nu, sigma] - dGamma[nu, rho, mu, sigma]
                    for lam in range(4):
                        term += Gamma[rho, mu, lam] * Gamma[lam, nu, sigma]
                        term -= Gamma[rho, nu, lam] * Gamma[lam, mu, sigma]
                    R[rho, sigma, mu, nu] = term
    return R


def ricci_tensor(metric_func, coords, params=None, h=1e-4):
    """Ricci tensor :math:`R_{\\mu\\nu} = R^\\rho{}_{\\mu\\rho\\nu}`, the trace of the Riemann tensor.

    Parameters
    ----------
    metric_func : callable
        A metric function ``g(coords, **params) -> ndarray`` of shape (4, 4).
    coords : array_like of shape (4,)
        Coordinates at which to evaluate the Ricci tensor.
    params : dict, optional
        Keyword parameters forwarded to ``metric_func``.
    h : float, default=1e-4
        Finite-difference step size, forwarded to :func:`riemann_tensor`.

    Returns
    -------
    ndarray of shape (4, 4)
        The Ricci tensor :math:`R_{\\mu\\nu}`.
    """
    R = riemann_tensor(metric_func, coords, params, h=h)
    return np.einsum("rmrn->mn", R)


def ricci_scalar(metric_func, coords, params=None, h=1e-4):
    """Ricci scalar :math:`R = g^{\\mu\\nu} R_{\\mu\\nu}`.

    Parameters
    ----------
    metric_func : callable
        A metric function ``g(coords, **params) -> ndarray`` of shape (4, 4).
    coords : array_like of shape (4,)
        Coordinates at which to evaluate the Ricci scalar.
    params : dict, optional
        Keyword parameters forwarded to ``metric_func``.
    h : float, default=1e-4
        Finite-difference step size, forwarded to :func:`riemann_tensor`.

    Returns
    -------
    float
        The Ricci scalar :math:`R`.
    """
    params = params or {}
    g_inv = inverse_metric(metric_func, coords, params)
    Ric = ricci_tensor(metric_func, coords, params, h=h)
    return float(np.einsum("mn,mn->", g_inv, Ric))


def einstein_tensor(metric_func, coords, params=None, h=1e-4):
    """Einstein tensor :math:`G_{\\mu\\nu} = R_{\\mu\\nu} - \\frac{1}{2} R g_{\\mu\\nu}`.

    In vacuum, Einstein's field equations require :math:`G_{\\mu\\nu} = 0`;
    evaluating this on the Schwarzschild metric and finding it numerically
    close to zero is a good sanity check of both the metric and this
    finite-difference curvature engine (see the Examples).

    Parameters
    ----------
    metric_func : callable
        A metric function ``g(coords, **params) -> ndarray`` of shape (4, 4).
    coords : array_like of shape (4,)
        Coordinates at which to evaluate the Einstein tensor.
    params : dict, optional
        Keyword parameters forwarded to ``metric_func``.
    h : float, default=1e-4
        Finite-difference step size, forwarded to :func:`riemann_tensor`.

    Returns
    -------
    ndarray of shape (4, 4)
        The Einstein tensor :math:`G_{\\mu\\nu}`.

    Examples
    --------
    >>> coords = np.array([0.0, 10.0, np.pi / 2, 0.0])
    >>> G = einstein_tensor(schwarzschild_metric, coords, {"M": 1.0})
    >>> bool(np.max(np.abs(G)) < 1e-2)
    True
    """
    params = params or {}
    g = metric_func(coords, **params)
    Ric = ricci_tensor(metric_func, coords, params, h=h)
    R = ricci_scalar(metric_func, coords, params, h=h)
    return Ric - 0.5 * R * g


# ---------------------------------------------------------------------------
# Standard metrics
# ---------------------------------------------------------------------------


def schwarzschild_metric(coords, M=1.0):
    """The Schwarzschild metric in Schwarzschild coordinates :math:`(t, r, \\theta, \\phi)`.

    .. math::

        ds^2 = -\\left(1 - \\frac{2M}{r}\\right) dt^2
             + \\left(1 - \\frac{2M}{r}\\right)^{-1} dr^2
             + r^2 d\\theta^2 + r^2 \\sin^2\\theta \\, d\\phi^2

    Parameters
    ----------
    coords : array_like of shape (4,)
        Coordinates :math:`(t, r, \\theta, \\phi)`.
    M : float, default=1.0
        Black hole mass, in geometrized units.

    Returns
    -------
    ndarray of shape (4, 4)
        The metric tensor :math:`g_{\\mu\\nu}`.
    """
    _t, r, theta, _phi = coords
    f = 1.0 - 2.0 * M / r
    g = np.zeros((4, 4))
    g[0, 0] = -f
    g[1, 1] = 1.0 / f
    g[2, 2] = r**2
    g[3, 3] = r**2 * np.sin(theta) ** 2
    return g


def kerr_metric_bl(coords, M=1.0, a=0.0):
    """The Kerr metric in Boyer-Lindquist coordinates :math:`(t, r, \\theta, \\phi)`.

    .. math::

        ds^2 = -\\left(1 - \\frac{2Mr}{\\Sigma}\\right) dt^2
             - \\frac{4Mar\\sin^2\\theta}{\\Sigma} dt\\, d\\phi
             + \\frac{\\Sigma}{\\Delta} dr^2 + \\Sigma\\, d\\theta^2
             + \\left(r^2 + a^2 + \\frac{2Ma^2 r \\sin^2\\theta}{\\Sigma}\\right)
               \\sin^2\\theta \\, d\\phi^2

    with :math:`\\Sigma = r^2 + a^2\\cos^2\\theta` and
    :math:`\\Delta = r^2 - 2Mr + a^2`.

    Parameters
    ----------
    coords : array_like of shape (4,)
        Coordinates :math:`(t, r, \\theta, \\phi)`.
    M : float, default=1.0
        Black hole mass, in geometrized units.
    a : float, default=0.0
        Spin parameter :math:`a = J/M`, with :math:`0 \\le a < M`.
        ``a=0`` recovers the Schwarzschild metric.

    Returns
    -------
    ndarray of shape (4, 4)
        The metric tensor :math:`g_{\\mu\\nu}`.
    """
    _t, r, theta, _phi = coords
    sin2 = np.sin(theta) ** 2
    Sigma = r**2 + a**2 * np.cos(theta) ** 2
    Delta = r**2 - 2.0 * M * r + a**2

    g = np.zeros((4, 4))
    g[0, 0] = -(1.0 - 2.0 * M * r / Sigma)
    g[0, 3] = g[3, 0] = -2.0 * M * a * r * sin2 / Sigma
    g[1, 1] = Sigma / Delta
    g[2, 2] = Sigma
    g[3, 3] = (r**2 + a**2 + 2.0 * M * a**2 * r * sin2 / Sigma) * sin2
    return g


def reissner_nordstrom_metric(coords, M=1.0, Q=0.0):
    """The Reissner-Nordstrom metric of a charged, non-rotating black hole.

    .. math::

        ds^2 = -\\left(1 - \\frac{2M}{r} + \\frac{Q^2}{r^2}\\right) dt^2
             + \\left(1 - \\frac{2M}{r} + \\frac{Q^2}{r^2}\\right)^{-1} dr^2
             + r^2 d\\theta^2 + r^2 \\sin^2\\theta \\, d\\phi^2

    Parameters
    ----------
    coords : array_like of shape (4,)
        Coordinates :math:`(t, r, \\theta, \\phi)`.
    M : float, default=1.0
        Mass, in geometrized units.
    Q : float, default=0.0
        Electric charge, in geometrized units (``Q=0`` recovers Schwarzschild).

    Returns
    -------
    ndarray of shape (4, 4)
        The metric tensor :math:`g_{\\mu\\nu}`.
    """
    _t, r, theta, _phi = coords
    f = 1.0 - 2.0 * M / r + Q**2 / r**2
    g = np.zeros((4, 4))
    g[0, 0] = -f
    g[1, 1] = 1.0 / f
    g[2, 2] = r**2
    g[3, 3] = r**2 * np.sin(theta) ** 2
    return g


def flrw_metric(coords, a_func, k=0.0):
    """The FLRW metric in comoving spherical coordinates :math:`(t, \\chi, \\theta, \\phi)`.

    .. math::

        ds^2 = -dt^2 + a(t)^2 \\left[d\\chi^2 + S_k(\\chi)^2 d\\Omega^2\\right]

    where :math:`S_k(\\chi) = \\chi` for a flat universe (:math:`k=0`),
    :math:`\\sin\\chi` for a closed universe (:math:`k=+1`), and
    :math:`\\sinh\\chi` for an open universe (:math:`k=-1`).

    Parameters
    ----------
    coords : array_like of shape (4,)
        Coordinates :math:`(t, \\chi, \\theta, \\phi)`.
    a_func : callable
        The scale factor as a function of cosmic time, ``a_func(t) -> float``.
    k : {-1, 0, 1}, default=0.0
        Spatial curvature sign.

    Returns
    -------
    ndarray of shape (4, 4)
        The metric tensor :math:`g_{\\mu\\nu}`.
    """
    t, chi, theta, _phi = coords
    a = a_func(t)
    if k == 0:
        Sk = chi
    elif k > 0:
        Sk = np.sin(chi)
    else:
        Sk = np.sinh(chi)

    g = np.zeros((4, 4))
    g[0, 0] = -1.0
    g[1, 1] = a**2
    g[2, 2] = a**2 * Sk**2
    g[3, 3] = a**2 * Sk**2 * np.sin(theta) ** 2
    return g


def alcubierre_metric(coords, v_s=2.0, sigma=8.0, R=1.0):
    """The Alcubierre "warp drive" metric in Cartesian coordinates :math:`(t, x, y, z)`.

    .. math::

        ds^2 = -dt^2 + \\left[dx - v_s(t) f(r_s) \\, dt\\right]^2 + dy^2 + dz^2

    where :math:`r_s = \\sqrt{(x - x_s(t))^2 + y^2 + z^2}` is the distance
    from the center of the "warp bubble" (moving along :math:`x` at
    coordinate speed :math:`v_s`, with :math:`x_s(t) = v_s t`), and

    .. math::

        f(r_s) = \\frac{\\tanh(\\sigma(r_s + R)) - \\tanh(\\sigma(r_s - R))}
                       {2 \\tanh(\\sigma R)}

    is a smooth top-hat function equal to 1 well inside the bubble
    (:math:`r_s \\ll R`) and 0 far outside it. Spacetime is flat both inside
    and outside the thin bubble wall, but a ship riding inside can traverse
    coordinate distance faster than light as measured by a distant observer
    -- entirely by locally contracting spacetime ahead of it and expanding
    spacetime behind, without ever locally exceeding the speed of light.
    Known solutions like this one require exotic (negative-energy-density)
    matter to source them via Einstein's equations.

    Parameters
    ----------
    coords : array_like of shape (4,)
        Coordinates :math:`(t, x, y, z)`.
    v_s : float, default=2.0
        Coordinate warp speed, in units of :math:`c`.
    sigma : float, default=8.0
        Bubble wall steepness.
    R : float, default=1.0
        Bubble radius, in geometrized length units.

    Returns
    -------
    ndarray of shape (4, 4)
        The metric tensor :math:`g_{\\mu\\nu}`.
    """
    t, x, y, z = coords
    x_s = v_s * t
    r_s = np.sqrt((x - x_s) ** 2 + y**2 + z**2)
    f = (np.tanh(sigma * (r_s + R)) - np.tanh(sigma * (r_s - R))) / (2.0 * np.tanh(sigma * R))

    g = np.zeros((4, 4))
    g[0, 0] = v_s**2 * f**2 - 1.0
    g[0, 1] = g[1, 0] = -v_s * f
    g[1, 1] = 1.0
    g[2, 2] = 1.0
    g[3, 3] = 1.0
    return g
