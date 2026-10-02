"""The Kerr-Newman black hole: geodesics and the shadow of a charged, rotating hole.

Newman and collaborators (1965) found the charged generalization of Kerr's
solution, the most general stationary black hole of Einstein-Maxwell theory.
In Boyer-Lindquist coordinates it is the Kerr metric with

.. math::

    \\Delta = r^2 - 2Mr + a^2 + Q^2,

so its horizons sit at :math:`r_\\pm = M \\pm \\sqrt{M^2 - a^2 - Q^2}`, and it
contains Schwarzschild (:math:`a = Q = 0`), Reissner-Nordström (:math:`a =
0`) and Kerr (:math:`Q = 0`) as special cases. Carter (1968) showed that the
motion of a test particle of mass :math:`\\mu` and charge :math:`e` still
separates. In Mino time :math:`d\\lambda = d\\tau/\\Sigma`, with
:math:`\\Sigma = r^2 + a^2\\cos^2\\theta`,

.. math::

    \\left(\\frac{dr}{d\\lambda}\\right)^2 = R(r) = P(r)^2 - \\Delta\\left[\\mu^2r^2 + (L - aE)^2 + \\mathcal{C}\\right],
    \\qquad P = E(r^2 + a^2) - aL - eQr,

    \\left(\\frac{d\\theta}{d\\lambda}\\right)^2 = \\Theta(\\theta) = \\mathcal{C}
    - \\cos^2\\theta\\left[a^2(\\mu^2 - E^2) + \\frac{L^2}{\\sin^2\\theta}\\right],

where :math:`\\mathcal C` is Carter's constant. Photons (:math:`\\mu = e = 0`)
on spherical orbits satisfy :math:`R = R' = 0`; their critical impact
parameters trace the edge of the shadow seen by a distant observer (Bardeen
1973; de Vries, Class. Quantum Grav. 17, 123 (2000)).
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.optimize import brentq

__all__ = ["KerrNewmanBlackHole", "trace_kerr_newman_geodesic"]

OUTCOME_ESCAPED = 0
OUTCOME_CAPTURED = 1
OUTCOME_UNRESOLVED = 2


@njit(cache=True)
def _R_and_dR(r, E, L, C, a, M, Qc, mu2, e):
    Delta = r * r - 2.0 * M * r + a * a + Qc * Qc
    dDelta = 2.0 * r - 2.0 * M
    P = E * (r * r + a * a) - a * L - e * Qc * r
    dP = 2.0 * E * r - e * Qc
    K = mu2 * r * r + (L - a * E) ** 2 + C
    R = P * P - Delta * K
    dR = 2.0 * P * dP - dDelta * K - Delta * 2.0 * mu2 * r
    return R, dR, P, Delta


@njit(cache=True)
def _Theta_and_dTheta(theta, E, L, C, a, mu2):
    c = np.cos(theta)
    s = np.sin(theta)
    s2 = s * s
    Theta = C - c * c * (a * a * (mu2 - E * E) + L * L / s2)
    # d/dtheta of -cos^2 a^2(mu^2 - E^2) - L^2 cot^2
    dTheta = 2.0 * c * s * a * a * (mu2 - E * E) + 2.0 * L * L * c / (s2 * s)
    return Theta, dTheta


@njit(cache=True)
def _rhs(y, E, L, C, a, M, Qc, mu2, e):
    r, pr, theta, ptheta = y[0], y[1], y[2], y[3]
    R, dR, P, Delta = _R_and_dR(r, E, L, C, a, M, Qc, mu2, e)
    Theta, dTheta = _Theta_and_dTheta(theta, E, L, C, a, mu2)
    s2 = np.sin(theta) ** 2
    out = np.empty(6)
    out[0] = pr
    out[1] = 0.5 * dR
    out[2] = ptheta
    out[3] = 0.5 * dTheta
    out[4] = -(a * E * s2 - L) / s2 * 1.0 + a * P / Delta  # dphi/dlambda
    out[5] = -a * (a * E * s2 - L) + (r * r + a * a) * P / Delta  # dt/dlambda
    return out


@njit(cache=True)
def trace_kerr_newman_geodesic(y0, E, L, C, a, M, Qc, mu2, e, n_steps, step, r_stop, r_max, record_every):
    """Integrate a Kerr-Newman geodesic in Mino time with RK4.

    The state is ``(r, dr/dlambda, theta, dtheta/dlambda, phi, t)``. The
    second-order forms :math:`r'' = R'/2`, :math:`\\theta'' = \\Theta'/2` pass
    through turning points smoothly; after every step the momenta are
    projected back onto :math:`r'^2 = R`, :math:`\\theta'^2 = \\Theta`, and the
    Mino-time step is scaled so that :math:`r`, :math:`\\theta` and
    :math:`\\phi` change by at most a fraction ``step`` (of :math:`r`, or in
    radians) per step.

    Parameters
    ----------
    y0 : ndarray of float, shape (6,)
        Initial state.
    E, L, C : float
        Energy, axial angular momentum and Carter constant.
    a, M, Qc : float
        Spin, mass and charge of the hole.
    mu2 : float
        Squared rest mass of the particle (0 for light).
    e : float
        Particle charge.
    n_steps : int
        Maximum number of steps.
    step : float
        Fractional step size.
    r_stop, r_max : float
        Stop when :math:`r < r_\\text{stop}` (captured) or :math:`r > r_\\text{max}` (escaped).
    record_every : int
        Store the state every this many steps.

    Returns
    -------
    outcome : int
        0 escaped, 1 captured, 2 still bound after ``n_steps``.
    path : ndarray of float, shape (k, 7)
        Recorded ``(lambda, r, dr, theta, dtheta, phi, t)``.
    """
    n_rec = n_steps // record_every + 2
    path = np.empty((n_rec, 7))
    y = y0.copy()
    lam = 0.0
    k = 0
    path[0, 0] = lam
    path[0, 1:] = y
    outcome = OUTCOME_UNRESOLVED
    for i in range(1, n_steps + 1):
        r = y[0]
        if r <= r_stop:
            outcome = OUTCOME_CAPTURED
            break
        if r >= r_max:
            outcome = OUTCOME_ESCAPED
            break
        # fractional rate of change of r, theta and phi; at a radial turning
        # point dr/dlambda = 0 and the radial acceleration R'/2 sets the scale
        k1 = _rhs(y, E, L, C, a, M, Qc, mu2, e)
        rr = max(r, 1.0)
        speed = max(abs(y[1]) / rr, abs(y[3]), abs(k1[4]), np.sqrt(abs(k1[1]) / rr), 1e-6)
        dlam = step / speed
        k2 = _rhs(y + 0.5 * dlam * k1, E, L, C, a, M, Qc, mu2, e)
        k3 = _rhs(y + 0.5 * dlam * k2, E, L, C, a, M, Qc, mu2, e)
        k4 = _rhs(y + dlam * k3, E, L, C, a, M, Qc, mu2, e)
        y = y + dlam / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        R = _R_and_dR(y[0], E, L, C, a, M, Qc, mu2, e)[0]
        Th = _Theta_and_dTheta(y[2], E, L, C, a, mu2)[0]
        y[1] = (1.0 if y[1] >= 0.0 else -1.0) * np.sqrt(max(R, 0.0))
        y[3] = (1.0 if y[3] >= 0.0 else -1.0) * np.sqrt(max(Th, 0.0))
        lam += dlam
        if i % record_every == 0 and k + 1 < n_rec:
            k += 1
            path[k, 0] = lam
            path[k, 1:] = y
    if k + 1 < n_rec:
        k += 1
        path[k, 0] = lam
        path[k, 1:] = y
    return outcome, path[: k + 1]


@njit(cache=True)
def _shadow_image(alphas, betas, r_o, theta_o, a, M, Qc, r_stop, n_steps):
    out = np.empty((betas.size, alphas.size), dtype=np.int64)
    s_o, c_o = np.sin(theta_o), np.cos(theta_o)
    for i in range(betas.size):
        for j in range(alphas.size):
            al, be = alphas[j], betas[i]
            L = -al * s_o
            C = be * be + (al * al - a * a) * c_o * c_o
            R = _R_and_dR(r_o, 1.0, L, C, a, M, Qc, 0.0, 0.0)[0]
            y0 = np.array([r_o, -np.sqrt(max(R, 0.0)), theta_o, be, 0.0, 0.0])
            res, _ = trace_kerr_newman_geodesic(y0, 1.0, L, C, a, M, Qc, 0.0, 0.0, n_steps, 0.01, r_stop, 2.0 * r_o, n_steps)
            out[i, j] = res
    return out


class KerrNewmanBlackHole:
    """A black hole of mass ``M``, spin ``a`` and charge ``Q``, in geometrized units.

    Parameters
    ----------
    M : float, default 1.0
    a : float, default 0.0
        Spin parameter :math:`J/M`.
    Q : float, default 0.0
        Electric charge (Gaussian geometrized units).

    Raises
    ------
    ValueError
        If :math:`a^2 + Q^2 \\ge M^2`, a naked singularity.

    Examples
    --------
    >>> bh = KerrNewmanBlackHole(M=1.0, a=0.0, Q=0.0)
    >>> round(float(bh.shadow_radius_nonrotating()), 6)  # Schwarzschild: sqrt(27) M
    5.196152
    >>> round(KerrNewmanBlackHole(a=0.6, Q=0.8 - 1e-9).outer_horizon_radius, 4)  # extremal: r+ = M
    1.0
    """

    def __init__(self, M: float = 1.0, a: float = 0.0, Q: float = 0.0):
        if a < 0 or M <= 0:
            raise ValueError("need M > 0 and a >= 0")
        if a * a + Q * Q >= M * M:
            raise ValueError("a^2 + Q^2 must be below M^2 for a black hole")
        self.M = float(M)
        self.a = float(a)
        self.Q = float(Q)

    def delta(self, r):
        """:math:`\\Delta(r) = r^2 - 2Mr + a^2 + Q^2`."""
        r = np.asarray(r, dtype=float)
        return r * r - 2 * self.M * r + self.a**2 + self.Q**2

    @property
    def outer_horizon_radius(self) -> float:
        """:math:`r_+ = M + \\sqrt{M^2 - a^2 - Q^2}`."""
        return float(self.M + np.sqrt(self.M**2 - self.a**2 - self.Q**2))

    @property
    def inner_horizon_radius(self) -> float:
        """:math:`r_- = M - \\sqrt{M^2 - a^2 - Q^2}`."""
        return float(self.M - np.sqrt(self.M**2 - self.a**2 - self.Q**2))

    @property
    def horizon_angular_velocity(self) -> float:
        """:math:`\\Omega_H = a/(r_+^2 + a^2)`."""
        return self.a / (self.outer_horizon_radius**2 + self.a**2)

    @property
    def electric_potential(self) -> float:
        """Horizon electric potential :math:`\\Phi_H = Q r_+/(r_+^2 + a^2)`."""
        rp = self.outer_horizon_radius
        return self.Q * rp / (rp**2 + self.a**2)

    def photon_orbit_constants(self, r):
        """Critical impact parameters of the spherical photon orbit at radius ``r``.

        Solving :math:`R = R' = 0` with :math:`E = 1` gives :math:`A = r^2 + a^2 -
        a\\xi = 4r\\Delta/\\Delta'`, hence

        .. math::

            \\xi = \\frac{L}{E} = \\frac{1}{a}\\left(r^2 + a^2 - \\frac{4r\\Delta}{\\Delta'}\\right),
            \\qquad
            \\eta = \\frac{\\mathcal C}{E^2} = \\frac{16r^2\\Delta}{\\Delta'^2} - (\\xi - a)^2.

        Parameters
        ----------
        r : array_like
            Orbit radius; requires ``a > 0``.

        Returns
        -------
        xi, eta : ndarray
        """
        if self.a == 0:
            raise ValueError("photon_orbit_constants needs a > 0; use photon_sphere_radius_nonrotating")
        r = np.asarray(r, dtype=float)
        D = self.delta(r)
        dD = 2 * r - 2 * self.M
        xi = (r * r + self.a**2 - 4 * r * D / dD) / self.a
        eta = 16 * r * r * D / dD**2 - (xi - self.a) ** 2
        return xi, eta

    def photon_sphere_radius_nonrotating(self) -> float:
        """Photon sphere of the non-rotating (Reissner-Nordström) case, :math:`r = \\tfrac12(3M + \\sqrt{9M^2 - 8Q^2})`."""
        return float(0.5 * (3 * self.M + np.sqrt(9 * self.M**2 - 8 * self.Q**2)))

    def shadow_radius_nonrotating(self) -> float:
        """Shadow radius :math:`b_c = r_p^2/\\sqrt{\\Delta(r_p)}` for ``a = 0`` (any charge)."""
        rp = self.photon_sphere_radius_nonrotating()
        return float(rp * rp / np.sqrt(rp * rp - 2 * self.M * rp + self.Q**2))

    def equatorial_photon_orbits(self) -> tuple[float, float]:
        """Radii of the prograde and retrograde circular photon orbits.

        They solve :math:`r^2 - 3Mr + 2Q^2 \\pm 2a\\sqrt{Mr - Q^2} = 0` (upper sign
        prograde), the :math:`\\eta = 0` limit of :meth:`photon_orbit_constants`;
        for :math:`Q = 0` this is Bardeen, Press and Teukolsky's (1972)
        :math:`r = 2M\\{1 + \\cos[\\tfrac23\\arccos(\\mp a/M)]\\}`.

        Returns
        -------
        r_prograde, r_retrograde : float
        """
        M, a, Q = self.M, self.a, self.Q
        lo = max(self.outer_horizon_radius, Q * Q / M) * (1 + 1e-12)
        hi = 4.0 * M + 1.0
        radii = []
        for sign in (1.0, -1.0):

            def f(r, sign=sign):
                return r * r - 3 * M * r + 2 * Q * Q + sign * 2 * a * np.sqrt(max(M * r - Q * Q, 0.0))

            grid = np.linspace(lo, hi, 2001)
            vals = np.array([f(r) for r in grid])
            i = np.nonzero(np.diff(np.sign(vals)))[0][-1]
            radii.append(float(brentq(f, grid[i], grid[i + 1], xtol=1e-14)))
        return radii[0], radii[1]

    def shadow_boundary(self, inclination: float = np.pi / 2, n: int = 400):
        """Edge of the shadow on a distant observer's sky, in Bardeen's celestial coordinates.

        .. math::

            \\alpha = -\\frac{\\xi}{\\sin\\theta_o}, \\qquad
            \\beta = \\pm\\sqrt{\\eta + a^2\\cos^2\\theta_o - \\xi^2\\cot^2\\theta_o}.

        Parameters
        ----------
        inclination : float, default ``pi/2``
            Observer inclination :math:`\\theta_o` from the spin axis.
        n : int, default 400
            Points per half of the curve.

        Returns
        -------
        alpha, beta : ndarray
            A closed curve, in units of :math:`M`.
        """
        th = inclination
        if self.a < 1e-6 * self.M:
            phi = np.linspace(0, 2 * np.pi, 2 * n)
            b = self.shadow_radius_nonrotating()
            return b * np.cos(phi), b * np.sin(phi)
        r1, r2 = self.equatorial_photon_orbits()
        r = np.linspace(r1, r2, 20 * n)
        xi, eta = self.photon_orbit_constants(r)
        b2 = eta + self.a**2 * np.cos(th) ** 2 - xi**2 / np.tan(th) ** 2
        ok = b2 >= 0
        r = r[ok]
        # refine on the visible range
        r = np.linspace(r.min(), r.max(), n)
        xi, eta = self.photon_orbit_constants(r)
        beta = np.sqrt(np.clip(eta + self.a**2 * np.cos(th) ** 2 - xi**2 / np.tan(th) ** 2, 0, None))
        alpha = -xi / np.sin(th)
        return np.concatenate((alpha, alpha[::-1])), np.concatenate((beta, -beta[::-1]))

    def shadow_area(self, inclination: float = np.pi / 2) -> float:
        """Area enclosed by :meth:`shadow_boundary` (shoelace formula)."""
        x, y = self.shadow_boundary(inclination, n=4000)
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def geodesic(self, r0, theta0, E, L, C, mu2=0.0, e=0.0, outward=False, poleward=False, n_steps=20000, step=0.005, record_every=10, r_max=1e4):
        """Integrate a geodesic (or charged orbit) from ``(r0, theta0)``.

        Parameters
        ----------
        r0, theta0 : float
            Starting point.
        E, L, C : float
            Energy, angular momentum and Carter constant.
        mu2 : float, default 0.0
            Squared rest mass (0 for light, 1 for a massive particle).
        e : float, default 0.0
            Particle charge (per unit mass when ``mu2 = 1``).
        outward, poleward : bool, default False
            Initial signs of :math:`dr/d\\lambda` and of :math:`d\\theta/d\\lambda` (toward :math:`\\theta = 0`).
        n_steps, step, record_every, r_max
            Integration controls, see :func:`trace_kerr_newman_geodesic`.

        Returns
        -------
        dict
            ``"outcome"`` and arrays ``"lambda"``, ``"r"``, ``"theta"``,
            ``"phi"``, ``"t"`` and the Cartesian ``"x"``, ``"y"``, ``"z"``.
        """
        R = _R_and_dR(r0, E, L, C, self.a, self.M, self.Q, mu2, e)[0]
        Th = _Theta_and_dTheta(theta0, E, L, C, self.a, mu2)[0]
        if R < -1e-10 or Th < -1e-10:
            raise ValueError("initial point is in a forbidden region (R < 0 or Theta < 0)")
        y0 = np.array([r0, (1 if outward else -1) * np.sqrt(max(R, 0)), theta0, (-1 if poleward else 1) * np.sqrt(max(Th, 0)), 0.0, 0.0])
        outcome, path = trace_kerr_newman_geodesic(
            y0, E, L, C, self.a, self.M, self.Q, mu2, e, n_steps, step, 1.001 * self.outer_horizon_radius, r_max, record_every
        )
        lam, r, theta, phi, t = path[:, 0], path[:, 1], path[:, 3], path[:, 5], path[:, 6]
        # Boyer-Lindquist to Kerr-Schild-like Cartesian coordinates for plotting
        rho = np.sqrt(r * r + self.a**2)
        return {
            "outcome": int(outcome),
            "lambda": lam,
            "r": r,
            "theta": theta,
            "phi": phi,
            "t": t,
            "x": rho * np.sin(theta) * np.cos(phi),
            "y": rho * np.sin(theta) * np.sin(phi),
            "z": r * np.cos(theta),
        }

    def ray_traced_shadow(self, alphas, betas, inclination=np.pi / 2, r_observer=500.0, n_steps=20000):
        """Backward-trace photons from a distant observer and mark the captured ones.

        Parameters
        ----------
        alphas, betas : ndarray of float
            Celestial coordinates of the image pixels.
        inclination : float, default ``pi/2``
        r_observer : float, default 500.0
        n_steps : int, default 20000
            Step limit per ray; rays still bound at the end (near the photon
            ring) are reported as 2.

        Returns
        -------
        ndarray of int, shape (len(betas), len(alphas))
            0 escaped, 1 captured, 2 unresolved.
        """
        return _shadow_image(
            np.asarray(alphas, dtype=float),
            np.asarray(betas, dtype=float),
            r_observer,
            inclination,
            self.a,
            self.M,
            self.Q,
            1.01 * self.outer_horizon_radius,
            n_steps,
        )
