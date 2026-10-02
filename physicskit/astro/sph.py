"""Smoothed-particle hydrodynamics (SPH) in one dimension.

SPH represents a gas by particles of fixed mass :math:`m` that move with the
flow (Lucy, Astron. J. 82, 1013 (1977); Gingold and Monaghan, MNRAS 181, 375
(1977)). Every field is a kernel-weighted sum over neighbours; the density
is

.. math::

    \\rho_i = \\sum_j m_j W(x_i - x_j, h_i),

and the Euler equations become ordinary differential equations for the
particles (Monaghan, Annu. Rev. Astron. Astrophys. 30, 543 (1992)),

.. math::

    \\frac{dv_i}{dt} = -\\sum_j m_j\\left(\\frac{P_i}{\\rho_i^2} + \\frac{P_j}{\\rho_j^2}
                       + \\Pi_{ij}\\right)\\partial_i W_{ij},
    \\qquad
    \\frac{du_i}{dt} = \\frac12\\sum_j m_j\\left(\\frac{P_i}{\\rho_i^2} + \\frac{P_j}{\\rho_j^2}
                       + \\Pi_{ij}\\right) v_{ij}\\,\\partial_i W_{ij},

with :math:`P = (\\gamma - 1)\\rho u`, the cubic-spline kernel, and Monaghan's
artificial viscosity :math:`\\Pi_{ij}` to capture shocks. Because the
particles follow the mass, resolution concentrates where the gas is dense,
which is why SPH became the standard method for star formation, galaxy
mergers and cosmological gas. :class:`SPH1D` evolves these equations, and
:meth:`SPH1D.sod_shock_tube` sets up Sod's test problem.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

__all__ = ["SPH1D", "cubic_spline_kernel"]


@njit(cache=True)
def _w(r, h):
    q = abs(r) / h
    sigma = 2.0 / (3.0 * h)
    if q < 1.0:
        return sigma * (1.0 - 1.5 * q * q + 0.75 * q**3)
    if q < 2.0:
        return sigma * 0.25 * (2.0 - q) ** 3
    return 0.0


@njit(cache=True)
def _dw(r, h):
    q = abs(r) / h
    sigma = 2.0 / (3.0 * h * h)
    s = 1.0 if r >= 0 else -1.0
    if q < 1.0:
        return sigma * s * (-3.0 * q + 2.25 * q * q)
    if q < 2.0:
        return -sigma * s * 0.75 * (2.0 - q) ** 2
    return 0.0


def cubic_spline_kernel(r: NDArray[np.float64], h: float) -> NDArray[np.float64]:
    """The 1D cubic-spline (M4) kernel :math:`W(r, h)`, normalized to :math:`\\int W\\,dr = 1`.

    .. math::

        W = \\frac{2}{3h}\\begin{cases} 1 - \\tfrac32 q^2 + \\tfrac34 q^3, & 0 \\le q < 1, \\\\
            \\tfrac14(2 - q)^3, & 1 \\le q < 2, \\\\ 0, & q \\ge 2, \\end{cases}
        \\qquad q = |r|/h.

    Parameters
    ----------
    r : ndarray of float
        Separations.
    h : float
        Smoothing length.

    Returns
    -------
    ndarray of float

    Examples
    --------
    >>> r = np.linspace(-3, 3, 60001)
    >>> round(float(np.trapezoid(cubic_spline_kernel(r, 1.0), r)), 6)
    1.0
    """
    r = np.asarray(r, dtype=np.float64)
    return np.array([_w(ri, h) for ri in r.ravel()]).reshape(r.shape)


@njit(cache=True)
def _density(x, m, h, order):
    n = x.size
    rho = np.zeros(n)
    xs = x[order]
    for a in range(n):
        i = order[a]
        rho[i] += m[i] * _w(0.0, h[i])
        b = a + 1
        while b < n and xs[b] - x[i] < 2.0 * h[i]:
            j = order[b]
            rho[i] += m[j] * _w(x[i] - x[j], h[i])
            b += 1
        b = a - 1
        while b >= 0 and x[i] - xs[b] < 2.0 * h[i]:
            j = order[b]
            rho[i] += m[j] * _w(x[i] - x[j], h[i])
            b -= 1
    return rho


@njit(cache=True)
def _forces(x, v, m, h, rho, u, gamma, alpha, beta, order):
    n = x.size
    acc = np.zeros(n)
    dudt = np.zeros(n)
    P = (gamma - 1.0) * rho * u
    c = np.sqrt(gamma * P / rho)
    xs = x[order]
    hmax = h.max()
    for a in range(n):
        i = order[a]
        b = a + 1
        while b < n and xs[b] - x[i] < 2.0 * hmax:
            j = order[b]
            xij = x[i] - x[j]
            hij = 0.5 * (h[i] + h[j])
            if abs(xij) < 2.0 * hij:
                vij = v[i] - v[j]
                dW = _dw(xij, hij)
                pi_ij = 0.0
                if vij * xij < 0.0:
                    mu = hij * vij * xij / (xij * xij + 0.01 * hij * hij)
                    pi_ij = (-alpha * 0.5 * (c[i] + c[j]) * mu + beta * mu * mu) / (0.5 * (rho[i] + rho[j]))
                term = P[i] / rho[i] ** 2 + P[j] / rho[j] ** 2 + pi_ij
                acc[i] -= m[j] * term * dW
                acc[j] += m[i] * term * dW
                dudt[i] += 0.5 * m[j] * term * vij * dW
                dudt[j] += 0.5 * m[i] * term * vij * dW
            b += 1
    return acc, dudt, c


class SPH1D:
    """One-dimensional SPH gas with the cubic-spline kernel and Monaghan artificial viscosity.

    Smoothing lengths follow the density, :math:`h_i = \\eta\\, m_i/\\rho_i`,
    and the equations are advanced with a kick-drift-kick leapfrog at a
    Courant-limited time step.

    Parameters
    ----------
    x, v : ndarray of float, shape (n,)
        Particle positions and velocities.
    m : ndarray of float or float
        Particle masses.
    u : ndarray of float, shape (n,)
        Specific internal energies.
    gamma : float, default 1.4
        Adiabatic index.
    eta : float, default 1.2
        Smoothing-length factor, in units of the local particle spacing.
    alpha, beta : float, default 1.0, 2.0
        Artificial-viscosity coefficients.
    fixed : ndarray of bool, optional
        Particles held at rest (boundaries).

    Examples
    --------
    >>> gas = SPH1D.sod_shock_tube(n_left=200)
    >>> gas.run(0.05)
    >>> bool(abs(gas.total_energy() - gas.energy_history[0]) < 1e-3 * gas.energy_history[0])
    True
    """

    def __init__(self, x, v, m, u, gamma=1.4, eta=1.2, alpha=1.0, beta=2.0, fixed=None):
        self.x = np.asarray(x, dtype=np.float64).copy()
        self.v = np.asarray(v, dtype=np.float64).copy()
        self.m = np.broadcast_to(np.asarray(m, dtype=np.float64), self.x.shape).copy()
        self.u = np.asarray(u, dtype=np.float64).copy()
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.fixed = np.zeros(self.x.size, dtype=bool) if fixed is None else np.asarray(fixed, dtype=bool)
        self.t = 0.0
        spacing = np.gradient(np.sort(self.x))[np.argsort(np.argsort(self.x))]
        self.h = self.eta * np.abs(spacing)
        self._update_density()
        self.energy_history = [self.total_energy()]

    @classmethod
    def sod_shock_tube(cls, n_left=400, gamma=1.4, left=(1.0, 1.0), right=(0.125, 0.1), length=1.0, **kwargs):
        """Equal-mass particles for Sod's shock tube, with the diaphragm at :math:`x = 0`.

        The left half :math:`-L < x < 0` has density and pressure ``left``,
        the right half ``right``; both are at rest. Particle spacing scales
        inversely with density, and the outermost few particles on each
        side are held fixed as walls.

        Parameters
        ----------
        n_left : int, default 400
            Particles in the left half.
        gamma : float, default 1.4
        left, right : tuple of float
            ``(rho, P)`` of each side.
        length : float, default 1.0
            Half-length :math:`L` of the tube.
        **kwargs
            Passed to the constructor.

        Returns
        -------
        SPH1D
        """
        (rL, pL), (rR, pR) = left, right
        dxL = length / n_left
        m = rL * dxL
        dxR = m / rR
        xL = -length + dxL * (np.arange(n_left) + 0.5)
        xR = dxR * (np.arange(int(round(length / dxR))) + 0.5)
        x = np.concatenate((xL, xR))
        u = np.concatenate((np.full(xL.size, pL / ((gamma - 1) * rL)), np.full(xR.size, pR / ((gamma - 1) * rR))))
        fixed = np.zeros(x.size, dtype=bool)
        fixed[:4] = fixed[-4:] = True
        return cls(x, np.zeros_like(x), m, u, gamma=gamma, fixed=fixed, **kwargs)

    def _update_density(self):
        order = np.argsort(self.x)
        for _ in range(2):
            self.rho = _density(self.x, self.m, self.h, order)
            self.h = self.eta * self.m / self.rho
        self.rho = _density(self.x, self.m, self.h, order)
        return order

    @property
    def pressure(self) -> NDArray[np.float64]:
        """Pressure :math:`P = (\\gamma - 1)\\rho u`."""
        return (self.gamma - 1.0) * self.rho * self.u

    def total_energy(self) -> float:
        """Total kinetic plus internal energy :math:`\\sum_i m_i(v_i^2/2 + u_i)`."""
        return float(np.sum(self.m * (0.5 * self.v**2 + self.u)))

    def _derivatives(self):
        order = self._update_density()
        acc, dudt, c = _forces(self.x, self.v, self.m, self.h, self.rho, self.u, self.gamma, self.alpha, self.beta, order)
        acc[self.fixed] = 0.0
        dudt[self.fixed] = 0.0
        return acc, dudt, c

    def run(self, t_end: float, cfl: float = 0.3) -> None:
        """Advance to ``self.t + t_end`` with kick-drift-kick leapfrog.

        Parameters
        ----------
        t_end : float
            Time to advance by.
        cfl : float, default 0.3
            Courant factor in :math:`\\Delta t = C\\min_i h_i/(c_i + |v_i|)`.
        """
        t_stop = self.t + t_end
        acc, dudt, c = self._derivatives()
        while self.t < t_stop - 1e-14:
            dt = min(cfl * np.min(self.h / (c + np.abs(self.v) + 1e-12)), t_stop - self.t)
            v_half = self.v + 0.5 * dt * acc
            u_half = self.u + 0.5 * dt * dudt
            self.x = self.x + dt * np.where(self.fixed, 0.0, v_half)
            self.v, self.u = v_half, u_half
            acc, dudt, c = self._derivatives()
            self.v = self.v + 0.5 * dt * acc
            self.u = self.u + 0.5 * dt * dudt
            self.t += dt
        self.energy_history.append(self.total_energy())
