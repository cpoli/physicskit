"""Numba-accelerated geodesic integrators for Schwarzschild and Kerr spacetimes.

A geodesic satisfies :math:`\\frac{d^2 x^\\mu}{d\\lambda^2} + \\Gamma^\\mu_{\\alpha\\beta}
\\frac{dx^\\alpha}{d\\lambda}\\frac{dx^\\beta}{d\\lambda} = 0`, where :math:`\\lambda` is
proper time :math:`\\tau` for a timelike (massive-particle) geodesic or an
affine parameter for a null (photon) geodesic. This module hand-codes the
analytic Christoffel symbols for two spacetimes -- rather than the generic
finite-difference engine in :mod:`physicskit.relativity.core.tensors` -- so that repeated
RK4 evaluation inside a JIT-compiled loop is fast enough for interactive use
and for tracing thousands of rays in :mod:`physicskit.relativity.core.raytracer`.

- :func:`integrate_schwarzschild_geodesic` integrates the full 4D
  coordinate geodesic equation with the exact Schwarzschild Christoffel
  symbols, supporting arbitrary (non-equatorial) timelike or null geodesics.
- :func:`integrate_kerr_equatorial_geodesic` integrates equatorial
  (:math:`\\theta = \\pi/2`) Kerr geodesics using the reduced
  Boyer-Lindquist equations of motion in terms of the conserved energy
  :math:`E` and angular momentum :math:`L` (Bardeen, Press & Teukolsky
  1972), which is both simpler and more numerically robust than the full
  Kerr Christoffel symbols for the equatorial-plane physics (ISCO, frame
  dragging, the Penrose process) this package focuses on.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = [
    "integrate_kerr_equatorial_geodesic",
    "integrate_schwarzschild_geodesic",
]


@njit(cache=True)
def _schwarzschild_rhs(y, M):
    """RHS of the first-order system equivalent to the Schwarzschild geodesic equation."""
    _t, r, th, _ph, ut, ur, uth, uphi = y

    two_m = 2.0 * M
    Gt_tr = M / (r * (r - two_m))
    Gr_tt = M * (r - two_m) / r**3
    Gr_rr = -M / (r * (r - two_m))
    Gr_thth = -(r - two_m)
    Gr_phph = -(r - two_m) * np.sin(th) ** 2
    Gth_rth = 1.0 / r
    Gth_phph = -np.sin(th) * np.cos(th)
    Gph_rph = 1.0 / r
    Gph_thph = np.cos(th) / np.sin(th)

    dut = -2.0 * Gt_tr * ut * ur
    dur = -(Gr_tt * ut * ut + Gr_rr * ur * ur + Gr_thth * uth * uth + Gr_phph * uphi * uphi)
    duth = -(2.0 * Gth_rth * ur * uth + Gth_phph * uphi * uphi)
    duphi = -(2.0 * Gph_rph * ur * uphi + 2.0 * Gph_thph * uth * uphi)

    out = np.empty(8)
    out[0] = ut
    out[1] = ur
    out[2] = uth
    out[3] = uphi
    out[4] = dut
    out[5] = dur
    out[6] = duth
    out[7] = duphi
    return out


@njit(cache=True)
def integrate_schwarzschild_geodesic(y0, M, dlambda, n_steps, r_horizon_factor=1.001, r_max=1.0e6):
    """Integrate a Schwarzschild geodesic with fixed-step RK4.

    Works for both timelike and null geodesics: the geodesic equation
    itself does not depend on the normalization of the initial 4-velocity,
    only the initial condition does (``y0[4:8]`` should satisfy
    :math:`g_{\\mu\\nu} u^\\mu u^\\nu = -1` for a timelike geodesic or
    :math:`=0` for a null one).

    Parameters
    ----------
    y0 : ndarray of shape (8,)
        Initial state :math:`(t, r, \\theta, \\phi, u^t, u^r, u^\\theta, u^\\phi)`.
    M : float
        Black hole mass, in geometrized units.
    dlambda : float
        Affine-parameter step size.
    n_steps : int
        Maximum number of steps to take.
    r_horizon_factor : float, default=1.001
        Integration stops once :math:`r` falls below
        ``r_horizon_factor * 2M`` (the particle has fallen in).
    r_max : float, default=1e6
        Integration stops once :math:`r` exceeds this value (the particle
        has escaped).

    Returns
    -------
    trajectory : ndarray of shape (n_valid + 1, 8)
        The integrated state at every step actually taken.
    n_valid : int
        Number of successful steps (so ``trajectory[: n_valid + 1]`` is the
        full valid trajectory; the array is already trimmed to this size).
    """
    y = y0.copy()
    trajectory = np.empty((n_steps + 1, 8))
    trajectory[0] = y
    r_stop = r_horizon_factor * 2.0 * M
    n_valid = 0

    for i in range(n_steps):
        r = y[1]
        if r <= r_stop or r >= r_max:
            break
        k1 = _schwarzschild_rhs(y, M)
        k2 = _schwarzschild_rhs(y + 0.5 * dlambda * k1, M)
        k3 = _schwarzschild_rhs(y + 0.5 * dlambda * k2, M)
        k4 = _schwarzschild_rhs(y + dlambda * k3, M)
        y = y + (dlambda / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[i + 1] = y
        n_valid = i + 1

    return trajectory[: n_valid + 1], n_valid


@njit(cache=True)
def _kerr_eq_radial_potential(r, E, L, a, M, mu2):
    """Carter's radial potential R(r)/Sigma^2 = (dr/dtau)^2 for equatorial Kerr orbits."""
    Delta = r * r - 2.0 * M * r + a * a
    A = E * (r * r + a * a) - L * a
    R = A * A - Delta * (mu2 * r * r + (L - a * E) ** 2)
    return R / r**4


@njit(cache=True)
def _kerr_eq_rhs(y, E, L, a, M, mu2, h):
    """RHS of the reduced equatorial Kerr equations of motion, state (t, r, phi, dr/dtau)."""
    _t, r, _phi, ur = y
    Delta = r * r - 2.0 * M * r + a * a
    A = E * (r * r + a * a) - L * a

    dt = (-a * (a * E - L) + (r * r + a * a) * A / Delta) / (r * r)
    dphi = (-(a * E - L) + a * A / Delta) / (r * r)

    F_plus = _kerr_eq_radial_potential(r + h, E, L, a, M, mu2)
    F_minus = _kerr_eq_radial_potential(r - h, E, L, a, M, mu2)
    d2r = 0.5 * (F_plus - F_minus) / (2.0 * h)

    out = np.empty(4)
    out[0] = dt
    out[1] = ur
    out[2] = dphi
    out[3] = d2r
    return out


@njit(cache=True)
def integrate_kerr_equatorial_geodesic(y0, E, L, a, M, mu2, dlambda, n_steps, r_horizon_factor=1.001, r_max=1.0e6):
    """Integrate an equatorial-plane Kerr geodesic with fixed-step RK4.

    Uses the reduced Boyer-Lindquist equations of motion for a particle of
    conserved specific energy :math:`E` and angular momentum :math:`L`
    (Carter's constant :math:`Q = 0` for equatorial orbits), rather than the
    full 4D Kerr Christoffel symbols.

    Parameters
    ----------
    y0 : ndarray of shape (4,)
        Initial state :math:`(t, r, \\phi, dr/d\\tau)`.
    E : float
        Conserved specific energy :math:`E = -u_t`.
    L : float
        Conserved specific angular momentum :math:`L = u_\\phi`.
    a : float
        Kerr spin parameter, :math:`0 \\le a < M`.
    M : float
        Black hole mass, in geometrized units.
    mu2 : float
        ``1.0`` for a timelike geodesic, ``0.0`` for a null (photon) geodesic.
    dlambda : float
        Affine-parameter (or proper-time) step size.
    n_steps : int
        Maximum number of steps to take.
    r_horizon_factor : float, default=1.001
        Integration stops once :math:`r` falls below
        ``r_horizon_factor * r_+``, where :math:`r_+ = M + \\sqrt{M^2 - a^2}`.
    r_max : float, default=1e6
        Integration stops once :math:`r` exceeds this value.

    Returns
    -------
    trajectory : ndarray of shape (n_valid + 1, 4)
        The integrated state at every step actually taken.
    n_valid : int
        Number of successful steps.
    """
    r_plus = M + np.sqrt(M * M - a * a)
    r_stop = r_horizon_factor * r_plus
    h = 1.0e-6 * max(1.0, y0[1])

    y = y0.copy()
    trajectory = np.empty((n_steps + 1, 4))
    trajectory[0] = y
    n_valid = 0

    for i in range(n_steps):
        r = y[1]
        if r <= r_stop or r >= r_max:
            break
        k1 = _kerr_eq_rhs(y, E, L, a, M, mu2, h)
        k2 = _kerr_eq_rhs(y + 0.5 * dlambda * k1, E, L, a, M, mu2, h)
        k3 = _kerr_eq_rhs(y + 0.5 * dlambda * k2, E, L, a, M, mu2, h)
        k4 = _kerr_eq_rhs(y + dlambda * k3, E, L, a, M, mu2, h)
        y = y + (dlambda / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[i + 1] = y
        n_valid = i + 1

    return trajectory[: n_valid + 1], n_valid
