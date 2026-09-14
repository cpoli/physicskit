"""A Numba-accelerated backward photon ray-tracer for Kerr gravitational lensing.

Unlike Schwarzschild, a spinning black hole's null geodesics are not
confined to a single plane, so :mod:`physicskit.relativity.core.raytracer`'s 2D orbit
equation no longer applies. This module integrates the full Carter (1968)
separated equations of motion in Mino time :math:`\\lambda` (where
:math:`d\\lambda = d\\tau / \\Sigma`), which decouple the radial and polar
motion into two independent effective-potential problems:

.. math::

    \\left(\\frac{dr}{d\\lambda}\\right)^2 = R(r), \\qquad
    \\left(\\frac{d\\theta}{d\\lambda}\\right)^2 = \\Theta(\\theta)

Differentiating once more removes the sign ambiguity at radial and polar
turning points entirely (the same trick used for equatorial Kerr orbits in
:mod:`physicskit.relativity.core.geodesics`, extended here to both :math:`r` and
:math:`\\theta`):

.. math::

    \\frac{d^2 r}{d\\lambda^2} = \\frac{1}{2}\\frac{dR}{dr}, \\qquad
    \\frac{d^2 \\theta}{d\\lambda^2} = \\frac{1}{2}\\frac{d\\Theta}{d\\theta}

Camera rays are parametrized by Bardeen's (1973) impact parameters
:math:`(\\alpha, \\beta)`, which fix the photon's conserved energy
(:math:`E=1`), angular momentum :math:`L = -\\alpha \\sin\\theta_o`, and
Carter constant :math:`Q = (\\alpha^2 - a^2)\\cos^2\\theta_o + \\beta^2`
given the observer's inclination :math:`\\theta_o`.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["render_kerr_shadow_image"]

OUTCOME_ESCAPED = 0
OUTCOME_CAPTURED = 1
OUTCOME_UNRESOLVED = 2
OUTCOME_DISK = 3


@njit(cache=True)
def _kerr_R(r, E, L, Q, a, M):
    Delta = r * r - 2.0 * M * r + a * a
    A = E * (r * r + a * a) - L * a
    return A * A - Delta * (Q + (L - a * E) ** 2)


@njit(cache=True)
def _kerr_Theta(theta, E, L, Q, a):
    cos2 = np.cos(theta) ** 2
    sin2 = np.sin(theta) ** 2
    return Q + cos2 * (a * a * E * E - L * L / sin2)


@njit(cache=True)
def _kerr_full_rhs(y, E, L, Q, a, M, h):
    """RHS of the 6D first-order system: state (r, dr/dlambda, theta, dtheta/dlambda, t, phi)."""
    r, pr, theta, ptheta, _t, _phi = y
    Delta = r * r - 2.0 * M * r + a * a
    A = E * (r * r + a * a) - L * a
    sin2 = np.sin(theta) ** 2

    dt = -a * a * E * sin2 + a * L + (r * r + a * a) * A / Delta
    dphi = -a * E + L / sin2 + a * A / Delta

    R_plus = _kerr_R(r + h, E, L, Q, a, M)
    R_minus = _kerr_R(r - h, E, L, Q, a, M)
    d2r = 0.5 * (R_plus - R_minus) / (2.0 * h)

    Theta_plus = _kerr_Theta(theta + h, E, L, Q, a)
    Theta_minus = _kerr_Theta(theta - h, E, L, Q, a)
    d2theta = 0.5 * (Theta_plus - Theta_minus) / (2.0 * h)

    out = np.empty(6)
    out[0] = pr
    out[1] = d2r
    out[2] = ptheta
    out[3] = d2theta
    out[4] = dt
    out[5] = dphi
    return out


@njit(cache=True)
def _trace_pixel_kerr(
    alpha,
    beta,
    r_o,
    theta_o,
    a,
    M,
    r_disk_inner,
    r_disk_outer,
    dlambda_max,
    n_steps,
    r_horizon_factor,
):
    """Trace one Kerr camera ray; returns (outcome_code, hit_radius).

    Uses an adaptive Mino-time step: in Mino time, ``dr/dlambda`` grows like
    ``r**2`` at large ``r`` (since Mino time absorbs a factor of
    :math:`\\Sigma \\sim r^2`), so a fixed step size that resolves the
    near-horizon dynamics would take absurdly large jumps far from the
    black hole. Scaling the step to the local radial "speed" keeps every
    step's fractional change in ``r`` bounded, automatically compressing
    the (physically boring, nearly straight-line) far-field approach into
    few steps while still resolving the near-horizon region finely.
    """
    E = 1.0
    L = -alpha * np.sin(theta_o)
    Q = (alpha * alpha - a * a) * np.cos(theta_o) ** 2 + beta * beta

    R0 = _kerr_R(r_o, E, L, Q, a, M)
    pr0 = -np.sqrt(max(R0, 0.0))
    ptheta0 = beta  # Theta(theta_o) = beta^2 exactly, by construction of Q

    y = np.array([r_o, pr0, theta_o, ptheta0, 0.0, 0.0])
    h = 1.0e-6 * max(1.0, r_o)

    r_plus = M + np.sqrt(M * M - a * a)
    r_stop = r_horizon_factor * r_plus
    r_max = 2.0 * r_o

    outcome = OUTCOME_UNRESOLVED
    hit_r = -1.0

    for _i in range(n_steps):
        r = y[0]
        theta = y[2]
        pr = y[1]
        if r <= r_stop:
            outcome = OUTCOME_CAPTURED
            break
        if r >= r_max:
            outcome = OUTCOME_ESCAPED
            break

        dlambda = min(dlambda_max, 0.02 * max(r, 1.0) / (abs(pr) + 1.0e-6))
        k1 = _kerr_full_rhs(y, E, L, Q, a, M, h)
        k2 = _kerr_full_rhs(y + 0.5 * dlambda * k1, E, L, Q, a, M, h)
        k3 = _kerr_full_rhs(y + 0.5 * dlambda * k2, E, L, Q, a, M, h)
        k4 = _kerr_full_rhs(y + dlambda * k3, E, L, Q, a, M, h)
        y_new = y + (dlambda / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Reproject the momenta onto the exact constraint surfaces
        # (dr/dlambda)^2=R(r), (dtheta/dlambda)^2=Theta(theta), keeping whatever
        # sign the raw RK4 step produced. Without this, small integration errors
        # in pr/ptheta feed back into the adaptive step-size estimate and can
        # snowball into a spurious sign flip (the ray appearing to "bounce").
        r_new = y_new[0]
        theta_new = y_new[2]
        R_new = _kerr_R(r_new, E, L, Q, a, M)
        sign_pr = 1.0 if y_new[1] >= 0.0 else -1.0
        y_new[1] = sign_pr * np.sqrt(max(R_new, 0.0))
        Theta_new = _kerr_Theta(theta_new, E, L, Q, a)
        sign_ptheta = 1.0 if y_new[3] >= 0.0 else -1.0
        y_new[3] = sign_ptheta * np.sqrt(max(Theta_new, 0.0))
        if (theta - np.pi / 2.0) * (theta_new - np.pi / 2.0) < 0.0:
            r_cross = 0.5 * (r + r_new)
            if r_disk_inner <= r_cross <= r_disk_outer:
                outcome = OUTCOME_DISK
                hit_r = r_cross
                y = y_new
                break
        y = y_new

    return outcome, hit_r


@njit(cache=True)
def render_kerr_shadow_image(
    ny,
    nx,
    screen_half_width,
    screen_half_height,
    r_observer,
    theta_observer,
    a,
    M,
    r_disk_inner,
    r_disk_outer,
    dlambda_max=0.02,
    n_steps=4000,
    r_horizon_factor=1.001,
):
    """Render a Kerr black hole shadow / accretion disk image by backward ray-tracing.

    Parameters
    ----------
    ny, nx : int
        Image height and width, in pixels.
    screen_half_width, screen_half_height : float
        Half-extent of the camera screen (Bardeen impact-parameter space),
        in geometrized length units.
    r_observer : float
        Observer's radial (Boyer-Lindquist) coordinate; should be
        :math:`\\gg M` for the flat, far-away camera approximation to hold.
    theta_observer : float
        Observer's polar angle (inclination from the spin axis).
    a : float
        Kerr spin parameter, :math:`0 \\le a < M`.
    M : float
        Black hole mass, in geometrized units.
    r_disk_inner, r_disk_outer : float
        Inner and outer radius of a thin equatorial accretion disk (set
        ``r_disk_inner > r_disk_outer`` to omit the disk).
    dlambda_max : float, default=0.02
        Maximum Mino-time step size (the actual step is adaptively reduced
        wherever the radial "speed" ``|dr/dlambda|`` is large; see
        :func:`_trace_pixel_kerr`).
    n_steps : int, default=4000
        Maximum steps per ray.
    r_horizon_factor : float, default=1.001
        A ray is marked captured once :math:`r` drops below this factor
        times the outer horizon radius :math:`r_+`.

    Returns
    -------
    outcomes : ndarray of shape (ny, nx), dtype int64
        Per-pixel outcome code (0=escaped, 1=captured, 2=unresolved, 3=disk).
    hit_radii : ndarray of shape (ny, nx)
        Disk-crossing radius for disk-hit pixels, ``-1`` elsewhere.

    Examples
    --------
    >>> outcomes, hit_radii = render_kerr_shadow_image(
    ...     ny=20, nx=20, screen_half_width=15.0, screen_half_height=15.0,
    ...     r_observer=200.0, theta_observer=1.3, a=0.9, M=1.0,
    ...     r_disk_inner=1.0, r_disk_outer=0.0,
    ... )
    >>> outcomes.shape
    (20, 20)
    """
    outcomes = np.zeros((ny, nx), dtype=np.int64)
    hit_radii = np.full((ny, nx), -1.0)
    for iy in range(ny):
        beta = screen_half_height * (1.0 - 2.0 * iy / (ny - 1))
        for ix in range(nx):
            alpha = screen_half_width * (2.0 * ix / (nx - 1) - 1.0)
            outcome, hit_r = _trace_pixel_kerr(
                alpha,
                beta,
                r_observer,
                theta_observer,
                a,
                M,
                r_disk_inner,
                r_disk_outer,
                dlambda_max,
                n_steps,
                r_horizon_factor,
            )
            outcomes[iy, ix] = outcome
            hit_radii[iy, ix] = hit_r
    return outcomes, hit_radii
