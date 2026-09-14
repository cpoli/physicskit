"""A Numba-accelerated backward photon ray-tracer for Schwarzschild gravitational lensing.

Rather than integrating the full 4D null geodesic equation per pixel, this
module exploits two facts about a spherically symmetric spacetime: every
photon trajectory is confined to a single plane through the black hole
(by symmetry), and within that plane, the trajectory obeys the classic
Binet-style orbit equation in :math:`u = 1/r`:

.. math::

    \\frac{d^2 u}{d\\varphi^2} = -u + 3Mu^2

This is numerically robust (no turning-point sign ambiguities to handle,
unlike integrating :math:`dr/d\\lambda` directly) and fast: it reduces each
pixel's ray to a 2-variable ODE stepped in the orbital-plane angle
:math:`\\varphi`. To render an accretion disk, each step's in-plane
position is reconstructed in the observer's fixed 3D frame (using the
per-pixel orbital-plane basis fixed by its impact parameter geometry) and
checked against the disk's equatorial plane and radial extent.

Every camera ray is treated as parallel to the line of sight (an
orthographic projection), the standard and excellent approximation for an
observer at :math:`r_0 \\gg M` with a modest field of view.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = [
    "OUTCOME_CAPTURED",
    "OUTCOME_DISK",
    "OUTCOME_ESCAPED",
    "camera_basis",
    "render_shadow_image",
]

#: Pixel outcome code: the photon escaped to infinity without hitting the disk.
OUTCOME_ESCAPED = 0
#: Pixel outcome code: the photon crossed the event horizon.
OUTCOME_CAPTURED = 1
#: Pixel outcome code: the photon's orbit could not be resolved within the step
#: budget (deep near-critical winding); treated as captured for imaging purposes.
OUTCOME_UNRESOLVED = 2
#: Pixel outcome code: the photon crossed the accretion disk.
OUTCOME_DISK = 3


def camera_basis(inclination):
    """Build an orthonormal camera frame looking at the origin from a given inclination.

    The black hole sits at the origin with its spin/disk axis along
    :math:`+Z`. The observer is placed in the :math:`X`-:math:`Z` plane at
    polar angle ``inclination`` from the :math:`+Z` axis (``0`` = face-on,
    looking down the pole; ``pi/2`` = edge-on, in the disk plane).

    Parameters
    ----------
    inclination : float
        Observer inclination, in radians, measured from the disk's normal axis.

    Returns
    -------
    view_dir : ndarray of shape (3,)
        Unit vector from the observer toward the origin.
    right : ndarray of shape (3,)
        Unit vector spanning the camera's horizontal screen axis.
    up : ndarray of shape (3,)
        Unit vector spanning the camera's vertical screen axis.
    """
    n_obs = np.array([np.sin(inclination), 0.0, np.cos(inclination)])
    view_dir = -n_obs
    up_hint = np.array([0.0, 0.0, 1.0]) if abs(inclination) > 1e-8 else np.array([1.0, 0.0, 0.0])
    right = np.cross(view_dir, up_hint)
    right = right / np.linalg.norm(right)
    up = np.cross(right, view_dir)
    up = up / np.linalg.norm(up)
    return view_dir, right, up


@njit(cache=True)
def _trace_pixel(
    alpha,
    beta,
    obs_pos,
    right,
    up,
    view_dir,
    M,
    r_disk_inner,
    r_disk_outer,
    dphi,
    n_steps,
    r_horizon_factor,
):
    """Trace one camera ray; returns (outcome_code, hit_radius)."""
    ray_origin = obs_pos + alpha * right + beta * up
    r0 = np.sqrt(ray_origin[0] ** 2 + ray_origin[1] ** 2 + ray_origin[2] ** 2)

    cross_ov = np.cross(ray_origin, view_dir)
    b = np.sqrt(cross_ov[0] ** 2 + cross_ov[1] ** 2 + cross_ov[2] ** 2)

    if b < 1.0e-8:
        return OUTCOME_CAPTURED, -1.0

    e1 = ray_origin / r0
    proj = view_dir - np.dot(view_dir, e1) * e1
    proj_norm = np.sqrt(proj[0] ** 2 + proj[1] ** 2 + proj[2] ** 2)
    e2 = proj / proj_norm

    u = 1.0 / r0
    val = 1.0 / (b * b) - u * u + 2.0 * M * u**3
    val = max(val, 0.0)
    dudphi = np.sqrt(val)

    u_crit = 1.0 / (r_horizon_factor * 2.0 * M)
    phi = 0.0
    pos0 = (1.0 / u) * (np.cos(phi) * e1 + np.sin(phi) * e2)
    prev_z = pos0[2]

    outcome = OUTCOME_UNRESOLVED
    hit_r = -1.0

    for i in range(n_steps):
        if u >= u_crit:
            outcome = OUTCOME_CAPTURED
            break
        if u <= 0.0 and i > 0:
            outcome = OUTCOME_ESCAPED
            break

        k1u = dudphi
        k1v = -u + 3.0 * M * u * u
        u2 = u + 0.5 * dphi * k1u
        v2 = dudphi + 0.5 * dphi * k1v
        k2u = v2
        k2v = -u2 + 3.0 * M * u2 * u2
        u3 = u + 0.5 * dphi * k2u
        v3 = dudphi + 0.5 * dphi * k2v
        k3u = v3
        k3v = -u3 + 3.0 * M * u3 * u3
        u4 = u + dphi * k3u
        v4 = dudphi + dphi * k3v
        k4u = v4
        k4v = -u4 + 3.0 * M * u4 * u4

        u = u + (dphi / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        dudphi = dudphi + (dphi / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        phi = phi + dphi

        if u > 1.0e-10:
            r = 1.0 / u
            pos = r * (np.cos(phi) * e1 + np.sin(phi) * e2)
            z = pos[2]
            if z * prev_z < 0.0 and r_disk_inner <= r <= r_disk_outer:
                hit_r = r
                outcome = OUTCOME_DISK
                break
            prev_z = z

    return outcome, hit_r


@njit(cache=True)
def render_shadow_image(
    ny,
    nx,
    screen_half_width,
    screen_half_height,
    obs_pos,
    right,
    up,
    view_dir,
    M,
    r_disk_inner,
    r_disk_outer,
    dphi=0.01,
    n_steps=3000,
    r_horizon_factor=1.001,
):
    """Render a black hole shadow / accretion disk image by backward ray-tracing.

    Parameters
    ----------
    ny, nx : int
        Image height and width, in pixels.
    screen_half_width, screen_half_height : float
        Half-extent of the camera screen, in geometrized length units
        (impact-parameter space); e.g. ``screen_half_width=15*M`` frames a
        region a few times larger than the shadow.
    obs_pos : ndarray of shape (3,)
        Observer position in the black hole's rest frame.
    right, up, view_dir : ndarray of shape (3,)
        Camera basis vectors, e.g. from :func:`camera_basis`.
    M : float
        Black hole mass, in geometrized units.
    r_disk_inner, r_disk_outer : float
        Inner and outer radius of a thin equatorial accretion disk (set
        ``r_disk_inner > r_disk_outer`` or both to 0 to omit the disk).
    dphi : float, default=0.01
        Angular step size for the orbital-plane integration.
    n_steps : int, default=3000
        Maximum steps per ray.
    r_horizon_factor : float, default=1.001
        A ray is marked captured once :math:`r` drops below this factor
        times the horizon radius :math:`2M`.

    Returns
    -------
    outcomes : ndarray of shape (ny, nx), dtype int64
        Per-pixel outcome code (see :data:`OUTCOME_ESCAPED`,
        :data:`OUTCOME_CAPTURED`, :data:`OUTCOME_UNRESOLVED`,
        :data:`OUTCOME_DISK`).
    hit_radii : ndarray of shape (ny, nx)
        Disk-crossing radius for :data:`OUTCOME_DISK` pixels, ``-1``
        elsewhere.
    """
    outcomes = np.zeros((ny, nx), dtype=np.int64)
    hit_radii = np.full((ny, nx), -1.0)
    for iy in range(ny):
        beta = screen_half_height * (1.0 - 2.0 * iy / (ny - 1))
        for ix in range(nx):
            alpha = screen_half_width * (2.0 * ix / (nx - 1) - 1.0)
            outcome, hit_r = _trace_pixel(
                alpha,
                beta,
                obs_pos,
                right,
                up,
                view_dir,
                M,
                r_disk_inner,
                r_disk_outer,
                dphi,
                n_steps,
                r_horizon_factor,
            )
            outcomes[iy, ix] = outcome
            hit_radii[iy, ix] = hit_r
    return outcomes, hit_radii
