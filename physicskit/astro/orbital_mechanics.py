r"""Two-body Keplerian orbital mechanics: elements, state vectors, and transfers.

Uses **gravitational units with** :math:`G=1` by default; the standard
gravitational parameter :math:`\mu=GM` appears directly as a function
argument, the normal convention in orbital mechanics regardless of unit
system (the same style :mod:`physicskit.relativity` uses for its own
geometrized units).

- :func:`orbital_period`, :func:`vis_viva_speed` -- Kepler's third law and
  the vis-viva equation.
- :func:`orbital_elements_from_state`, :func:`state_from_orbital_elements`
  -- converting between Cartesian state vectors and the six classical
  orbital elements.
- :func:`hohmann_transfer` -- the two-burn transfer between circular orbits.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "orbital_period",
    "vis_viva_speed",
    "orbital_elements_from_state",
    "state_from_orbital_elements",
    "hohmann_transfer",
]


def orbital_period(a, mu):
    r"""Kepler's third law, :math:`T=2\pi\sqrt{a^3/\mu}`.

    Parameters
    ----------
    a : float
        Semi-major axis (bound orbit, ``a > 0``).
    mu : float
        Gravitational parameter, :math:`\mu=GM`.

    Returns
    -------
    float

    Examples
    --------
    >>> round(orbital_period(1.0, 1.0) / (2 * np.pi), 6)
    1.0
    """
    return float(2.0 * np.pi * np.sqrt(a**3 / mu))


def vis_viva_speed(r, a, mu):
    r"""The vis-viva equation, :math:`v=\sqrt{\mu\left(2/r-1/a\right)}`.

    Works for ``a > 0`` (ellipse), ``a < 0`` (hyperbola), and ``a = inf``
    (parabolic escape trajectory, where the formula reduces to
    :math:`v=\sqrt{2\mu/r}`).

    Parameters
    ----------
    r : float
        Current orbital radius.
    a : float
        Semi-major axis.
    mu : float
        Gravitational parameter.

    Returns
    -------
    float

    Examples
    --------
    >>> round(vis_viva_speed(1.0, np.inf, 1.0), 6)
    1.414214
    """
    return float(np.sqrt(mu * (2.0 / r - 1.0 / a)))


def orbital_elements_from_state(r_vec, v_vec, mu):
    r"""Classical (Keplerian) orbital elements from a Cartesian state vector.

    Returns semi-major axis, eccentricity, inclination, right ascension of
    the ascending node, argument of periapsis, and true anomaly (angles
    in radians). For a near-equatorial orbit (:math:`i\approx0` or
    :math:`\pi`), the node vector is near zero and ``raan`` is undefined;
    this implementation returns ``raan=0.0`` in that case rather than
    raising -- a documented simplification, not a fully general
    equatorial-orbit treatment.

    Parameters
    ----------
    r_vec, v_vec : ndarray of shape (3,)
        Position and velocity.
    mu : float
        Gravitational parameter.

    Returns
    -------
    a, e, i, raan, argp, nu : float

    Examples
    --------
    >>> r_vec, v_vec = state_from_orbital_elements(1.0, 0.3, 0.5, 1.0, 0.7, 1.2, mu=1.0)
    >>> a, e, i, raan, argp, nu = orbital_elements_from_state(r_vec, v_vec, mu=1.0)
    >>> [round(x, 6) for x in (a, e, i, raan, argp, nu)]
    [1.0, 0.3, 0.5, 1.0, 0.7, 1.2]
    """
    r_vec = np.asarray(r_vec, dtype=float)
    v_vec = np.asarray(v_vec, dtype=float)
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    n_vec = np.cross([0.0, 0.0, 1.0], h_vec)
    n = np.linalg.norm(n_vec)
    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
    e = np.linalg.norm(e_vec)

    energy = v**2 / 2.0 - mu / r
    a = np.inf if abs(e - 1.0) < 1e-12 else -mu / (2.0 * energy)

    i = float(np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0)))

    if n > 1e-12:
        raan = float(np.arccos(np.clip(n_vec[0] / n, -1.0, 1.0)))
        if n_vec[1] < 0:
            raan = 2 * np.pi - raan
    else:
        raan = 0.0

    if n > 1e-12 and e > 1e-12:
        argp = float(np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1.0, 1.0)))
        if e_vec[2] < 0:
            argp = 2 * np.pi - argp
    else:
        argp = 0.0

    if e > 1e-12:
        nu = float(np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0)))
        if np.dot(r_vec, v_vec) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = 0.0

    return float(a), float(e), i, raan, argp, nu


def state_from_orbital_elements(a, e, i, raan, argp, nu, mu):
    """The inverse of :func:`orbital_elements_from_state`.

    Parameters
    ----------
    a, e, i, raan, argp, nu : float
        Classical orbital elements (angles in radians).
    mu : float
        Gravitational parameter.

    Returns
    -------
    r_vec, v_vec : ndarray of shape (3,)
    """
    p = a * (1.0 - e**2)
    r = p / (1.0 + e * np.cos(nu))
    r_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(argp), np.sin(argp)

    # 3-1-3 Euler rotation (raan, i, argp): perifocal -> reference frame.
    R = np.array(
        [
            [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
            [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
            [sw * si, cw * si, ci],
        ]
    )
    return R @ r_pf, R @ v_pf


def hohmann_transfer(r1, r2, mu):
    r"""The two burns and transfer time for a Hohmann transfer.

    Parameters
    ----------
    r1 : float
        Initial circular orbit radius.
    r2 : float
        Final circular orbit radius.
    mu : float
        Gravitational parameter.

    Returns
    -------
    delta_v1 : float
        Speed change (magnitude) to leave the initial circular orbit and
        enter the transfer ellipse.
    delta_v2 : float
        Speed change (magnitude) to circularize at ``r2``.
    transfer_time : float
        Time to complete the transfer, half the transfer ellipse's period.

    Examples
    --------
    >>> dv1, dv2, t = hohmann_transfer(1.0, 4.0, mu=1.0)
    >>> dv1 > 0 and dv2 > 0 and t > 0
    True
    """
    a_t = (r1 + r2) / 2.0
    v1_circular = vis_viva_speed(r1, r1, mu)
    v1_transfer = vis_viva_speed(r1, a_t, mu)
    v2_transfer = vis_viva_speed(r2, a_t, mu)
    v2_circular = vis_viva_speed(r2, r2, mu)
    delta_v1 = abs(v1_transfer - v1_circular)
    delta_v2 = abs(v2_circular - v2_transfer)
    transfer_time = orbital_period(a_t, mu) / 2.0
    return float(delta_v1), float(delta_v2), float(transfer_time)
