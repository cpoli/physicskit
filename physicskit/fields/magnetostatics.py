"""Magnetostatics: the Biot-Savart law for current-carrying wires, loops, and solenoids.

A steady current :math:`I` along a wire produces the magnetic field

.. math::

    \\mathbf B(\\mathbf r) = \\frac{\\mu_0 I}{4\\pi}\\oint
    \\frac{d\\boldsymbol\\ell \\times (\\mathbf r - \\mathbf r')}{|\\mathbf r - \\mathbf r'|^3}

(Biot and Savart 1820; Jackson, *Classical Electrodynamics*, 3rd ed., eq.
5.4). :func:`biot_savart_field` evaluates it for any wire given as a
polyline, integrating each straight segment in closed form rather than by
quadrature, so the only discretization error is the polygonal
approximation to a curved wire. Units are SI, like the rest of
:mod:`physicskit.fields` (:data:`MU0` is the vacuum permeability).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from physicskit.fields.electrodynamics import MU0

__all__ = [
    "biot_savart_field",
    "circular_loop_path",
    "solenoid_path",
    "loop_axial_field",
    "solenoid_axial_field",
    "infinite_wire_field",
    "magnetic_dipole_field",
]


@njit(cache=True)
def _segments_field(vertices, points, prefactor):
    n_pts = points.shape[0]
    out = np.zeros((n_pts, 3))
    for p in range(n_pts):
        bx = 0.0
        by = 0.0
        bz = 0.0
        for seg in range(vertices.shape[0] - 1):
            r1x = points[p, 0] - vertices[seg, 0]
            r1y = points[p, 1] - vertices[seg, 1]
            r1z = points[p, 2] - vertices[seg, 2]
            r2x = points[p, 0] - vertices[seg + 1, 0]
            r2y = points[p, 1] - vertices[seg + 1, 1]
            r2z = points[p, 2] - vertices[seg + 1, 2]
            n1 = np.sqrt(r1x * r1x + r1y * r1y + r1z * r1z)
            n2 = np.sqrt(r2x * r2x + r2y * r2y + r2z * r2z)
            cx = r1y * r2z - r1z * r2y
            cy = r1z * r2x - r1x * r2z
            cz = r1x * r2y - r1y * r2x
            cross2 = cx * cx + cy * cy + cz * cz
            dot = r1x * r2x + r1y * r2y + r1z * r2z
            # n1 n2 + r1.r2 cancels catastrophically when the point sits
            # beside a long segment (r1.r2 ~ -n1 n2); rewrite it there as
            # |r1 x r2|^2 / (n1 n2 - r1.r2).
            if dot >= 0.0:
                lsum = n1 * n2 + dot
            else:
                lsum = cross2 / (n1 * n2 - dot)
            # lsum -> 0 on the segment itself, where the field is singular.
            if cross2 == 0.0 or lsum <= 0.0:
                continue
            f = (n1 + n2) / (n1 * n2 * lsum)
            bx += f * cx
            by += f * cy
            bz += f * cz
        out[p, 0] = prefactor * bx
        out[p, 1] = prefactor * by
        out[p, 2] = prefactor * bz
    return out


def biot_savart_field(path, points, current: float = 1.0, mu: float = MU0) -> np.ndarray:
    """Magnetic field of a current-carrying polyline wire, by the Biot-Savart law.

    Each straight segment from :math:`\\mathbf a` to :math:`\\mathbf b` is
    integrated exactly: with :math:`\\mathbf r_1 = \\mathbf r - \\mathbf a`
    and :math:`\\mathbf r_2 = \\mathbf r - \\mathbf b`,

    .. math::

        \\mathbf B_{\\text{seg}} = \\frac{\\mu I}{4\\pi}
        \\frac{(r_1 + r_2)\\,(\\mathbf r_1 \\times \\mathbf r_2)}
        {r_1 r_2\\,(r_1 r_2 + \\mathbf r_1\\cdot\\mathbf r_2)},

    the vector form of the textbook finite-wire result
    :math:`B = \\mu I(\\sin\\theta_2 - \\sin\\theta_1)/(4\\pi s)` (Griffiths,
    *Introduction to Electrodynamics*, 4th ed., Example 5.5; J. D. Hanson
    and S. P. Hirshman, Phys. Plasmas 9, 4410 (2002)). Points on a
    segment or its straight-line extension get no contribution from it
    (the field of a segment vanishes on its own axis).

    Parameters
    ----------
    path : array_like, shape (n_vertices, 3)
        Wire vertices, in meters, in the direction of current flow. Close a
        loop by repeating the first vertex at the end (as
        :func:`circular_loop_path` does).
    points : array_like, shape (..., 3)
        Observation points.
    current : float, default=1.0
        Current in amperes.
    mu : float, default=MU0
        Permeability of the medium.

    Returns
    -------
    ndarray, shape (..., 3)
        Magnetic field in tesla.

    Examples
    --------
    A long straight wire recovers :math:`\\mu_0 I / (2\\pi s)`:

    >>> wire = [[0, 0, -1e4], [0, 0, 1e4]]
    >>> B = biot_savart_field(wire, [[0.1, 0.0, 0.0]], current=1.0)
    >>> round(float(B[0, 1] / infinite_wire_field(1.0, 0.1)), 6)
    1.0
    """
    verts = np.ascontiguousarray(path, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] < 2:
        raise ValueError("path must have shape (n_vertices >= 2, 3)")
    pts = np.asarray(points, dtype=np.float64)
    flat = np.ascontiguousarray(pts.reshape(-1, 3))
    B = _segments_field(verts, flat, mu * current / (4.0 * np.pi))
    return B.reshape(pts.shape)


def circular_loop_path(radius: float, n_segments: int = 256, center=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Vertices of a closed circular loop in a plane of constant ``z``.

    The loop runs counter-clockwise seen from ``+z``, so a positive current
    produces a field along ``+z`` at its center.

    Parameters
    ----------
    radius : float
        Loop radius in meters.
    n_segments : int, default=256
        Number of straight segments; the polygon's on-axis field differs
        from a true circle's by :math:`O(1/n^2)`.
    center : array_like, shape (3,), default=(0, 0, 0)
        Loop center.

    Returns
    -------
    ndarray, shape (n_segments + 1, 3)
        Vertices, with the first repeated at the end.

    Examples
    --------
    >>> path = circular_loop_path(1.0, n_segments=4)
    >>> path.shape, bool(np.allclose(path[0], path[-1]))
    ((5, 3), True)
    """
    phi = np.linspace(0.0, 2.0 * np.pi, n_segments + 1)
    c = np.asarray(center, dtype=np.float64)
    path = np.stack([radius * np.cos(phi), radius * np.sin(phi), np.zeros_like(phi)], axis=1) + c
    path[-1] = path[0]
    return path


def solenoid_path(radius: float, length: float, n_turns: int, points_per_turn: int = 64, center=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Vertices of a finite helical solenoid wound along ``z``.

    Parameters
    ----------
    radius : float
        Winding radius in meters.
    length : float
        Axial length in meters, from ``z = -length/2`` to ``+length/2``
        about ``center``.
    n_turns : int
        Number of turns.
    points_per_turn : int, default=64
        Polyline vertices per turn.
    center : array_like, shape (3,), default=(0, 0, 0)
        Solenoid center.

    Returns
    -------
    ndarray, shape (n_turns * points_per_turn + 1, 3)
        Helix vertices, winding counter-clockwise seen from ``+z`` (field
        inside along ``+z`` for positive current).

    Examples
    --------
    >>> path = solenoid_path(0.01, 0.1, n_turns=10)
    >>> round(float(path[-1, 2] - path[0, 2]), 12)
    0.1
    """
    n = n_turns * points_per_turn
    phi = np.linspace(0.0, 2.0 * np.pi * n_turns, n + 1)
    z = np.linspace(-0.5 * length, 0.5 * length, n + 1)
    return np.stack([radius * np.cos(phi), radius * np.sin(phi), z], axis=1) + np.asarray(center, dtype=np.float64)


def loop_axial_field(current: float, radius: float, z, mu: float = MU0) -> np.ndarray:
    """On-axis field of a circular current loop.

    .. math::

        B_z(z) = \\frac{\\mu I R^2}{2 (R^2 + z^2)^{3/2}}

    (Griffiths Example 5.6; Jackson eq. 5.40 on the axis).

    Parameters
    ----------
    current : float
        Current in amperes.
    radius : float
        Loop radius ``R`` in meters.
    z : array_like
        Axial distance from the loop's center.
    mu : float, default=MU0
        Permeability of the medium.

    Returns
    -------
    ndarray
        :math:`B_z` in tesla.

    Examples
    --------
    >>> round(float(loop_axial_field(1.0, 1.0, 0.0) / MU0), 12)  # mu0 I / (2R)
    0.5
    """
    z = np.asarray(z, dtype=np.float64)
    return mu * current * radius**2 / (2.0 * (radius**2 + z**2) ** 1.5)


def solenoid_axial_field(current: float, n_turns: int, radius: float, length: float, z, mu: float = MU0) -> np.ndarray:
    """On-axis field of a finite solenoid (uniform current sheet approximation).

    .. math::

        B_z(z) = \\frac{\\mu n I}{2}\\left[
        \\frac{z + L/2}{\\sqrt{R^2 + (z + L/2)^2}} -
        \\frac{z - L/2}{\\sqrt{R^2 + (z - L/2)^2}}\\right],
        \\qquad n = N/L,

    obtained by integrating :func:`loop_axial_field` over the winding
    (Griffiths Problem 5.11). Tends to :math:`\\mu n I` deep inside a long
    solenoid and to half that at either end.

    Parameters
    ----------
    current : float
        Current in amperes.
    n_turns : int
        Total number of turns ``N``.
    radius : float
        Winding radius ``R`` in meters.
    length : float
        Solenoid length ``L`` in meters, centered on ``z = 0``.
    z : array_like
        Axial position.
    mu : float, default=MU0
        Permeability of the medium.

    Returns
    -------
    ndarray
        :math:`B_z` in tesla.

    Examples
    --------
    >>> ratio = solenoid_axial_field(1.0, 1000, 0.01, 1.0, [0.0, 0.5]) / (MU0 * 1000)
    >>> ratio.round(4).tolist()  # center, and one end of a long solenoid
    [0.9998, 0.5]
    """
    z = np.asarray(z, dtype=np.float64)
    n = n_turns / length
    zp = z + 0.5 * length
    zm = z - 0.5 * length
    return 0.5 * mu * n * current * (zp / np.sqrt(radius**2 + zp**2) - zm / np.sqrt(radius**2 + zm**2))


def infinite_wire_field(current: float, s, mu: float = MU0) -> np.ndarray:
    """Field magnitude at distance ``s`` from an infinite straight wire, :math:`\\mu I / (2\\pi s)`.

    (Griffiths eq. 5.38.) The field circles the wire in the right-hand
    sense about the current.

    Parameters
    ----------
    current : float
        Current in amperes.
    s : array_like
        Perpendicular distance from the wire, in meters.
    mu : float, default=MU0
        Permeability of the medium.

    Returns
    -------
    ndarray
        :math:`|\\mathbf B|` in tesla.

    Examples
    --------
    >>> round(float(infinite_wire_field(1.0, 1.0)), 10)  # 2e-7 T per ampere at 1 m
    2e-07
    """
    s = np.asarray(s, dtype=np.float64)
    return mu * current / (2.0 * np.pi * s)


def magnetic_dipole_field(m, points, origin=(0.0, 0.0, 0.0), mu: float = MU0) -> np.ndarray:
    """Magnetic field of an ideal point dipole.

    .. math::

        \\mathbf B(\\mathbf x) = \\frac{\\mu}{4\\pi}
        \\frac{3\\hat{\\mathbf n}(\\mathbf m\\cdot\\hat{\\mathbf n}) - \\mathbf m}{r^3}

    (Jackson eq. 5.56, away from the origin). A small loop of area
    :math:`A` carrying current :math:`I` has :math:`m = IA` along its
    normal, and its Biot-Savart field approaches this one far away.

    Parameters
    ----------
    m : array_like, shape (3,)
        Magnetic moment in A m^2.
    points : array_like, shape (..., 3)
        Observation points.
    origin : array_like, shape (3,), default=(0, 0, 0)
        Dipole location.
    mu : float, default=MU0
        Permeability of the medium.

    Returns
    -------
    ndarray, shape (..., 3)
        Field in tesla.

    Examples
    --------
    >>> B = magnetic_dipole_field([0, 0, 1.0], [[0, 0, 1.0]], mu=4 * np.pi)
    >>> B.round(12).tolist()
    [[0.0, 0.0, 2.0]]
    """
    m = np.asarray(m, dtype=np.float64)
    x = np.asarray(points, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    r = np.linalg.norm(x, axis=-1, keepdims=True)
    n = x / r
    return mu * (3.0 * n * np.sum(n * m, axis=-1, keepdims=True) - m) / (4.0 * np.pi * r**3)
