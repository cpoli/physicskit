"""2D billiard systems with Numba-accelerated ray-boundary collision and specular reflection.

Every billiard's boundary is represented as a set of straight *segments*
(``(x1, y1, x2, y2)`` rows) and circular *arcs* (``(cx, cy, r, theta1, theta2)``
rows, always swept counter-clockwise from ``theta1`` to ``theta2`` modulo
``2*pi``). Walls are always constructed in traversal order matching the
direction of increasing boundary arclength ``s``, which is what lets a single
shared, jitted ray-tracer serve every billiard shape (Circle, Rectangle, Sinai,
Bunimovich Stadium, Truncated Circle, Ellipse) without per-shape collision
code. The Ellipse billiard is the one shape without an exact representation
in this scheme (there is no closed-form ray-ellipse intersection kernel here)
and is instead approximated as a fine closed polygon of straight segments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit, prange
from numpy.typing import ArrayLike, NDArray

from physicskit.chaos.core.base_system import BilliardSystem
from physicskit.chaos.exceptions import InvalidParameterError

TWO_PI: float = 2.0 * np.pi
_EPS: float = 1e-9
_NO_HIT: float = 1e18

#: A boundary wall specification consumed by ``_finalize_boundary``: either
#: ``("seg", (x1, y1, x2, y2))`` or ``("arc", (cx, cy, r, theta1, theta2),
#: is_full_circle)``.
WallSpec = tuple[str, tuple[float, ...]] | tuple[str, tuple[float, ...], bool]

# ---------------------------------------------------------------------------
# Low-level Numba kernels
# ---------------------------------------------------------------------------


@njit(cache=True)
def reflect(vx: float, vy: float, nx: float, ny: float) -> tuple[float, float]:
    """Specular reflection of a velocity about a wall normal.

    Computes ``v' = v - 2 (v . n) n``. The sign of `nx`, `ny` is irrelevant:
    flipping the normal leaves the result unchanged.

    Parameters
    ----------
    vx, vy : float
        Incoming velocity components.
    nx, ny : float
        Unit normal components (need not be outward-facing).

    Returns
    -------
    rvx, rvy : float
        Reflected (outgoing) velocity components.
    """
    d = vx * nx + vy * ny
    return vx - 2.0 * d * nx, vy - 2.0 * d * ny


@njit(cache=True)
def _segment_hit(px: float, py: float, vx: float, vy: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Ray-segment intersection distance.

    Parameters
    ----------
    px, py : float
        Ray origin.
    vx, vy : float
        Ray direction (need not be unit length).
    x1, y1, x2, y2 : float
        Segment endpoints.

    Returns
    -------
    float
        The intersection distance `t` such that ``(px + t*vx, py + t*vy)``
        lies on the segment, or the ``_NO_HIT`` sentinel if there is none.
    """
    ex, ey = x2 - x1, y2 - y1
    denom = vx * ey - vy * ex
    if abs(denom) < 1e-14:
        return _NO_HIT
    t = ((x1 - px) * ey - (y1 - py) * ex) / denom
    if t < _EPS:
        return _NO_HIT
    if abs(ex) > abs(ey):
        u = (px + t * vx - x1) / ex
    else:
        u = (py + t * vy - y1) / ey
    if u < -_EPS or u > 1.0 + _EPS:
        return _NO_HIT
    return t


@njit(cache=True)
def _angle_in_range(ang: float, theta1: float, theta2: float) -> bool:
    """Whether angle `ang` lies on the counter-clockwise sweep from `theta1` to `theta2`.

    Parameters
    ----------
    ang : float
        Angle to test, in radians (any range).
    theta1, theta2 : float
        Sweep start and end angles, in radians (any range); the sweep is
        always interpreted as going counter-clockwise from `theta1` to
        `theta2`, modulo ``2*pi``.

    Returns
    -------
    bool
        ``True`` if `ang` lies within the swept range.
    """
    a = ang % TWO_PI
    lo = theta1 % TWO_PI
    hi = theta2 % TWO_PI
    if lo <= hi:
        return (lo - 1e-7) <= a <= (hi + 1e-7)
    return a >= (lo - 1e-7) or a <= (hi + 1e-7)


@njit(cache=True)
def _arc_hit(
    px: float,
    py: float,
    vx: float,
    vy: float,
    cx: float,
    cy: float,
    r: float,
    theta1: float,
    theta2: float,
    full_circle: bool,
) -> float:
    """Nearest positive ray-circle intersection distance, respecting the arc's angular span.

    Parameters
    ----------
    px, py : float
        Ray origin.
    vx, vy : float
        Ray direction (need not be unit length).
    cx, cy, r : float
        Circle center and radius.
    theta1, theta2 : float
        Arc sweep start and end angles, in radians; ignored if `full_circle`.
    full_circle : bool
        If ``True``, treat the circle as a full circle and skip the angular
        range check.

    Returns
    -------
    float
        The nearest intersection distance `t` (with ``t > 0``), or the
        ``_NO_HIT`` sentinel if there is none within the arc's span.
    """
    fx, fy = px - cx, py - cy
    a = vx * vx + vy * vy
    b = 2.0 * (fx * vx + fy * vy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return _NO_HIT
    sq = np.sqrt(disc)
    best = _NO_HIT
    for t in ((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)):
        if t > _EPS and t < best:
            if full_circle:
                best = t
            else:
                hx, hy = px + t * vx - cx, py + t * vy - cy
                ang = np.arctan2(hy, hx)
                if _angle_in_range(ang, theta1, theta2):
                    best = t
    return best


@njit(cache=True)
def _find_collision(
    px: float,
    py: float,
    vx: float,
    vy: float,
    segments: NDArray[np.float64],
    arcs: NDArray[np.float64],
    arc_full: NDArray[np.bool_],
) -> tuple[float, int, int, float, int, int]:
    """Find the nearest (and second-nearest) boundary wall hit by a ray.

    The second-nearest hit lets :func:`bounce` detect exact corner hits,
    where two walls tie for nearest and both must contribute to the
    reflection.

    Parameters
    ----------
    px, py : float
        Ray origin.
    vx, vy : float
        Ray direction (need not be unit length).
    segments : ndarray of float, shape (n_segments, 4)
        Straight walls, each row ``(x1, y1, x2, y2)``.
    arcs : ndarray of float, shape (n_arcs, 5)
        Circular-arc walls, each row ``(cx, cy, r, theta1, theta2)``.
    arc_full : ndarray of bool, shape (n_arcs,)
        Whether each arc is a full circle.

    Returns
    -------
    t : float
        Distance to the nearest hit (``_NO_HIT`` if the boundary is empty).
    wall_type : int
        ``0`` if the nearest hit is a segment, ``1`` if an arc, ``-1`` if none.
    wall_idx : int
        Row index of the nearest wall into `segments` or `arcs`, ``-1`` if none.
    t2 : float
        Distance to the second-nearest hit (``_NO_HIT`` if there is none).
    wall_type2 : int
        Type of the second-nearest wall, ``-1`` if none.
    wall_idx2 : int
        Row index of the second-nearest wall, ``-1`` if none.
    """
    best_t = _NO_HIT
    best_type = -1
    best_idx = -1
    second_t = _NO_HIT
    second_type = -1
    second_idx = -1
    for i in range(segments.shape[0]):
        t = _segment_hit(
            px,
            py,
            vx,
            vy,
            segments[i, 0],
            segments[i, 1],
            segments[i, 2],
            segments[i, 3],
        )
        if t < best_t:
            second_t, second_type, second_idx = best_t, best_type, best_idx
            best_t, best_type, best_idx = t, 0, i
        elif t < second_t:
            second_t, second_type, second_idx = t, 0, i
    for j in range(arcs.shape[0]):
        t = _arc_hit(
            px,
            py,
            vx,
            vy,
            arcs[j, 0],
            arcs[j, 1],
            arcs[j, 2],
            arcs[j, 3],
            arcs[j, 4],
            arc_full[j],
        )
        if t < best_t:
            second_t, second_type, second_idx = best_t, best_type, best_idx
            best_t, best_type, best_idx = t, 1, j
        elif t < second_t:
            second_t, second_type, second_idx = t, 1, j
    return best_t, best_type, best_idx, second_t, second_type, second_idx


@njit(cache=True)
def _wall_normal(
    hx: float,
    hy: float,
    wall_type: int,
    wall_idx: int,
    segments: NDArray[np.float64],
    arcs: NDArray[np.float64],
) -> tuple[float, float]:
    """Unit normal to a wall at a given hit point (sign is arbitrary).

    Parameters
    ----------
    hx, hy : float
        Point on the wall at which to evaluate the normal.
    wall_type : int
        ``0`` for a segment, ``1`` for an arc.
    wall_idx : int
        Row index of the wall into `segments` or `arcs`.
    segments : ndarray of float, shape (n_segments, 4)
        Straight walls, each row ``(x1, y1, x2, y2)``.
    arcs : ndarray of float, shape (n_arcs, 5)
        Circular-arc walls, each row ``(cx, cy, r, theta1, theta2)``.

    Returns
    -------
    nx, ny : float
        Unit normal components at ``(hx, hy)``.
    """
    if wall_type == 0:
        x1, y1, x2, y2 = (
            segments[wall_idx, 0],
            segments[wall_idx, 1],
            segments[wall_idx, 2],
            segments[wall_idx, 3],
        )
        ex, ey = x2 - x1, y2 - y1
        elen = np.sqrt(ex * ex + ey * ey)
        return -ey / elen, ex / elen
    cx, cy = arcs[wall_idx, 0], arcs[wall_idx, 1]
    nx, ny = hx - cx, hy - cy
    nlen = np.sqrt(nx * nx + ny * ny)
    return nx / nlen, ny / nlen


@njit(cache=True)
def bounce(
    px: float,
    py: float,
    vx: float,
    vy: float,
    segments: NDArray[np.float64],
    arcs: NDArray[np.float64],
    arc_full: NDArray[np.bool_],
) -> tuple[float, float, float, float, int, int]:
    """Advance a ray to its next boundary collision and specularly reflect it.

    If the ray lands (near-)exactly on a corner where two walls meet, both
    walls' normals contribute (sequential reflection) -- unless the two walls
    share the same normal (a smooth tangential join, e.g. where a Bunimovich
    stadium's straight edge meets its semicircle), in which case reflecting
    twice would spuriously cancel out and only one reflection is applied.

    Parameters
    ----------
    px, py : float
        Current position.
    vx, vy : float
        Current (unit) velocity.
    segments : ndarray of float, shape (n_segments, 4)
        Straight walls, each row ``(x1, y1, x2, y2)``.
    arcs : ndarray of float, shape (n_arcs, 5)
        Circular-arc walls, each row ``(cx, cy, r, theta1, theta2)``.
    arc_full : ndarray of bool, shape (n_arcs,)
        Whether each arc is a full circle.

    Returns
    -------
    hx, hy : float
        Position of the collision (hit) point.
    rvx, rvy : float
        Outgoing (post-reflection) velocity.
    wall_type : int
        ``0`` if a segment was hit, ``1`` if an arc.
    wall_idx : int
        Row index of the hit wall into `segments` or `arcs`.
    """
    t, wtype, widx, t2, wtype2, widx2 = _find_collision(px, py, vx, vy, segments, arcs, arc_full)
    hx, hy = px + t * vx, py + t * vy
    nx, ny = _wall_normal(hx, hy, wtype, widx, segments, arcs)
    rvx, rvy = reflect(vx, vy, nx, ny)
    if wtype2 != -1 and (t2 - t) < 1e-7:
        nx2, ny2 = _wall_normal(hx, hy, wtype2, widx2, segments, arcs)
        if abs(nx * nx2 + ny * ny2) < 0.999:
            rvx, rvy = reflect(rvx, rvy, nx2, ny2)
    return hx, hy, rvx, rvy, wtype, widx


@njit(cache=True)
def simulate_bounces(
    px: float,
    py: float,
    vx: float,
    vy: float,
    segments: NDArray[np.float64],
    arcs: NDArray[np.float64],
    arc_full: NDArray[np.bool_],
    n_bounces: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    """Trace a ray through ``n_bounces`` specular reflections.

    Parameters
    ----------
    px, py : float
        Initial position.
    vx, vy : float
        Initial (unit) velocity.
    segments : ndarray of float, shape (n_segments, 4)
        Straight walls, each row ``(x1, y1, x2, y2)``.
    arcs : ndarray of float, shape (n_arcs, 5)
        Circular-arc walls, each row ``(cx, cy, r, theta1, theta2)``.
    arc_full : ndarray of bool, shape (n_arcs,)
        Whether each arc is a full circle.
    n_bounces : int
        Number of reflections to simulate.

    Returns
    -------
    xs, ys : ndarray of float, shape (n_bounces,)
        Hit-point coordinates at each bounce.
    vxs, vys : ndarray of float, shape (n_bounces,)
        Outgoing velocity components at each bounce.
    wtypes : ndarray of int64, shape (n_bounces,)
        Wall type (``0`` segment, ``1`` arc) struck at each bounce.
    widxs : ndarray of int64, shape (n_bounces,)
        Wall row index struck at each bounce.
    """
    xs = np.empty(n_bounces)
    ys = np.empty(n_bounces)
    vxs = np.empty(n_bounces)
    vys = np.empty(n_bounces)
    wtypes = np.empty(n_bounces, dtype=np.int64)
    widxs = np.empty(n_bounces, dtype=np.int64)
    for i in range(n_bounces):
        px, py, vx, vy, wt, wi = bounce(px, py, vx, vy, segments, arcs, arc_full)
        xs[i] = px
        ys[i] = py
        vxs[i] = vx
        vys[i] = vy
        wtypes[i] = wt
        widxs[i] = wi
    return xs, ys, vxs, vys, wtypes, widxs


@njit(cache=True, parallel=True)
def simulate_bounces_many(
    px: float,
    py: float,
    angles: NDArray[np.float64],
    segments: NDArray[np.float64],
    arcs: NDArray[np.float64],
    arc_full: NDArray[np.bool_],
    n_bounces: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    """Trace many independent rays from one point, in parallel across rays.

    Same physics as :func:`simulate_bounces`, run once per entry of `angles`
    (all launched from the same ``(px, py)``) and parallelized across rays
    with Numba's ``prange`` -- the embarrassingly parallel workload behind a
    Poincare section (:func:`physicskit.chaos.visualizers.phase_space.plot_poincare_section`),
    where each ray is fully independent of every other.

    Parameters
    ----------
    px, py : float
        Shared initial position for every ray.
    angles : ndarray of float, shape (n_rays,)
        Initial launch angle (radians) of each ray.
    segments : ndarray of float, shape (n_segments, 4)
        Straight walls, each row ``(x1, y1, x2, y2)``.
    arcs : ndarray of float, shape (n_arcs, 5)
        Circular-arc walls, each row ``(cx, cy, r, theta1, theta2)``.
    arc_full : ndarray of bool, shape (n_arcs,)
        Whether each arc is a full circle.
    n_bounces : int
        Number of reflections to simulate per ray.

    Returns
    -------
    xs, ys : ndarray of float, shape (n_rays, n_bounces)
        Hit-point coordinates at each bounce of each ray.
    vxs, vys : ndarray of float, shape (n_rays, n_bounces)
        Outgoing velocity components at each bounce of each ray.
    wtypes : ndarray of int64, shape (n_rays, n_bounces)
        Wall type (``0`` segment, ``1`` arc) struck at each bounce.
    widxs : ndarray of int64, shape (n_rays, n_bounces)
        Wall row index struck at each bounce.
    """
    n_rays = angles.shape[0]
    xs = np.empty((n_rays, n_bounces))
    ys = np.empty((n_rays, n_bounces))
    vxs = np.empty((n_rays, n_bounces))
    vys = np.empty((n_rays, n_bounces))
    wtypes = np.empty((n_rays, n_bounces), dtype=np.int64)
    widxs = np.empty((n_rays, n_bounces), dtype=np.int64)
    for r in prange(n_rays):  # type: ignore[attr-defined]  # numba lacks type stubs for prange
        x, y = px, py
        vx, vy = np.cos(angles[r]), np.sin(angles[r])
        for i in range(n_bounces):
            x, y, vx, vy, wt, wi = bounce(x, y, vx, vy, segments, arcs, arc_full)
            xs[r, i] = x
            ys[r, i] = y
            vxs[r, i] = vx
            vys[r, i] = vy
            wtypes[r, i] = wt
            widxs[r, i] = wi
    return xs, ys, vxs, vys, wtypes, widxs


# ---------------------------------------------------------------------------
# BilliardSystem base implementation
# ---------------------------------------------------------------------------


class _WallBoundaryMixin:
    """Shared arclength (``s``) bookkeeping on top of a boundary built from an
    ordered list of walls (the order defines the direction of increasing ``s``).
    """

    def _finalize_boundary(self, walls: list[WallSpec]) -> None:
        """Build the segment/arc arrays and cumulative arclength offsets.

        Parameters
        ----------
        walls : list of tuple
            Wall specifications in traversal order (the direction of
            increasing arclength ``s``): each is either
            ``("seg", (x1, y1, x2, y2))`` or
            ``("arc", (cx, cy, r, theta1, theta2), is_full_circle)``.
        """
        seg_rows: list[tuple[float, ...]] = []
        arc_rows: list[tuple[float, ...]] = []
        arc_full: list[bool] = []
        order: list[tuple[str, int]] = []
        for wall in walls:
            kind = wall[0]
            if kind == "seg":
                seg_rows.append(wall[1])
                order.append(("seg", len(seg_rows) - 1))
            else:
                arc_rows.append(wall[1])
                arc_full.append(bool(wall[2]) if len(wall) > 2 else False)
                order.append(("arc", len(arc_rows) - 1))

        segments = np.asarray(seg_rows, dtype=np.float64).reshape(-1, 4)
        arcs = np.asarray(arc_rows, dtype=np.float64).reshape(-1, 5)
        arc_full_arr = np.asarray(arc_full, dtype=np.bool_)

        seg_lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1]) if segments.shape[0] else np.empty(0)
        if arcs.shape[0]:
            sweep = (arcs[:, 4] - arcs[:, 3]) % TWO_PI
            sweep = np.where(arc_full_arr, TWO_PI, sweep)
            arc_lengths = arcs[:, 2] * sweep
        else:
            arc_lengths = np.empty(0)

        s_offsets_seg = np.zeros(segments.shape[0])
        s_offsets_arc = np.zeros(arcs.shape[0])
        running = 0.0
        for kind, idx in order:
            if kind == "seg":
                s_offsets_seg[idx] = running
                running += seg_lengths[idx]
            else:
                s_offsets_arc[idx] = running
                running += arc_lengths[idx]

        self._segments = segments
        self._arcs = arcs
        self._arc_full = arc_full_arr
        self._seg_lengths = seg_lengths
        self._arc_lengths = arc_lengths
        self._s_offsets_seg = s_offsets_seg
        self._s_offsets_arc = s_offsets_arc
        self._perimeter = float(running)

    def boundary_coordinate(
        self,
        hit_x: float,
        hit_y: float,
        wall_type: int,
        wall_idx: int,
        out_vx: float,
        out_vy: float,
    ) -> tuple[float, float]:
        """Map a boundary hit point to Poincare-section coordinates ``(s, sin_phi)``.

        Parameters
        ----------
        hit_x, hit_y : float
            Coordinates of the boundary hit point.
        wall_type : int
            ``0`` if `hit_x`, `hit_y` lies on a segment, ``1`` if on an arc.
        wall_idx : int
            Row index of the wall hit into ``self._segments`` or ``self._arcs``.
        out_vx, out_vy : float
            Outgoing (post-reflection) unit velocity at the hit point.

        Returns
        -------
        s : float
            Boundary arclength coordinate of the hit point.
        sin_phi : float
            Sine of the angle between the outgoing velocity and the local
            boundary tangent (the standard Birkhoff coordinate).
        """
        if wall_type == 0:
            x1, y1, x2, y2 = self._segments[wall_idx]
            ex, ey = x2 - x1, y2 - y1
            elen = np.hypot(ex, ey)
            u = ((hit_x - x1) * ex + (hit_y - y1) * ey) / (elen * elen)
            u = min(max(u, 0.0), 1.0)
            s = self._s_offsets_seg[wall_idx] + u * elen
            tx, ty = ex / elen, ey / elen
        else:
            cx, cy, r, theta1 = self._arcs[wall_idx, :4]
            ang = np.arctan2(hit_y - cy, hit_x - cx)
            sweep_pos = (ang - theta1) % TWO_PI
            arc_len = self._arc_lengths[wall_idx]
            local_s = min(max(sweep_pos * r, 0.0), arc_len)
            s = self._s_offsets_arc[wall_idx] + local_s
            tx, ty = -np.sin(ang), np.cos(ang)
        sin_phi = out_vx * tx + out_vy * ty
        return s, sin_phi

    def poincare_coordinates(
        self,
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        wtypes: NDArray[np.int64],
        widxs: NDArray[np.int64],
        out_vxs: NDArray[np.float64],
        out_vys: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Map arrays of bounce data to arrays of Poincare-section coordinates.

        Parameters
        ----------
        xs, ys : ndarray of float, shape (n,)
            Boundary hit-point coordinates.
        wtypes : ndarray of int, shape (n,)
            Wall type struck at each hit (``0`` segment, ``1`` arc).
        widxs : ndarray of int, shape (n,)
            Wall row index struck at each hit.
        out_vxs, out_vys : ndarray of float, shape (n,)
            Outgoing (post-reflection) unit velocity at each hit.

        Returns
        -------
        s : ndarray of float, shape (n,)
            Boundary arclength coordinate of each hit.
        sin_phi : ndarray of float, shape (n,)
            Sine of the reflection angle at each hit.
        """
        n = xs.shape[0]
        s = np.empty(n)
        sin_phi = np.empty(n)
        for i in range(n):
            s[i], sin_phi[i] = self.boundary_coordinate(xs[i], ys[i], int(wtypes[i]), int(widxs[i]), out_vxs[i], out_vys[i])
        return s, sin_phi


class _RayTracingBilliard(BilliardSystem, _WallBoundaryMixin):
    """Concrete ray-tracing behavior shared by every billiard shape below."""

    def simulate(self, pos: ArrayLike, vel: ArrayLike, n_bounces: int) -> dict[str, NDArray[Any]]:
        """Trace ``n_bounces`` specular reflections from an initial ray.

        Parameters
        ----------
        pos : array_like of float, shape (2,)
            Initial position ``(x, y)``; must lie in the billiard's interior.
        vel : array_like of float, shape (2,)
            Initial velocity direction ``(vx, vy)``; normalized internally.
        n_bounces : int
            Number of reflections to simulate.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``x``, ``y`` : ndarray of float, shape (n_bounces,) -- hit-point
              coordinates.
            - ``vx``, ``vy`` : ndarray of float, shape (n_bounces,) -- outgoing
              velocity components.
            - ``s`` : ndarray of float, shape (n_bounces,) -- boundary
              arclength coordinate at each hit.
            - ``sin_phi`` : ndarray of float, shape (n_bounces,) -- sine of
              the reflection angle at each hit.
            - ``wall_type`` : ndarray of int64, shape (n_bounces,) -- ``0``
              for a segment, ``1`` for an arc.
            - ``wall_idx`` : ndarray of int64, shape (n_bounces,) -- row index
              of the wall struck.
        """
        pos = np.asarray(pos, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64)
        vel = vel / float(np.linalg.norm(vel))
        xs, ys, vxs, vys, wtypes, widxs = simulate_bounces(
            pos[0],
            pos[1],
            vel[0],
            vel[1],
            self._segments,
            self._arcs,
            self._arc_full,
            n_bounces,
        )
        s, sin_phi = self.poincare_coordinates(xs, ys, wtypes, widxs, vxs, vys)
        return {
            "x": xs,
            "y": ys,
            "vx": vxs,
            "vy": vys,
            "s": s,
            "sin_phi": sin_phi,
            "wall_type": wtypes,
            "wall_idx": widxs,
        }

    def simulate_many_rays(self, pos: ArrayLike, angles: ArrayLike, n_bounces: int) -> dict[str, NDArray[Any]]:
        """Trace many independent rays from one point, in parallel.

        Same result as calling :meth:`simulate` once per angle in `angles`
        and pooling the outputs, but launches all rays in a single
        ``prange``-parallelized Numba kernel (:func:`simulate_bounces_many`)
        instead of a Python loop -- the batch entry point behind
        :func:`physicskit.chaos.visualizers.phase_space.plot_poincare_section`.

        Parameters
        ----------
        pos : array_like of float, shape (2,)
            Shared initial position ``(x, y)`` for every ray; must lie in the
            billiard's interior.
        angles : array_like of float, shape (n_rays,)
            Initial launch angle (radians) of each ray.
        n_bounces : int
            Number of reflections to simulate per ray.

        Returns
        -------
        dict
            Same keys as :meth:`simulate`, but each value is the
            concatenation (in ray order, then bounce order) of every ray's
            results, shape ``(n_rays * n_bounces,)``.
        """
        pos = np.asarray(pos, dtype=np.float64)
        angles = np.asarray(angles, dtype=np.float64)
        xs, ys, vxs, vys, wtypes, widxs = simulate_bounces_many(
            pos[0],
            pos[1],
            angles,
            self._segments,
            self._arcs,
            self._arc_full,
            n_bounces,
        )
        xs, ys, vxs, vys = xs.ravel(), ys.ravel(), vxs.ravel(), vys.ravel()
        wtypes, widxs = wtypes.ravel(), widxs.ravel()
        s, sin_phi = self.poincare_coordinates(xs, ys, wtypes, widxs, vxs, vys)
        return {
            "x": xs,
            "y": ys,
            "vx": vxs,
            "vy": vys,
            "s": s,
            "sin_phi": sin_phi,
            "wall_type": wtypes,
            "wall_idx": widxs,
        }

    def trajectory_segments(self, pos: ArrayLike, vel: ArrayLike, n_bounces: int) -> tuple[NDArray[np.float64], dict[str, NDArray[Any]]]:
        """Trace a trajectory and return its full real-space polyline.

        Parameters
        ----------
        pos : array_like of float, shape (2,)
            Initial position ``(x, y)``; must lie in the billiard's interior.
        vel : array_like of float, shape (2,)
            Initial velocity direction ``(vx, vy)``; normalized internally.
        n_bounces : int
            Number of reflections to simulate.

        Returns
        -------
        path : ndarray of float, shape (n_bounces + 1, 2)
            Positions visited, including the initial position as row 0.
        result : dict
            The same dictionary returned by :meth:`simulate`.
        """
        result = self.simulate(pos, vel, n_bounces)
        pos = np.asarray(pos, dtype=np.float64)
        xs = np.concatenate(([pos[0]], result["x"]))
        ys = np.concatenate(([pos[1]], result["y"]))
        return np.column_stack([xs, ys]), result

    def boundary_polyline(self, points_per_arc: int = 200) -> NDArray[np.float64]:
        """Trace the boundary as one or more closed polylines, suitable for plotting.

        Most billiards' boundaries are a single closed loop, but some (e.g.
        :class:`SinaiBilliard`, whose boundary is the outer square *plus* the
        separate, disjoint scatterer circle) have more than one closed
        component. A full-circle arc is never chained to a neighboring wall
        (it is already closed on its own), so it always starts a new
        component; every other wall is chained to its neighbors and closes
        back on itself once the chain returns to its own starting wall.

        Parameters
        ----------
        points_per_arc : int, default 200
            Number of points used to sample each circular arc.

        Returns
        -------
        ndarray of float, shape (M, 2)
            Points tracing the boundary in traversal (increasing-``s``)
            order, with each closed component's own first point repeated at
            its end. If the boundary has more than one closed component,
            they are separated by a row of ``NaN`` -- the standard
            Matplotlib/Plotly convention for a line break, so a single
            ``ax.plot(boundary[:, 0], boundary[:, 1])`` call still draws
            every component correctly, with no spurious connecting line
            between them.
        """
        walls: list[tuple[float, str, int]] = []
        for idx in range(self._segments.shape[0]):
            walls.append((self._s_offsets_seg[idx], "seg", idx))
        for idx in range(self._arcs.shape[0]):
            walls.append((self._s_offsets_arc[idx], "arc", idx))
        walls.sort(key=lambda w: w[0])

        components: list[list[Any]] = []
        current: list[Any] = []
        for _, kind, idx in walls:
            if kind == "seg":
                x1, y1 = self._segments[idx, 0], self._segments[idx, 1]
                current.append([x1, y1])
            elif self._arc_full[idx]:
                # A full circle is already closed and never connects to a
                # neighboring wall, so it is always its own component.
                if current:
                    components.append(current)
                    current = []
                cx, cy, r, theta1, _theta2 = self._arcs[idx]
                angles = theta1 + np.linspace(0.0, TWO_PI, points_per_arc, endpoint=False)
                components.append(np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)]).tolist())
            else:
                cx, cy, r, theta1, theta2 = self._arcs[idx]
                sweep = (theta2 - theta1) % TWO_PI
                angles = theta1 + np.linspace(0.0, sweep, points_per_arc, endpoint=False)
                current.extend(np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)]).tolist())
        if current:
            components.append(current)

        rows: list[Any] = []
        for i, component in enumerate(components):
            if i > 0:
                rows.append([np.nan, np.nan])
            rows.extend(component)
            rows.append(component[0])
        return np.asarray(rows)


# ---------------------------------------------------------------------------
# Concrete billiard shapes
# ---------------------------------------------------------------------------


class CircleBilliard(_RayTracingBilliard):
    """Integrable circular billiard centered at the origin.

    Parameters
    ----------
    radius : float, default 1.0
        Circle radius.

    Attributes
    ----------
    radius : float
        Circle radius.
    """

    def __init__(self, radius: float = 1.0):
        self.radius = float(radius)
        super().__init__()

    def _build_boundary(self) -> None:
        self._finalize_boundary([("arc", (0.0, 0.0, self.radius, 0.0, TWO_PI), True)])

    def sample_interior_point(self) -> NDArray[np.float64]:
        return np.array([0.0, 0.0])


class RectangleBilliard(_RayTracingBilliard):
    """Integrable rectangular billiard centered at the origin.

    Parameters
    ----------
    width : float, default 2.0
        Full extent along ``x``.
    height : float, default 1.0
        Full extent along ``y``.

    Attributes
    ----------
    width : float
        Full extent along ``x``.
    height : float
        Full extent along ``y``.
    """

    def __init__(self, width: float = 2.0, height: float = 1.0):
        self.width = float(width)
        self.height = float(height)
        super().__init__()

    def _build_boundary(self) -> None:
        w, h = self.width / 2.0, self.height / 2.0
        corners = [(-w, -h), (w, -h), (w, h), (-w, h)]
        walls: list[WallSpec] = [
            (
                "seg",
                (corners[i][0], corners[i][1], corners[(i + 1) % 4][0], corners[(i + 1) % 4][1]),
            )
            for i in range(4)
        ]
        self._finalize_boundary(walls)

    def sample_interior_point(self) -> NDArray[np.float64]:
        return np.array([0.0, 0.0])


class SinaiBilliard(_RayTracingBilliard):
    """Chaotic (defocusing) Sinai billiard.

    A square cell with a circular scatterer removed from its center.

    Parameters
    ----------
    cell_size : float, default 2.0
        Full side length of the square cell.
    scatterer_radius : float, default 0.5
        Radius of the central circular scatterer; must be smaller than half
        of `cell_size`.

    Attributes
    ----------
    cell_size : float
        Full side length of the square cell.
    scatterer_radius : float
        Radius of the central circular scatterer.

    Raises
    ------
    ValueError
        If `scatterer_radius` is not smaller than half of `cell_size`.
    """

    def __init__(self, cell_size: float = 2.0, scatterer_radius: float = 0.5):
        if scatterer_radius >= cell_size / 2.0:
            raise InvalidParameterError("scatterer_radius must be smaller than half the cell_size")
        self.cell_size = float(cell_size)
        self.scatterer_radius = float(scatterer_radius)
        super().__init__()

    def _build_boundary(self) -> None:
        h = self.cell_size / 2.0
        corners = [(-h, -h), (h, -h), (h, h), (-h, h)]
        walls: list[WallSpec] = [
            (
                "seg",
                (corners[i][0], corners[i][1], corners[(i + 1) % 4][0], corners[(i + 1) % 4][1]),
            )
            for i in range(4)
        ]
        walls.append(("arc", (0.0, 0.0, self.scatterer_radius, 0.0, TWO_PI), True))
        self._finalize_boundary(walls)

    def sample_interior_point(self) -> NDArray[np.float64]:
        h = self.cell_size / 2.0
        return np.array([(h + self.scatterer_radius) / 2.0, 0.0])


class BunimovichStadium(_RayTracingBilliard):
    """Chaotic (defocusing) Bunimovich stadium billiard.

    Two semicircles joined by straight edges.

    Parameters
    ----------
    radius : float, default 1.0
        Radius of the two semicircular end-caps.
    straight_length : float, default 2.0
        Length of the straight edges joining the semicircles.

    Attributes
    ----------
    radius : float
        Radius of the two semicircular end-caps.
    straight_length : float
        Length of the straight edges joining the semicircles.
    """

    def __init__(self, radius: float = 1.0, straight_length: float = 2.0):
        self.radius = float(radius)
        self.straight_length = float(straight_length)
        super().__init__()

    def _build_boundary(self) -> None:
        r, a = self.radius, self.straight_length / 2.0
        walls: list[WallSpec] = [
            ("seg", (-a, -r, a, -r)),  # bottom edge
            ("arc", (a, 0.0, r, -np.pi / 2.0, np.pi / 2.0), False),  # right semicircle
            ("seg", (a, r, -a, r)),  # top edge
            ("arc", (-a, 0.0, r, np.pi / 2.0, 3.0 * np.pi / 2.0), False),  # left semicircle
        ]
        self._finalize_boundary(walls)

    def sample_interior_point(self) -> NDArray[np.float64]:
        return np.array([0.0, 0.0])


class TruncatedCircleBilliard(_RayTracingBilliard):
    """A disk truncated by a straight chord.

    The region ``x <= radius - cut`` of the disk of the given `radius` is
    kept.

    Parameters
    ----------
    radius : float, default 1.0
        Disk radius.
    cut : float, default 0.3
        Distance the chord is cut in from the disk's edge; must satisfy
        ``0 < cut < radius``.

    Attributes
    ----------
    radius : float
        Disk radius.
    cut : float
        Distance the chord is cut in from the disk's edge.

    Raises
    ------
    ValueError
        If `cut` does not satisfy ``0 < cut < radius``.
    """

    def __init__(self, radius: float = 1.0, cut: float = 0.3):
        if not 0.0 < cut < radius:
            raise InvalidParameterError("cut must satisfy 0 < cut < radius")
        self.radius = float(radius)
        self.cut = float(cut)
        super().__init__()

    def _build_boundary(self) -> None:
        r = self.radius
        x_cut = r - self.cut
        y_cut = np.sqrt(r * r - x_cut * x_cut)
        theta_cut = np.arctan2(y_cut, x_cut)
        walls: list[WallSpec] = [
            # Major arc through pi, from theta_cut to (2*pi - theta_cut).
            ("arc", (0.0, 0.0, r, theta_cut, TWO_PI - theta_cut), False),
            # Chord closing the loop, continuing CCW.
            ("seg", (x_cut, -y_cut, x_cut, y_cut)),
        ]
        self._finalize_boundary(walls)

    def sample_interior_point(self) -> NDArray[np.float64]:
        return np.array([-self.radius / 2.0, 0.0])


class EllipseBilliard(_RayTracingBilliard):
    """Integrable elliptical billiard, centered at the origin.

    The ellipse billiard is integrable: every trajectory remains tangent to a
    single confocal caustic for all time -- either a confocal ellipse (for
    trajectories that never cross the segment joining the two foci) or a
    confocal hyperbola (for trajectories that do) -- so the Poincare section
    is foliated by smooth invariant curves, like the Circle and Rectangle
    billiards. Unlike those two, its boundary is not built from exact circular
    arcs, so it is represented as a fine closed polygon of `n_segments`
    straight edges sampled from the ellipse's parametric form; the resulting
    discretization error in the physics is negligible at the default
    resolution (perimeter error ``O(1/n_segments^2)``).

    Parameters
    ----------
    semi_major : float, default 1.5
        Semi-major axis, along ``x``.
    semi_minor : float, default 1.0
        Semi-minor axis, along ``y``; must be smaller than `semi_major`.
    n_segments : int, default 2000
        Number of straight edges used to approximate the smooth ellipse.

    Attributes
    ----------
    semi_major, semi_minor : float
        Ellipse semi-axes.
    n_segments : int
        Polygon-approximation resolution.

    Raises
    ------
    ValueError
        If `semi_major` or `semi_minor` is not positive, or `semi_major` does
        not exceed `semi_minor`.
    """

    def __init__(self, semi_major: float = 1.5, semi_minor: float = 1.0, n_segments: int = 2000):
        if semi_major <= 0.0 or semi_minor <= 0.0:
            raise InvalidParameterError("semi_major and semi_minor must be positive")
        if semi_major <= semi_minor:
            raise InvalidParameterError("semi_major must be greater than semi_minor")
        self.semi_major = float(semi_major)
        self.semi_minor = float(semi_minor)
        self.n_segments = int(n_segments)
        super().__init__()

    def _build_boundary(self) -> None:
        a, b, n = self.semi_major, self.semi_minor, self.n_segments
        theta = np.linspace(0.0, TWO_PI, n, endpoint=False)
        xs = a * np.cos(theta)
        ys = b * np.sin(theta)
        walls: list[WallSpec] = [("seg", (xs[i], ys[i], xs[(i + 1) % n], ys[(i + 1) % n])) for i in range(n)]
        self._finalize_boundary(walls)

    def sample_interior_point(self) -> NDArray[np.float64]:
        return np.array([0.0, 0.0])

    def foci(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The two foci of the ellipse, at ``(+-c, 0)`` with ``c = sqrt(a^2 - b^2)``.

        Returns
        -------
        f1, f2 : ndarray of float, shape (2,)
            The two focal points.
        """
        c = np.sqrt(self.semi_major**2 - self.semi_minor**2)
        return np.array([c, 0.0]), np.array([-c, 0.0])
