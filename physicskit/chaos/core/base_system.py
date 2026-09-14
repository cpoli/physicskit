"""Abstract base classes for dynamical systems, discrete maps, and billiards."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def get_init_params(obj: object) -> dict[str, Any]:
    """Read an object's constructor arguments back out of its own attributes.

    Reads each of ``type(obj).__init__``'s parameter names and looks up a
    same-named attribute on `obj` (the convention every concrete system in
    physicskit.chaos follows, e.g. ``self.sigma = float(sigma)``), skipping any
    parameter not stored under its own name. Used by :func:`_init_repr` for
    generic ``__repr__`` support and by
    :mod:`physicskit.chaos.utils.io` to serialize a system's configuration without
    each subclass writing its own serializer.

    Parameters
    ----------
    obj : object
        Instance to introspect.

    Returns
    -------
    dict
        Mapping of ``__init__`` parameter name to its current value on `obj`.
    """
    cls = type(obj)
    try:
        parameters = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return {}

    params = {}
    for name, param in parameters.items():
        if name == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if not hasattr(obj, name):
            continue
        params[name] = getattr(obj, name)
    return params


def _init_repr(obj: object) -> str:
    """Build an ``eval``-able-looking repr from an object's ``__init__`` signature.

    See :func:`get_init_params`; this just formats the result as
    ``ClassName(param=value, ...)``.
    """
    parts = []
    for name, value in get_init_params(obj).items():
        value_repr = np.array2string(value, separator=", ") if isinstance(value, np.ndarray) else repr(value)
        parts.append(f"{name}={value_repr}")
    return f"{type(obj).__name__}({', '.join(parts)})"


class DynamicalSystem(ABC):
    """Base class for continuous-time dynamical systems integrated as ODEs.

    Subclasses expose the right-hand side of ``dx/dt = f(x, t)`` both as a
    plain Python method (for convenience/plotting) and, where available, as a
    Numba-jitted module-level function usable with
    :mod:`physicskit.chaos.core.integrators`.
    """

    #: Dimension of the state vector ``x``. Set by each concrete subclass.
    dim: int

    @abstractmethod
    def rhs(self, state: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        """Evaluate the vector field at ``state`` and time ``t``.

        Parameters
        ----------
        state : ndarray of float, shape (dim,)
            Current state vector.
        t : float
            Current time.

        Returns
        -------
        ndarray of float, shape (dim,)
            The time derivative ``dx/dt`` evaluated at ``(state, t)``.
        """

    def initial_state(self) -> NDArray[np.float64]:
        """Return a reasonable default initial condition, if defined by the subclass.

        Returns
        -------
        ndarray of float, shape (dim,)
            A default initial state.

        Raises
        ------
        NotImplementedError
            If the subclass does not define a default initial condition.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return _init_repr(self)


class DiscreteMap(ABC):
    """Base class for discrete-time dynamical systems (iterated maps)."""

    #: Dimension of the state vector ``x``. Set by each concrete subclass.
    dim: int

    @abstractmethod
    def step(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Advance ``state`` by one iteration of the map.

        Parameters
        ----------
        state : ndarray of float, shape (dim,)
            Current state vector.

        Returns
        -------
        ndarray of float, shape (dim,)
            The next state vector.
        """

    def initial_state(self) -> NDArray[np.float64]:
        """Return a reasonable default initial condition, if defined by the subclass.

        Returns
        -------
        ndarray of float, shape (dim,)
            A default initial state.

        Raises
        ------
        NotImplementedError
            If the subclass does not define a default initial condition.
        """
        raise NotImplementedError

    def trajectory(self, state0: NDArray[np.float64] | None = None, n_iter: int = 1000) -> NDArray[np.float64]:
        """Iterate the map ``n_iter`` times starting from ``state0``.

        Parameters
        ----------
        state0 : array_like of float, shape (dim,), optional
            Initial state; defaults to :meth:`initial_state`.
        n_iter : int, default 1000
            Number of iterations to perform.

        Returns
        -------
        ndarray of float, shape (n_iter + 1, dim)
            The state at each iteration, including ``state0`` as row 0.
        """
        state0 = self.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
        states = np.empty((n_iter + 1, self.dim))
        states[0] = state0
        state = state0.copy()
        for i in range(n_iter):
            state = self.step(state)
            states[i + 1] = state
        return states

    def __repr__(self) -> str:
        return _init_repr(self)


class BilliardSystem(ABC):
    """Base class for 2D billiard geometries.

    A billiard is described as a closed boundary made of straight segments
    and circular arcs. Subclasses build this boundary representation once (in
    ``_build_boundary``) and this base class provides the shared,
    Numba-accelerated ray-tracing / specular-reflection machinery plus
    boundary-coordinate (arclength ``s``, sine of the reflection angle
    ``sin(phi)``) bookkeeping used for Poincare sections.

    Attributes
    ----------
    _segments : ndarray of float, shape (n_segments, 4)
        Straight boundary walls, each row ``(x1, y1, x2, y2)``.
    _arcs : ndarray of float, shape (n_arcs, 5)
        Circular-arc boundary walls, each row ``(cx, cy, r, theta1, theta2)``.
    _arc_full : ndarray of bool, shape (n_arcs,)
        Whether each arc is a full circle (``True``) or a bounded arc.
    _s_offsets_seg : ndarray of float, shape (n_segments,)
        Cumulative boundary arclength at the start of each segment.
    _s_offsets_arc : ndarray of float, shape (n_arcs,)
        Cumulative boundary arclength at the start of each arc.
    _perimeter : float
        Total boundary perimeter.
    """

    def __init__(self) -> None:
        self._segments: NDArray[np.float64]
        self._arcs: NDArray[np.float64]
        self._arc_full: NDArray[np.bool_]
        self._s_offsets_seg: NDArray[np.float64]
        self._s_offsets_arc: NDArray[np.float64]
        self._perimeter: float
        self._build_boundary()

    @abstractmethod
    def _build_boundary(self) -> None:
        """Populate the boundary arrays and cumulative arclength offsets.

        Sets ``self._segments``, ``self._arcs``, ``self._arc_full``,
        ``self._s_offsets_seg``, ``self._s_offsets_arc``, and
        ``self._perimeter``, typically by delegating to
        ``self._finalize_boundary``.
        """

    @abstractmethod
    def sample_interior_point(self) -> NDArray[np.float64]:
        """Return a point guaranteed to lie in the billiard's interior.

        Returns
        -------
        ndarray of float, shape (2,)
            An ``(x, y)`` point strictly inside the billiard.
        """

    def perimeter(self) -> float:
        """Total boundary perimeter.

        Returns
        -------
        float
            The billiard boundary's total arclength.
        """
        return self._perimeter

    def boundary_arrays(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Return the raw boundary wall arrays.

        Returns
        -------
        segments : ndarray of float, shape (n_segments, 4)
            Straight walls, each row ``(x1, y1, x2, y2)``.
        arcs : ndarray of float, shape (n_arcs, 5)
            Circular-arc walls, each row ``(cx, cy, r, theta1, theta2)``.
        arc_full : ndarray of bool, shape (n_arcs,)
            Whether each arc is a full circle.
        """
        return self._segments, self._arcs, self._arc_full

    @abstractmethod
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
            Dictionary with keys ``x``, ``y``, ``vx``, ``vy``, ``s``,
            ``sin_phi``, ``wall_type``, ``wall_idx``, each an array of
            length `n_bounces`; see the concrete implementation (shared by
            every billiard shape) for details.
        """

    @abstractmethod
    def simulate_many_rays(self, pos: ArrayLike, angles: ArrayLike, n_bounces: int) -> dict[str, NDArray[Any]]:
        """Trace many independent rays from one point, in parallel.

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

    @abstractmethod
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

    @abstractmethod
    def boundary_polyline(self, points_per_arc: int = 200) -> NDArray[np.float64]:
        """Trace the boundary as one or more closed polylines, suitable for plotting.

        Parameters
        ----------
        points_per_arc : int, default 200
            Number of points used to sample each circular arc.

        Returns
        -------
        ndarray of float, shape (M, 2)
            Points tracing the boundary; disjoint closed components (if any)
            are separated by a row of ``NaN``. See the concrete
            implementation (shared by every billiard shape) for details.
        """

    def __repr__(self) -> str:
        return _init_repr(self)
