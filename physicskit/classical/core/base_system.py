"""Abstract base classes for ODE, Lagrangian, and Hamiltonian systems.

These classes define the common interface that every physical system in
:mod:`physicskit.classical.systems` implements, and wire each system up to the
integrators in :mod:`physicskit.classical.core.integrators`. A :class:`SimulationResult`
is returned by every ``integrate`` call and is what the visualizers in
:mod:`physicskit.classical.visualizers` consume.

Notes
-----
The step/loop functions in :mod:`physicskit.classical.core.integrators` are themselves
``@njit``-compiled and call the supplied ``force_func``/``deriv_func``
*from inside nopython code*. Numba can only do this when that callback
is itself a genuine ``@njit`` dispatcher (numba's "first-class function"
support) -- a plain bound Python method (e.g. ``self.derivatives``)
cannot be typed and would fail. Every concrete system therefore builds
its own standalone njit callbacks (conventionally stored as
``self._force_njit`` and/or ``self._deriv_njit``, produced by a
module-level factory function that closes over the system's numeric
parameters) and the base classes below always integrate using *those*
attributes, never a bound method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from physicskit.classical.core import integrators as intg

__all__ = [
    "SimulationResult",
    "DynamicalSystem",
    "ODESystem",
    "HamiltonianSystem",
    "LagrangianSystem",
]


@dataclass
class SimulationResult:
    """Container for the output of a system's ``integrate`` call."""

    t: np.ndarray
    """ndarray, shape (n_steps + 1,): Time samples."""

    y: np.ndarray
    """ndarray, shape (n_steps + 1, state_dim): The raw stacked-state
    trajectory, in whatever layout the producing system uses internally."""

    q: Optional[np.ndarray] = None
    """ndarray, shape (n_steps + 1, ndof), optional: Generalized positions,
    for Hamiltonian/Lagrangian systems."""

    p: Optional[np.ndarray] = None
    """ndarray, shape (n_steps + 1, ndof), optional: Canonical momenta for
    Hamiltonian systems, or qdot for Lagrangian systems."""

    energy: Optional[np.ndarray] = None
    """ndarray, shape (n_steps + 1,), optional: Total mechanical energy
    H(t) (or T+V) at each sample."""

    method: str = ""
    """str: Name of the integrator used (e.g. ``"yoshida4"``)."""

    extra: dict = field(default_factory=dict)
    """dict: Free-form slot for any additional diagnostics a system
    chooses to attach."""


class DynamicalSystem(ABC):
    """Common base for any system with a state vector evolving in time."""

    def __init__(self, state0):
        self.state = np.asarray(state0, dtype=np.float64)
        self.t = 0.0

    @abstractmethod
    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """Return dstate/dt at ``(t, state)``.

        Parameters
        ----------
        t : float
            Current time.
        state : ndarray
            Current state vector.

        Returns
        -------
        ndarray
            Time derivative of ``state``.
        """

    @abstractmethod
    def energy(self, state: Optional[np.ndarray] = None) -> float:
        """Return the total mechanical energy of ``state``.

        Parameters
        ----------
        state : ndarray, optional
            State to evaluate; defaults to ``self.state``.

        Returns
        -------
        float
            Total mechanical energy.
        """

    def reset(self, state0=None, t0: float = 0.0):
        """Reset the system's state and clock.

        Parameters
        ----------
        state0 : array-like, optional
            New state; if omitted, the current state is kept.
        t0 : float, default 0.0
            New time.

        Returns
        -------
        ndarray
            The (possibly updated) current state.
        """
        if state0 is not None:
            self.state = np.asarray(state0, dtype=np.float64)
        self.t = t0
        return self.state

    @abstractmethod
    def integrate(self, t_span, dt: float, method: str = "rk4") -> SimulationResult:
        """Integrate the system forward in time.

        Parameters
        ----------
        t_span : tuple of float
            ``(t0, t1)``, start and end time.
        dt : float
            Fixed step size.
        method : str
            Integrator to use; valid values depend on the subclass.

        Returns
        -------
        SimulationResult
            The full trajectory and diagnostics.
        """


class ODESystem(DynamicalSystem):
    """A general (possibly non-conservative) first-order ODE system.

    Concrete subclasses must set ``self._deriv_njit`` (an ``@njit``
    dispatcher with signature ``(t, state) -> dstate``) in ``__init__``.
    Suitable for systems that are not naturally Hamiltonian/symplectic,
    e.g. central-force problems with drag, or rigid-body Euler equations
    (whose implicit-midpoint flow exactly conserves the quadratic energy
    and angular-momentum-squared invariants).
    """

    _deriv_njit = None

    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        return self._deriv_njit(t, state)

    def integrate(self, t_span, dt: float, method: str = "implicit_midpoint") -> SimulationResult:
        """Integrate the system forward in time.

        Parameters
        ----------
        t_span : tuple of float
            ``(t0, t1)``, start and end time.
        dt : float
            Fixed step size.
        method : {"implicit_midpoint", "rk4"}
            Integrator to use.

        Returns
        -------
        SimulationResult
            The full trajectory and energy diagnostics.
        """
        t0, t1 = t_span
        n_steps = int(round((t1 - t0) / dt))
        if method == "rk4":
            ts, ys = intg.rk4_integrate(self._deriv_njit, self.state, t0, n_steps, dt)
        elif method == "implicit_midpoint":
            ts, ys = intg.implicit_midpoint_integrate(self._deriv_njit, self.state, t0, n_steps, dt)
        else:
            raise ValueError(f"Unknown method '{method}' for ODESystem")
        energies = np.array([self.energy(y) for y in ys])
        self.state = ys[-1]
        self.t = ts[-1]
        return SimulationResult(t=ts, y=ys, energy=energies, method=method)


class HamiltonianSystem(DynamicalSystem):
    """Base class for canonical Hamiltonian systems H(q, p, t).

    Subclasses set ``ndof``, ``separable``, and ``mass_inv``, and build
    ``self._force_njit(q, t) -> array`` (required when ``separable`` is
    True, giving -dV/dq for H = T(p) + V(q)) and/or
    ``self._deriv_njit(t, y) -> dy`` for the full stacked state (required
    for non-separable systems, and optional otherwise -- if omitted it
    is built automatically from ``_force_njit``).

    Parameters
    ----------
    q0, p0 : array-like, shape (ndof,)
        Initial position and momentum.

    Notes
    -----
    ``state`` is stored flattened as ``[q_1..q_n, p_1..p_n]``.
    """

    ndof: int = 0
    """int: Number of degrees of freedom, inferred from ``q0``."""

    separable: bool = True
    """bool: Whether ``H = T(p) + V(q)``, enabling the Verlet/Yoshida4 integrators."""

    mass_inv = 1.0
    """float or ndarray: 1/m per coordinate; ``dq/dt = mass_inv * p``."""

    _force_njit = None
    _deriv_njit = None

    def __init__(self, q0, p0):
        q0 = np.asarray(q0, dtype=np.float64)
        p0 = np.asarray(p0, dtype=np.float64)
        self.ndof = q0.shape[0]
        super().__init__(np.concatenate([q0, p0]))

    @property
    def q(self) -> np.ndarray:
        """ndarray: Generalized positions (the first ``ndof`` entries of ``state``)."""
        return self.state[: self.ndof]

    @property
    def p(self) -> np.ndarray:
        """ndarray: Canonical momenta (the last ``ndof`` entries of ``state``)."""
        return self.state[self.ndof :]

    @staticmethod
    def split(state: np.ndarray, ndof: int):
        """Split a stacked ``[q, p]`` state vector into its two halves.

        Parameters
        ----------
        state : ndarray, shape (2 * ndof,)
        ndof : int

        Returns
        -------
        q, p : ndarray, shape (ndof,)
        """
        return state[:ndof], state[ndof:]

    @abstractmethod
    def kinetic_energy(self, p: np.ndarray) -> float:
        """Return T(p).

        Parameters
        ----------
        p : ndarray
            Canonical momenta.

        Returns
        -------
        float
        """
        ...

    @abstractmethod
    def potential_energy(self, q: np.ndarray) -> float:
        """Return V(q).

        Parameters
        ----------
        q : ndarray
            Generalized positions.

        Returns
        -------
        float
        """
        ...

    def force(self, q: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Return -dV/dq (only defined for separable systems).

        Parameters
        ----------
        q : ndarray
            Generalized positions.
        t : float, default 0.0
            Current time.

        Returns
        -------
        ndarray

        Raises
        ------
        NotImplementedError
            If the system is non-separable (no ``_force_njit`` set).
        """
        if self._force_njit is None:
            raise NotImplementedError("This system has no separable force() (non-separable Hamiltonian)")
        return self._force_njit(q, t)

    def _ensure_deriv_njit(self):
        if self._deriv_njit is None:
            if not self.separable:
                raise NotImplementedError("Non-separable systems must provide self._deriv_njit")
            self._deriv_njit = intg.make_separable_derivatives(self._force_njit, self.mass_inv, self.ndof)
        return self._deriv_njit

    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        return self._ensure_deriv_njit()(t, state)

    def energy(self, state: Optional[np.ndarray] = None) -> float:
        state = self.state if state is None else state
        q, p = self.split(state, self.ndof)
        return self.kinetic_energy(p) + self.potential_energy(q)

    def integrate(self, t_span, dt: float, method: str = "yoshida4") -> SimulationResult:
        """Integrate the system forward in time.

        Parameters
        ----------
        t_span : tuple of float
            ``(t0, t1)``, start and end time.
        dt : float
            Fixed step size.
        method : {"yoshida4", "verlet", "implicit_midpoint", "rk4"}
            Integrator to use. ``"yoshida4"``/``"verlet"`` require
            ``self.separable``.

        Returns
        -------
        SimulationResult
            The full ``(q, p)`` trajectory and energy diagnostics.
        """
        t0, t1 = t_span
        n_steps = int(round((t1 - t0) / dt))
        q0, p0 = self.q.copy(), self.p.copy()
        if method in ("verlet", "yoshida4"):
            if not self.separable:
                raise ValueError(f"method='{method}' requires a separable Hamiltonian")
            integ = intg.velocity_verlet_integrate if method == "verlet" else intg.yoshida4_integrate
            ts, qs, ps = integ(self._force_njit, self.mass_inv, q0, p0, t0, n_steps, dt)
        elif method in ("implicit_midpoint", "rk4"):
            y0 = np.concatenate([q0, p0])
            deriv = self._ensure_deriv_njit()
            integ = intg.implicit_midpoint_integrate if method == "implicit_midpoint" else intg.rk4_integrate
            ts, ys = integ(deriv, y0, t0, n_steps, dt)
            qs, ps = ys[:, : self.ndof], ys[:, self.ndof :]
        else:
            raise ValueError(f"Unknown method '{method}' for HamiltonianSystem")
        energies = np.array([self.kinetic_energy(p) + self.potential_energy(q) for q, p in zip(qs, ps)])
        self.state = np.concatenate([qs[-1], ps[-1]])
        self.t = ts[-1]
        return SimulationResult(t=ts, y=np.concatenate([qs, ps], axis=1), q=qs, p=ps, energy=energies, method=method)


class LagrangianSystem(DynamicalSystem):
    """Base class for systems defined via a Lagrangian L(q, qdot, t).

    Subclasses typically obtain their equations of motion from
    :class:`physicskit.classical.utils.symbolic.LagrangianEngine`, which derives both a
    direct acceleration function (for reporting/plotting in (q, qdot)
    coordinates, via ``self._accel_njit``) and a canonical Hamiltonian
    form (``self._canonical_deriv_njit``, used for symplectic
    integration via ``implicit_midpoint`` -- valid even when the mass
    matrix M(q) is configuration-dependent and the system is therefore
    non-separable, e.g. the double pendulum).

    Parameters
    ----------
    q0, qdot0 : array-like, shape (ndof,)
        Initial generalized position and velocity.

    Notes
    -----
    ``state`` is stored flattened as ``[q_1..q_n, qdot_1..qdot_n]``.
    """

    ndof: int = 0
    """int: Number of degrees of freedom, inferred from ``q0``."""

    _accel_njit = None
    _momentum_njit = None
    _canonical_deriv_njit = None  # (t, [q,p]) -> [dq, dp] = [dH/dp, -dH/dq]
    _velocity_njit = None  # (q, p) -> qdot = dH/dp, used to report qdot after integration
    _deriv_njit = None  # (t, [q,qdot]) -> [qdot, qddot], used only by method='rk4'

    def __init__(self, q0, qdot0):
        q0 = np.asarray(q0, dtype=np.float64)
        qdot0 = np.asarray(qdot0, dtype=np.float64)
        self.ndof = q0.shape[0]
        super().__init__(np.concatenate([q0, qdot0]))

    @property
    def q(self) -> np.ndarray:
        """ndarray: Generalized positions (the first ``ndof`` entries of ``state``)."""
        return self.state[: self.ndof]

    @property
    def qdot(self) -> np.ndarray:
        """ndarray: Generalized velocities (the last ``ndof`` entries of ``state``)."""
        return self.state[self.ndof :]

    @staticmethod
    def split(state: np.ndarray, ndof: int):
        """Split a stacked ``[q, qdot]`` state vector into its two halves.

        Parameters
        ----------
        state : ndarray, shape (2 * ndof,)
        ndof : int

        Returns
        -------
        q, qdot : ndarray, shape (ndof,)
        """
        return state[:ndof], state[ndof:]

    def acceleration(self, q: np.ndarray, qdot: np.ndarray, t: float) -> np.ndarray:
        """Return qddot at ``(q, qdot, t)`` via the Euler-Lagrange equations.

        Parameters
        ----------
        q, qdot : ndarray
        t : float

        Returns
        -------
        ndarray
        """
        return self._accel_njit(q, qdot, t)

    def momentum(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """Return the canonical momentum p = dL/dqdot at ``(q, qdot)``.

        Parameters
        ----------
        q, qdot : ndarray

        Returns
        -------
        ndarray
        """
        return self._momentum_njit(q, qdot)

    @abstractmethod
    def energy(self, state: Optional[np.ndarray] = None) -> float: ...

    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        q, qdot = self.split(state, self.ndof)
        return np.concatenate([qdot, self.acceleration(q, qdot, t)])

    def integrate(self, t_span, dt: float, method: str = "implicit_midpoint") -> SimulationResult:
        """Integrate the system forward in time.

        Parameters
        ----------
        t_span : tuple of float
            ``(t0, t1)``, start and end time.
        dt : float
            Fixed step size.
        method : {"implicit_midpoint", "rk4"}
            Integrator to use. ``"implicit_midpoint"`` (the default) is
            symplectic even for non-separable systems, via the
            Legendre-transformed canonical form; ``"rk4"`` integrates
            the direct ``(q, qdot)`` acceleration form and is not
            symplectic.

        Returns
        -------
        SimulationResult
            The full ``(q, qdot)`` trajectory and energy diagnostics
            (with ``result.p`` holding qdot, not a canonical momentum).
        """
        t0, t1 = t_span
        n_steps = int(round((t1 - t0) / dt))
        q0, qdot0 = self.q.copy(), self.qdot.copy()
        if method == "implicit_midpoint":
            p0 = self.momentum(q0, qdot0)
            y0 = np.concatenate([q0, p0])
            ts, ys = intg.implicit_midpoint_integrate(self._canonical_deriv_njit, y0, t0, n_steps, dt)
            qs = ys[:, : self.ndof]
            ps = ys[:, self.ndof :]
            qdots = np.array([self._velocity_njit(q, p) for q, p in zip(qs, ps)])
            ys_out = np.concatenate([qs, qdots], axis=1)
        elif method == "rk4":
            y0 = np.concatenate([q0, qdot0])
            ts, ys_out = intg.rk4_integrate(self._deriv_njit, y0, t0, n_steps, dt)
            qs = ys_out[:, : self.ndof]
            qdots = ys_out[:, self.ndof :]
        else:
            raise ValueError(f"Unknown method '{method}' for LagrangianSystem")
        energies = np.array([self.energy(np.concatenate([q, qd])) for q, qd in zip(qs, qdots)])
        self.state = ys_out[-1]
        self.t = ts[-1]
        return SimulationResult(t=ts, y=ys_out, q=qs, p=qdots, energy=energies, method=method)
