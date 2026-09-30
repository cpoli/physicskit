"""A shared, unit-aware container for simulation output.

Subpackages return results in several shapes: a ``(times, states)`` tuple
from the shared :mod:`physicskit.integrators` and the chaos flows, a
``(times, positions, velocities)`` tuple from the symplectic integrators,
the classical :class:`~physicskit.classical.core.base_system.SimulationResult`,
an orbit array from a chaos map, a quantum
:class:`~physicskit.quantum.core.eigensolvers.EigenResult`, and an rmt
:class:`~physicskit.rmt.spectrum.Spectrum`. :class:`Result` is a single
container they can all be converted to, which :mod:`physicskit.io` can
save and load and which knows its units.

The container is opt-in. Subpackages keep returning their own types, and
the ``from_*`` adapters below convert them. The adapters read attributes
by name and do not import the subpackages.

Examples
--------
>>> import numpy as np
>>> from physicskit.results import Result
>>> from physicskit.units import geometrized_units
>>> sun = geometrized_units(mass_kg=1.98847e30)
>>> r = Result(np.linspace(0.0, 10.0, 11), np.zeros((11, 2)), units={"times": "time"}, unit_system=sun)
>>> round(float(r.to_si().times[-1]) * 1e6, 2)  # 10 M in microseconds
49.26
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physicskit.units import DIMENSIONS, SI, UnitSystem

__all__ = [
    "Result",
    "from_integrator",
    "from_simulation_result",
    "from_map_orbit",
    "from_eigen_result",
    "from_spectrum",
]

_RESERVED = ("times", "states")


@dataclass(eq=False)
class Result:
    """Time series or other array output, with metadata and units.

    Parameters
    ----------
    times : array_like or None
        Sample times, shape ``(n,)``, or None for output with no time
        axis (an eigenvalue spectrum, say).
    states : array_like
        The main output array. When ``times`` is given, its first axis
        must have length ``n``.
    metadata : dict, optional
        JSON-compatible information such as the integrator name,
        parameters and provenance. It must be JSON-compatible for
        :func:`physicskit.io.save` to store it.
    units : dict of str to str, optional
        Maps an array name (``"times"``, ``"states"`` or a key of
        ``arrays``) to a dimension name from
        :data:`physicskit.units.DIMENSIONS`. Arrays without an entry are
        treated as dimensionless or of unknown dimension, and
        :meth:`convert` leaves them unchanged.
    unit_system : UnitSystem, optional
        The system the dimensioned arrays are expressed in. None means
        unspecified; :data:`physicskit.units.SI` means SI.
    arrays : dict of str to array_like, optional
        Further named arrays, such as momenta, energies or a spatial grid.

    Raises
    ------
    ValueError
        If lengths disagree, an array name is reserved or invalid, or a
        ``units`` entry names an unknown array or dimension.

    Examples
    --------
    >>> r = Result([0.0, 0.5, 1.0], [[1.0], [0.5], [0.0]], arrays={"energy": [2.0, 2.0, 2.0]})
    >>> r.names
    ['times', 'states', 'energy']
    >>> r["energy"].shape
    (3,)
    """

    times: NDArray[Any] | None
    states: NDArray[Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    unit_system: UnitSystem | None = None
    arrays: dict[str, NDArray[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.states = np.asarray(self.states)
        if self.times is not None:
            self.times = np.asarray(self.times)
            if self.times.ndim != 1:
                raise ValueError(f"times must be 1-D, got shape {self.times.shape}")
            if self.states.ndim == 0 or self.states.shape[0] != self.times.shape[0]:
                raise ValueError(f"states must have a first axis of length len(times)={self.times.shape[0]}, got shape {self.states.shape}")
        self.arrays = {str(k): np.asarray(v) for k, v in self.arrays.items()}
        for name in self.arrays:
            if name in _RESERVED:
                raise ValueError(f"array name {name!r} is reserved")
            if not name or "/" in name:
                raise ValueError(f"array name {name!r} must be non-empty and contain no '/'")
        self.metadata = dict(self.metadata)
        self.units = {str(k): str(v) for k, v in self.units.items()}
        for name, dim in self.units.items():
            if name not in self.names:
                raise ValueError(f"units refers to unknown array {name!r}; available: {self.names}")
            if dim not in DIMENSIONS:
                raise ValueError(f"unknown dimension {dim!r} for {name!r}; use one of {sorted(DIMENSIONS)}")

    @property
    def names(self) -> list[str]:
        """Names of all arrays held, in the order times, states, extras.

        Returns
        -------
        list of str
        """
        head = ["times"] if self.times is not None else []
        return [*head, "states", *self.arrays]

    def __getitem__(self, name: str) -> NDArray[Any]:
        if name == "times" and self.times is not None:
            return self.times
        if name == "states":
            return self.states
        if name in self.arrays:
            return self.arrays[name]
        raise KeyError(name)

    def __repr__(self) -> str:
        shapes = ", ".join(f"{n}={self[n].shape}" for n in self.names)
        system = self.unit_system.name if self.unit_system is not None else None
        return f"Result({shapes}, unit_system={system!r})"

    def convert(self, system: UnitSystem) -> Result:
        """Return a copy with every dimensioned array expressed in ``system``.

        Arrays listed in ``units`` are converted through SI; all other
        arrays and the metadata are copied unchanged.

        Parameters
        ----------
        system : UnitSystem
            Target unit system.

        Returns
        -------
        Result

        Raises
        ------
        ValueError
            If ``units`` is non-empty but ``unit_system`` is None, or
            either system leaves a needed dimension undetermined.

        Examples
        --------
        >>> from physicskit.units import SI, statphys_units
        >>> r = Result(None, [1.0, 2.0], units={"states": "temperature"}, unit_system=statphys_units(temperature_k=300.0))
        >>> r.convert(SI).states
        array([300., 600.])
        """
        if self.units and self.unit_system is None:
            raise ValueError("this Result has dimensioned arrays but no unit_system to convert from")
        source = self.unit_system

        def conv(name: str, arr: NDArray[Any]) -> NDArray[Any]:
            if name not in self.units or source is None:
                return arr.copy()
            dim = self.units[name]
            return np.asarray(system.from_si(source.to_si(arr, dim), dim))

        return Result(
            times=None if self.times is None else conv("times", self.times),
            states=conv("states", self.states),
            metadata=dict(self.metadata),
            units=dict(self.units),
            unit_system=system,
            arrays={k: conv(k, v) for k, v in self.arrays.items()},
        )

    def to_si(self) -> Result:
        """Return a copy with every dimensioned array in SI units.

        Shorthand for ``convert(physicskit.units.SI)``.

        Returns
        -------
        Result
        """
        return self.convert(SI)


def _build(
    times: ArrayLike | None,
    states: ArrayLike,
    source: str,
    arrays: Mapping[str, ArrayLike],
    metadata: Mapping[str, Any] | None,
    units: Mapping[str, str] | None,
    unit_system: UnitSystem | None,
    base_meta: Mapping[str, Any],
) -> Result:
    meta: dict[str, Any] = {"source": source, **base_meta}
    meta.update(metadata or {})
    return Result(
        times=None if times is None else np.asarray(times),
        states=np.asarray(states),
        metadata=meta,
        units=dict(units or {}),
        unit_system=unit_system,
        arrays={k: np.asarray(v) for k, v in arrays.items()},
    )


def from_integrator(
    output: Sequence[ArrayLike],
    *,
    method: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    units: Mapping[str, str] | None = None,
    unit_system: UnitSystem | None = None,
) -> Result:
    """Wrap the tuple returned by a :mod:`physicskit.integrators` function.

    Accepts ``(times, states)`` from :func:`~physicskit.integrators.rk4_integrate`,
    :func:`~physicskit.integrators.dopri5_integrate` and the chaos flows'
    ``trajectory()`` methods, or ``(times, positions, velocities)`` from
    :func:`~physicskit.integrators.leapfrog_integrate` and
    :func:`~physicskit.integrators.yoshida4_integrate`. For the latter,
    ``states`` is the phase-space trajectory ``[positions, velocities]``
    stacked along the last axis, and the two halves are also kept as
    ``arrays["positions"]`` and ``arrays["velocities"]``.

    Parameters
    ----------
    output : tuple of ndarray
        The integrator's return value.
    method : str, optional
        Integrator name, stored as ``metadata["method"]``.
    metadata : dict, optional
        Extra metadata, merged over the adapter's own.
    units : dict of str to str, optional
        See :class:`Result`.
    unit_system : UnitSystem, optional
        See :class:`Result`.

    Returns
    -------
    Result

    Examples
    --------
    >>> import numpy as np
    >>> r = from_integrator((np.array([0.0, 0.1]), np.ones((2, 1)), np.zeros((2, 1))), method="leapfrog")
    >>> r.states.shape, r.metadata["method"]
    ((2, 2), 'leapfrog')
    """
    parts = tuple(output)
    base: dict[str, Any] = {} if method is None else {"method": method}
    if len(parts) == 2:
        times, states = parts
        return _build(times, states, "physicskit.integrators", {}, metadata, units, unit_system, base)
    if len(parts) == 3:
        times, pos, vel = (np.asarray(p) for p in parts)
        states = np.concatenate([pos, vel], axis=-1)
        return _build(
            times,
            states,
            "physicskit.integrators",
            {"positions": pos, "velocities": vel},
            metadata,
            units,
            unit_system,
            {**base, "state_layout": "[positions, velocities]"},
        )
    raise ValueError(f"expected a (times, states) or (times, positions, velocities) tuple, got {len(parts)} elements")


def _split_extra(extra: Mapping[str, Any]) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    arrays: dict[str, NDArray[Any]] = {}
    scalars: dict[str, Any] = {}
    for key, value in extra.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        elif isinstance(value, np.generic):
            scalars[key] = value.item()
        elif value is None or isinstance(value, (bool, int, float, str)):
            scalars[key] = value
        else:
            scalars[key] = repr(value)
    return arrays, scalars


def from_simulation_result(
    result: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    units: Mapping[str, str] | None = None,
    unit_system: UnitSystem | None = None,
) -> Result:
    """Wrap a classical :class:`~physicskit.classical.core.base_system.SimulationResult`.

    ``t`` becomes ``times`` and ``y`` becomes ``states``. ``q``, ``p`` and
    ``energy``, when present, go to ``arrays``. Entries of ``extra`` go to
    ``arrays`` if they are arrays and to ``metadata["extra"]`` otherwise;
    values that are not JSON scalars are stored as their ``repr``.

    Parameters
    ----------
    result : SimulationResult
        Output of a classical system's ``integrate()``.
    metadata, units, unit_system
        See :func:`from_integrator`.

    Returns
    -------
    Result

    Examples
    --------
    >>> from physicskit.classical.systems.hamiltonian import HenonHeilesSystem
    >>> sim = HenonHeilesSystem(q0=[0.1, 0.0], p0=[0.0, 0.2]).integrate((0.0, 1.0), dt=0.01)
    >>> r = from_simulation_result(sim)
    >>> r.names, r.metadata["method"]
    (['times', 'states', 'q', 'p', 'energy'], 'yoshida4')
    """
    arrays = {name: getattr(result, name) for name in ("q", "p", "energy") if getattr(result, name, None) is not None}
    extra_arrays, extra_scalars = _split_extra(getattr(result, "extra", {}) or {})
    clash = set(extra_arrays) & (set(arrays) | set(_RESERVED))
    arrays.update({(f"extra_{k}" if k in clash else k): v for k, v in extra_arrays.items()})
    base: dict[str, Any] = {"method": getattr(result, "method", "")}
    if extra_scalars:
        base["extra"] = extra_scalars
    return _build(result.t, result.y, "physicskit.classical.SimulationResult", arrays, metadata, units, unit_system, base)


def from_map_orbit(
    orbit: ArrayLike,
    *,
    metadata: Mapping[str, Any] | None = None,
    units: Mapping[str, str] | None = None,
    unit_system: UnitSystem | None = None,
) -> Result:
    """Wrap the orbit returned by a chaos map's ``trajectory()``.

    Maps are discrete, so ``times`` is the iteration index ``0, 1, ...``
    and ``metadata["time_kind"]`` is ``"iteration"``.

    Parameters
    ----------
    orbit : array_like, shape (n_iter + 1, dim)
        The map's orbit.
    metadata, units, unit_system
        See :func:`from_integrator`.

    Returns
    -------
    Result

    Examples
    --------
    >>> from physicskit.chaos.systems.maps import LogisticMap
    >>> r = from_map_orbit(LogisticMap(r=3.9).trajectory(n_iter=5))
    >>> r.times
    array([0, 1, 2, 3, 4, 5])
    """
    states = np.asarray(orbit)
    return _build(np.arange(states.shape[0]), states, "physicskit.chaos map", {}, metadata, units, unit_system, {"time_kind": "iteration"})


def from_eigen_result(
    result: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    units: Mapping[str, str] | None = None,
    unit_system: UnitSystem | None = None,
) -> Result:
    """Wrap a quantum :class:`~physicskit.quantum.core.eigensolvers.EigenResult`.

    There is no time axis. ``states`` holds the wavefunctions, shape
    ``(n_states, len(x))``, and ``arrays`` holds ``x`` and ``energies``.

    Parameters
    ----------
    result : EigenResult
        Output of :meth:`~physicskit.quantum.core.eigensolvers.NumerovSolver.solve`.
    metadata, units, unit_system
        See :func:`from_integrator`.

    Returns
    -------
    Result

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.quantum.core.eigensolvers import NumerovSolver, infinite_well
    >>> eig = NumerovSolver(np.linspace(0.0, 1.0, 201), infinite_well()).solve(n_states=3)
    >>> r = from_eigen_result(eig)
    >>> r.times is None, r.states.shape, r["energies"].shape
    (True, (3, 201), (3,))
    """
    return _build(
        None,
        result.wavefunctions,
        "physicskit.quantum.EigenResult",
        {"x": result.x, "energies": result.energies},
        metadata,
        units,
        unit_system,
        {},
    )


def from_spectrum(
    spectrum: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Result:
    """Wrap an rmt :class:`~physicskit.rmt.spectrum.Spectrum`.

    There is no time axis. ``states`` holds the raw eigenvalues, shape
    ``(n_samples, n)``. ``n``, ``beta``, ``ensemble`` and ``scale`` go to
    ``metadata``, and ``eigenvectors`` (if sampled) to ``arrays``.
    Random-matrix spectra are dimensionless, so no units are attached.

    Parameters
    ----------
    spectrum : Spectrum
        Output of an ensemble's ``sample()``.
    metadata : dict, optional
        Extra metadata, merged over the adapter's own.

    Returns
    -------
    Result

    Examples
    --------
    >>> import physicskit.rmt as rmt
    >>> r = from_spectrum(rmt.ensembles.GOE(n=4, seed=1).sample(n_samples=3))
    >>> r.states.shape, r.metadata["ensemble"], r.metadata["beta"]
    ((3, 4), 'GOE', 1.0)
    """
    arrays = {} if getattr(spectrum, "eigenvectors", None) is None else {"eigenvectors": spectrum.eigenvectors}
    beta = spectrum.beta
    base = {
        "n": int(spectrum.n),
        "beta": None if beta is None else float(beta),
        "ensemble": str(spectrum.ensemble),
        "scale": float(spectrum.scale),
    }
    return _build(None, spectrum.eigenvalues, "physicskit.rmt.Spectrum", arrays, metadata, None, None, base)
