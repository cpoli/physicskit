r"""Natural-unit systems and conversions to and from SI.

Each physicskit subpackage computes in the unit system natural to its own
field: :mod:`physicskit.astro` sets :math:`G = 1`,
:mod:`physicskit.relativity` sets :math:`G = c = 1`,
:mod:`physicskit.quantum` sets :math:`\hbar = 1`, and
:mod:`physicskit.statphys` sets :math:`k_B = 1`. Setting a constant to 1
alone does not fix a unit system: :math:`G = c = 1` still leaves one scale
free (conventionally a mass :math:`M`), and :math:`G = 1` leaves two. This
module makes those remaining choices explicit. Every preset has keyword
arguments you must supply for the free scales; none of them has a hidden
default.

How it works
------------
A :class:`UnitSystem` is defined by a set of *anchors*: pairs of a
:class:`Dimension` and the SI value of one unit of that dimension. A
constant set to 1 is an anchor (for :math:`c = 1`, one unit of velocity is
:math:`c_{SI}` m/s), and so is each user-chosen scale (one unit of mass is
:math:`M_\odot`). Writing the unknown SI sizes of the base units as
:math:`[L], [M], [T], [\Theta]`, each anchor with dimension
:math:`L^a M^b T^c \Theta^d` and SI value :math:`v` gives one linear
equation in log space,

.. math::

   a \ln[L] + b \ln[M] + c \ln[T] + d \ln[\Theta] = \ln v .

A dimension is convertible exactly when its exponent vector lies in the
span of the anchors' exponent vectors. So :func:`statphys_units` with only
an energy scale can convert energies and temperatures but refuses to
convert a length, rather than returning a meaningless number.

SI values come from :mod:`physicskit.constants`. Converting to and from
:mod:`pint` quantities is supported when pint is installed
(``pip install "physicskit[units]"``); see :func:`to_pint` and
:func:`from_pint`.

Examples
--------
The geometrized (:math:`G = c = 1`) length of one solar mass is
:math:`GM_\odot/c^2 \approx 1476.6` m (to the precision of
:data:`physicskit.constants.SOLAR_MASS_KG`):

>>> import physicskit.constants as const
>>> from physicskit.units import geometrized_units
>>> sun = geometrized_units(mass_kg=const.SOLAR_MASS_KG)
>>> abs(sun.scale("length") - 1476.6) < 0.1
True
>>> round(sun.to_si(10.0, "time") * 1e6, 2)  # 10 M in microseconds
49.26
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

import physicskit.constants as const

__all__ = [
    "Dimension",
    "DIMENSIONS",
    "NATURAL_CONSTANTS",
    "SUBPACKAGE_CONVENTIONS",
    "UnitSystem",
    "SI",
    "natural_units",
    "astro_units",
    "geometrized_units",
    "quantum_units",
    "statphys_units",
    "to_si",
    "from_si",
    "to_pint",
    "from_pint",
]

# Relative tolerance used when checking that redundant anchors agree, in
# log space (so ~1e-9 relative disagreement between SI values).
_CONSISTENCY_RTOL = 1e-9


@dataclass(frozen=True)
class Dimension:
    r"""Physical dimension :math:`L^a M^b T^c \Theta^d` as an exponent vector.

    Dimensions multiply, divide and raise to powers like the quantities
    they describe, so derived dimensions can be built from base ones.

    Parameters
    ----------
    length, mass, time, temperature : float, default 0
        Exponents of length, mass, time and temperature.

    Examples
    --------
    >>> L, M, T = Dimension(length=1), Dimension(mass=1), Dimension(time=1)
    >>> M * L**2 / T**2 == DIMENSIONS["energy"]
    True
    """

    length: float = 0.0
    mass: float = 0.0
    time: float = 0.0
    temperature: float = 0.0

    def as_vector(self) -> NDArray[np.float64]:
        """Return the exponents as a ``(4,)`` array ``(length, mass, time, temperature)``.

        Returns
        -------
        ndarray of float, shape (4,)
        """
        return np.array([self.length, self.mass, self.time, self.temperature], dtype=np.float64)

    def __mul__(self, other: Dimension) -> Dimension:
        return Dimension(*(self.as_vector() + other.as_vector()).tolist())

    def __truediv__(self, other: Dimension) -> Dimension:
        return Dimension(*(self.as_vector() - other.as_vector()).tolist())

    def __pow__(self, power: float) -> Dimension:
        return Dimension(*(self.as_vector() * power).tolist())

    def __str__(self) -> str:
        parts = [f"{sym}^{exp:g}" for sym, exp in zip("LMTΘ", self.as_vector()) if exp != 0]
        return " ".join(parts) if parts else "1"


_L = Dimension(length=1)
_M = Dimension(mass=1)
_T = Dimension(time=1)
_K = Dimension(temperature=1)

#: Named dimensions accepted wherever a ``dimension`` argument is taken.
DIMENSIONS: dict[str, Dimension] = {
    "dimensionless": Dimension(),
    "length": _L,
    "mass": _M,
    "time": _T,
    "temperature": _K,
    "area": _L**2,
    "volume": _L**3,
    "frequency": _T**-1,
    "velocity": _L / _T,
    "acceleration": _L / _T**2,
    "momentum": _M * _L / _T,
    "force": _M * _L / _T**2,
    "energy": _M * _L**2 / _T**2,
    "power": _M * _L**2 / _T**3,
    "action": _M * _L**2 / _T,
    "angular_momentum": _M * _L**2 / _T,
    "density": _M / _L**3,
    "number_density": _L**-3,
    "pressure": _M / (_L * _T**2),
    "entropy": _M * _L**2 / (_T**2 * _K),
}

#: Constants that a natural-unit system can set to 1, as
#: ``name -> (dimension, SI value)``. SI values are from
#: :mod:`physicskit.constants` (CODATA values via :mod:`scipy.constants`).
NATURAL_CONSTANTS: dict[str, tuple[Dimension, float]] = {
    "G": (_L**3 / (_M * _T**2), const.G),
    "c": (_L / _T, const.C),
    "hbar": (_M * _L**2 / _T, const.HBAR),
    "k_B": (_M * _L**2 / (_T**2 * _K), const.K_B),
}

#: Which constants each subpackage sets to 1, and the preset that builds a
#: matching :class:`UnitSystem` once the remaining free scales are chosen.
SUBPACKAGE_CONVENTIONS: dict[str, dict[str, Any]] = {
    "astro": {"constants": ("G",), "preset": "astro_units"},
    "relativity": {"constants": ("G", "c"), "preset": "geometrized_units"},
    "quantum": {"constants": ("hbar",), "preset": "quantum_units"},
    "statphys": {"constants": ("k_B",), "preset": "statphys_units"},
}

# Keyword name -> dimension, for the user-chosen scale arguments.
_SCALE_KWARGS: dict[str, Dimension] = {
    "length_m": DIMENSIONS["length"],
    "mass_kg": DIMENSIONS["mass"],
    "time_s": DIMENSIONS["time"],
    "energy_j": DIMENSIONS["energy"],
    "temperature_k": DIMENSIONS["temperature"],
}


def _as_dimension(dimension: str | Dimension) -> Dimension:
    if isinstance(dimension, Dimension):
        return dimension
    try:
        return DIMENSIONS[dimension]
    except KeyError:
        raise ValueError(f"unknown dimension {dimension!r}; use a Dimension or one of {sorted(DIMENSIONS)}") from None


@dataclass(frozen=True)
class UnitSystem:
    r"""A system of units fixed by a set of anchors.

    Most users should build one with a preset (:func:`astro_units`,
    :func:`geometrized_units`, :func:`quantum_units`,
    :func:`statphys_units`) or with :func:`natural_units`.

    Parameters
    ----------
    name : str
        Human-readable label, e.g. ``"relativity (G=c=1)"``.
    anchors : tuple of (Dimension, float)
        Each anchor states that one unit of ``dimension`` in this system
        equals ``value`` in SI. The anchors must be mutually consistent.

    Raises
    ------
    ValueError
        If an anchor value is not a positive finite number, or redundant
        anchors disagree.

    Examples
    --------
    A plain "centimetre-gram-second" mechanical system:

    >>> cgs = UnitSystem("cgs", ((Dimension(length=1), 1e-2), (Dimension(mass=1), 1e-3), (Dimension(time=1), 1.0)))
    >>> round(cgs.to_si(1.0, "energy"), 15)  # 1 erg in J
    1e-07
    """

    name: str
    anchors: tuple[tuple[Dimension, float], ...]

    def __post_init__(self) -> None:
        for dim, value in self.anchors:
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"anchor for dimension {dim} must be a positive finite SI value, got {value!r}")
        if self.anchors:
            A, b = self._system()
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
            residual = np.max(np.abs(A @ x - b))
            if residual > _CONSISTENCY_RTOL * max(1.0, float(np.max(np.abs(b)))):
                raise ValueError(f"unit system {self.name!r} is overdetermined: its anchors are mutually inconsistent")

    def _system(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        A = np.array([dim.as_vector() for dim, _ in self.anchors], dtype=np.float64).reshape(-1, 4)
        b = np.log(np.array([value for _, value in self.anchors], dtype=np.float64))
        return A, b

    def is_determined(self, dimension: str | Dimension) -> bool:
        """Whether the anchors fix the SI size of one unit of ``dimension``.

        Parameters
        ----------
        dimension : str or Dimension
            A key of :data:`DIMENSIONS` or a :class:`Dimension`.

        Returns
        -------
        bool

        Examples
        --------
        >>> u = statphys_units(energy_j=1.0e-21)
        >>> u.is_determined("temperature"), u.is_determined("length")
        (True, False)
        """
        d = _as_dimension(dimension).as_vector()
        if not d.any():
            return True
        if not self.anchors:
            return False
        A, _ = self._system()
        return bool(np.linalg.matrix_rank(np.vstack([A, d])) == np.linalg.matrix_rank(A))

    def scale(self, dimension: str | Dimension) -> float:
        """SI value of one unit of ``dimension`` in this system.

        Parameters
        ----------
        dimension : str or Dimension
            A key of :data:`DIMENSIONS` or a :class:`Dimension`.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the system's anchors do not determine this dimension.

        Examples
        --------
        >>> u = quantum_units(mass_kg=const.ELECTRON_MASS, length_m=5.29177210903e-11)
        >>> print(f"{u.scale('time'):.4e}")  # atomic unit of time, s
        2.4189e-17
        """
        dim = _as_dimension(dimension)
        if not self.is_determined(dim):
            raise ValueError(
                f"unit system {self.name!r} does not fix the scale of dimension {dim}; supply another scale (e.g. length_m or mass_kg) when building it"
            )
        if not dim.as_vector().any():
            return 1.0
        A, b = self._system()
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return float(np.exp(dim.as_vector() @ x))

    def to_si(self, value: ArrayLike, dimension: str | Dimension) -> Any:
        """Convert ``value`` from this system to SI.

        Parameters
        ----------
        value : float or array_like
            Quantity expressed in this system's units.
        dimension : str or Dimension
            The quantity's dimension.

        Returns
        -------
        float or ndarray
            The same quantity in SI units.

        Examples
        --------
        >>> u = astro_units(length_m=const.ASTRONOMICAL_UNIT_M, mass_kg=const.SOLAR_MASS_KG)
        >>> round(u.to_si(2 * np.pi, "time") / 86400 / 365.25, 4)  # Earth's orbit in years
        1.0
        """
        factor = self.scale(dimension)
        if np.isscalar(value):
            return float(value) * factor  # type: ignore[arg-type]
        return np.asarray(value) * factor

    def from_si(self, value: ArrayLike, dimension: str | Dimension) -> Any:
        """Convert ``value`` from SI to this system.

        Parameters
        ----------
        value : float or array_like
            Quantity in SI units.
        dimension : str or Dimension
            The quantity's dimension.

        Returns
        -------
        float or ndarray
            The same quantity in this system's units.

        Examples
        --------
        >>> u = geometrized_units(mass_kg=const.SOLAR_MASS_KG)
        >>> round(u.from_si(const.C, "velocity"), 12)
        1.0
        """
        factor = self.scale(dimension)
        if np.isscalar(value):
            return float(value) / factor  # type: ignore[arg-type]
        return np.asarray(value) / factor

    def scales(self) -> dict[str, float]:
        """SI value of one unit for every named dimension this system fixes.

        Returns
        -------
        dict of str to float
            Keys are the entries of :data:`DIMENSIONS` that are determined.

        Examples
        --------
        >>> sorted(statphys_units(temperature_k=300.0).scales())
        ['dimensionless', 'energy', 'entropy', 'temperature']
        """
        return {name: self.scale(dim) for name, dim in DIMENSIONS.items() if self.is_determined(dim)}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (inverse of :meth:`from_dict`).

        Returns
        -------
        dict
        """
        return {
            "name": self.name,
            "anchors": [[dim.as_vector().tolist(), value] for dim, value in self.anchors],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UnitSystem:
        """Rebuild a :class:`UnitSystem` from :meth:`to_dict` output.

        Parameters
        ----------
        data : mapping
            Output of :meth:`to_dict`.

        Returns
        -------
        UnitSystem

        Examples
        --------
        >>> u = geometrized_units(mass_kg=const.SOLAR_MASS_KG)
        >>> UnitSystem.from_dict(u.to_dict()) == u
        True
        """
        anchors = tuple((Dimension(*map(float, vec)), float(value)) for vec, value in data["anchors"])
        return cls(str(data["name"]), anchors)


#: The SI system itself: one unit of every dimension is its SI unit.
#: Converting to it is the identity.
SI = UnitSystem("SI", ((_L, 1.0), (_M, 1.0), (_T, 1.0), (_K, 1.0)))


def natural_units(
    *constants: str,
    name: str | None = None,
    length_m: float | None = None,
    mass_kg: float | None = None,
    time_s: float | None = None,
    energy_j: float | None = None,
    temperature_k: float | None = None,
) -> UnitSystem:
    r"""Build a unit system with the given constants set to 1 and explicit scales.

    Parameters
    ----------
    *constants : str
        Names from :data:`NATURAL_CONSTANTS` (``"G"``, ``"c"``, ``"hbar"``,
        ``"k_B"``) to set equal to 1.
    name : str, optional
        Label for the system; defaults to e.g. ``"G=c=1"``.
    length_m, mass_kg, time_s, energy_j, temperature_k : float, optional
        SI size of one unit of length, mass, time, energy or temperature.

    Returns
    -------
    UnitSystem

    Raises
    ------
    ValueError
        For an unknown constant name, or scales that contradict the
        constants set to 1 (e.g. :math:`c = 1` with both a length and a
        time scale that are not in ratio :math:`c`).

    Examples
    --------
    Particle-physics units :math:`\hbar = c = 1` with energies in GeV:

    >>> gev = natural_units("hbar", "c", energy_j=1e9 * const.ELECTRONVOLT)
    >>> round(gev.scale("length") * 1e15, 4)  # hbar*c / GeV, in fm
    0.1973
    """
    anchors: list[tuple[Dimension, float]] = []
    for c in constants:
        if c not in NATURAL_CONSTANTS:
            raise ValueError(f"unknown constant {c!r}; choose from {sorted(NATURAL_CONSTANTS)}")
        anchors.append(NATURAL_CONSTANTS[c])
    given = {"length_m": length_m, "mass_kg": mass_kg, "time_s": time_s, "energy_j": energy_j, "temperature_k": temperature_k}
    anchors.extend((_SCALE_KWARGS[k], float(v)) for k, v in given.items() if v is not None)
    return UnitSystem(name if name is not None else "=".join(constants) + "=1", tuple(anchors))


def _exactly_one(**kwargs: float | None) -> None:
    supplied = [k for k, v in kwargs.items() if v is not None]
    if len(supplied) != 1:
        raise ValueError(f"supply exactly one of {', '.join(kwargs)} (got {supplied or 'none'})")


def astro_units(*, length_m: float, mass_kg: float) -> UnitSystem:
    r"""Gravitational units, :math:`G = 1`, as used by :mod:`physicskit.astro`.

    The time unit follows from :math:`G_{SI} = L^3/(M T^2)`, i.e.
    :math:`T = \sqrt{L^3/(G_{SI} M)}` (same as
    :func:`physicskit.constants.gravitational_unit_system`).

    Parameters
    ----------
    length_m : float
        SI length of one length unit, in m.
    mass_kg : float
        SI mass of one mass unit, in kg.

    Returns
    -------
    UnitSystem

    Examples
    --------
    With 1 pc and 1 solar mass, one time unit is about 14.9 Myr:

    >>> u = astro_units(length_m=const.PARSEC_M, mass_kg=const.SOLAR_MASS_KG)
    >>> round(u.scale("time") / (1e6 * 365.25 * 86400), 2)
    14.91
    """
    return natural_units("G", name="astro (G=1)", length_m=length_m, mass_kg=mass_kg)


def geometrized_units(*, mass_kg: float | None = None, length_m: float | None = None) -> UnitSystem:
    r"""Geometrized units, :math:`G = c = 1`, as used by :mod:`physicskit.relativity`.

    With a mass scale :math:`M`, one unit of length is :math:`GM/c^2` and
    one unit of time is :math:`GM/c^3` (Misner, Thorne & Wheeler,
    *Gravitation*, 1973, Box 1.8).

    Parameters
    ----------
    mass_kg : float, optional
        SI mass of one mass unit (conventionally the black-hole or system
        mass :math:`M`).
    length_m : float, optional
        SI length of one length unit, as an alternative to ``mass_kg``.
        Supply exactly one of the two.

    Returns
    -------
    UnitSystem

    Examples
    --------
    >>> u = geometrized_units(mass_kg=const.SOLAR_MASS_KG)
    >>> print(f"{u.scale('length'):.5g} m, {u.scale('time') * 1e6:.4g} us")
    1476.7 m, 4.926 us
    """
    _exactly_one(mass_kg=mass_kg, length_m=length_m)
    return natural_units("G", "c", name="relativity (G=c=1)", mass_kg=mass_kg, length_m=length_m)


def quantum_units(*, mass_kg: float, length_m: float | None = None, energy_j: float | None = None) -> UnitSystem:
    r"""Units with :math:`\hbar = 1`, as used by :mod:`physicskit.quantum`.

    Choosing a mass :math:`m` and a length :math:`a` fixes the energy unit
    :math:`\hbar^2/(m a^2)` and time unit :math:`m a^2/\hbar`. With the
    electron mass and the Bohr radius these are the Hartree atomic units
    (CODATA: :math:`E_h = 27.211` eV, :math:`t_{au} = 2.4189\times10^{-17}` s).

    Parameters
    ----------
    mass_kg : float
        SI mass of one mass unit (the particle mass for ``m = 1`` code).
    length_m : float, optional
        SI length of one length unit.
    energy_j : float, optional
        SI energy of one energy unit, as an alternative to ``length_m``.
        Supply exactly one of the two.

    Returns
    -------
    UnitSystem

    Examples
    --------
    >>> au = quantum_units(mass_kg=const.ELECTRON_MASS, length_m=5.29177210903e-11)
    >>> round(const.joules_to_ev(au.scale("energy")), 3)  # Hartree, eV
    27.211
    """
    _exactly_one(length_m=length_m, energy_j=energy_j)
    return natural_units("hbar", name="quantum (hbar=1)", mass_kg=mass_kg, length_m=length_m, energy_j=energy_j)


def statphys_units(
    *,
    energy_j: float | None = None,
    temperature_k: float | None = None,
    length_m: float | None = None,
    mass_kg: float | None = None,
) -> UnitSystem:
    r"""Units with :math:`k_B = 1`, as used by :mod:`physicskit.statphys`.

    With :math:`k_B = 1` an energy scale :math:`\epsilon` and a
    temperature scale :math:`\epsilon/k_B` are the same choice, which is
    enough for lattice models (e.g. Ising with coupling :math:`J = 1`).
    Molecular dynamics additionally needs a length :math:`\sigma` and
    mass :math:`m`, giving the Lennard-Jones reduced time unit
    :math:`\tau = \sigma\sqrt{m/\epsilon}` (Allen & Tildesley, *Computer
    Simulation of Liquids*, 2nd ed., 2017, Appendix B).

    Parameters
    ----------
    energy_j : float, optional
        SI energy of one energy unit (e.g. the coupling :math:`J`).
    temperature_k : float, optional
        SI temperature of one temperature unit, as an alternative to
        ``energy_j``. Supply exactly one of the two.
    length_m, mass_kg : float, optional
        Length and mass scales, needed only to convert mechanical
        quantities such as time or pressure.

    Returns
    -------
    UnitSystem

    Examples
    --------
    Lennard-Jones argon (:math:`\sigma = 3.405` Å, :math:`\epsilon/k_B =
    119.8` K, :math:`m = 39.948` u) has a time unit of about 2.16 ps:

    >>> u = statphys_units(temperature_k=119.8, length_m=3.405e-10, mass_kg=39.948 * 1.66053906660e-27)
    >>> round(u.scale("time") * 1e12, 2)
    2.16
    """
    _exactly_one(energy_j=energy_j, temperature_k=temperature_k)
    return natural_units(
        "k_B",
        name="statphys (k_B=1)",
        energy_j=energy_j,
        temperature_k=temperature_k,
        length_m=length_m,
        mass_kg=mass_kg,
    )


def to_si(value: ArrayLike, dimension: str | Dimension, system: UnitSystem) -> Any:
    """Convert ``value`` from ``system``'s units to SI.

    Equivalent to ``system.to_si(value, dimension)``.

    Parameters
    ----------
    value : float or array_like
        Quantity in ``system``'s units.
    dimension : str or Dimension
        The quantity's dimension.
    system : UnitSystem
        The unit system ``value`` is expressed in.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> to_si(1.0, "length", geometrized_units(mass_kg=const.SOLAR_MASS_KG)) > 1476
    True
    """
    return system.to_si(value, dimension)


def from_si(value: ArrayLike, dimension: str | Dimension, system: UnitSystem) -> Any:
    """Convert an SI ``value`` into ``system``'s units.

    Equivalent to ``system.from_si(value, dimension)``.

    Parameters
    ----------
    value : float or array_like
        Quantity in SI units.
    dimension : str or Dimension
        The quantity's dimension.
    system : UnitSystem
        The target unit system.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(from_si(300.0, "temperature", statphys_units(temperature_k=300.0)), 12)
    1.0
    """
    return system.from_si(value, dimension)


# --------------------------------------------------------------------------
# Optional pint interop
# --------------------------------------------------------------------------

_PINT_BASE = (("[length]", "meter"), ("[mass]", "kilogram"), ("[time]", "second"), ("[temperature]", "kelvin"))


def _require_pint() -> Any:
    try:
        import pint
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise ImportError('pint is required for this function; install it with `pip install "physicskit[units]"`') from exc
    return pint


def _si_unit_string(dim: Dimension) -> str:
    terms = [f"{unit} ** {exp:g}" for (_, unit), exp in zip(_PINT_BASE, dim.as_vector()) if exp != 0]
    return " * ".join(terms) if terms else "dimensionless"


def to_pint(value: ArrayLike, dimension: str | Dimension, system: UnitSystem, registry: Any = None) -> Any:
    """Convert a natural-unit value into a :mod:`pint` quantity in SI base units.

    Requires the optional ``pint`` dependency (``physicskit[units]``).

    Parameters
    ----------
    value : float or array_like
        Quantity in ``system``'s units.
    dimension : str or Dimension
        The quantity's dimension.
    system : UnitSystem
        The unit system ``value`` is expressed in.
    registry : pint.UnitRegistry, optional
        Registry to create the quantity in; defaults to
        :func:`pint.get_application_registry`.

    Returns
    -------
    pint.Quantity

    Examples
    --------
    >>> q = to_pint(1.0, "length", geometrized_units(mass_kg=const.SOLAR_MASS_KG))  # doctest: +SKIP
    >>> q.to("km")  # doctest: +SKIP
    <Quantity(1.47662..., 'kilometer')>
    """
    pint = _require_pint()
    ureg = registry if registry is not None else pint.get_application_registry()
    dim = _as_dimension(dimension)
    return ureg.Quantity(system.to_si(value, dim), _si_unit_string(dim))


def from_pint(quantity: Any, system: UnitSystem) -> Any:
    """Convert a :mod:`pint` quantity into ``system``'s units.

    The dimension is read from the quantity itself. Requires the optional
    ``pint`` dependency (``physicskit[units]``).

    Parameters
    ----------
    quantity : pint.Quantity
        Any quantity whose dimensionality is built from length, mass, time
        and temperature.
    system : UnitSystem
        The target unit system.

    Returns
    -------
    float or ndarray
        The magnitude in ``system``'s units.

    Raises
    ------
    ValueError
        If ``quantity`` involves another base dimension (e.g. current).

    Examples
    --------
    >>> import pint  # doctest: +SKIP
    >>> ureg = pint.UnitRegistry()  # doctest: +SKIP
    >>> from_pint(1476.6 * ureg.meter, geometrized_units(mass_kg=const.SOLAR_MASS_KG))  # doctest: +SKIP
    0.99997...
    """
    _require_pint()
    dimensionality = dict(quantity.dimensionality)
    known = {key for key, _ in _PINT_BASE}
    extra = set(dimensionality) - known
    if extra:
        raise ValueError(f"cannot convert a quantity with dimensions {sorted(extra)}; only length, mass, time and temperature are supported")
    dim = Dimension(*(float(dimensionality.get(key, 0)) for key, _ in _PINT_BASE))
    magnitude = quantity.to(_si_unit_string(dim)).magnitude
    return system.from_si(magnitude, dim)
