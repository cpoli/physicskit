r"""Relativistic four-vector kinematics.

Works throughout in **natural units with** :math:`c=1`: energy, momentum,
and mass all share the same energy-like unit (GeV for particle physics,
MeV for nuclear physics -- the caller's choice, so long as it is used
consistently), the same convention :mod:`physicskit.relativity` uses for
its own geometrized units (:math:`G=c=1`).

- :class:`FourVector` -- an energy-momentum four-vector :math:`p^\mu =
  (E, p_x, p_y, p_z)`, with its invariant mass, speed, and Lorentz factor
  as derived properties.
- :func:`boost` -- a Lorentz boost along a Cartesian axis.
- :func:`boost_generic` -- a Lorentz boost along an arbitrary 3-velocity.
- :func:`invariant_mass` -- the invariant mass of a system of particles.
- :func:`rapidity` -- the additive (under boosts) rapidity variable.
- :func:`boost_to_com` -- the velocity of a system's center-of-momentum frame.
"""

from __future__ import annotations

import numpy as np

__all__ = ["FourVector", "boost", "boost_generic", "invariant_mass", "rapidity", "boost_to_com"]


class FourVector:
    r"""An energy-momentum four-vector :math:`p^\mu=(E,p_x,p_y,p_z)`.

    Parameters
    ----------
    E : float
        Energy.
    px, py, pz : float
        Momentum components.

    Examples
    --------
    >>> p = FourVector(2.0, 0.0, 0.0, 1.0)
    >>> round(p.mass, 6)
    1.732051
    """

    def __init__(self, E, px, py, pz):
        self.E = float(E)
        self.px = float(px)
        self.py = float(py)
        self.pz = float(pz)

    @property
    def p_vec(self):
        """``ndarray`` of shape (3,): the momentum 3-vector."""
        return np.array([self.px, self.py, self.pz])

    @property
    def p_mag(self):
        r"""Momentum magnitude, :math:`|\vec p|`."""
        return float(np.linalg.norm(self.p_vec))

    @property
    def mass(self):
        r"""Invariant mass, :math:`m=\sqrt{E^2-|\vec p|^2}`.

        The radicand is clipped at 0 before the square root, so a
        four-vector that is numerically spacelike only by floating-point
        noise (e.g. the sum of several on-shell four-vectors) returns
        ``0.0`` rather than raising on a tiny negative argument.
        """
        radicand = self.E**2 - self.p_mag**2
        return float(np.sqrt(max(radicand, 0.0)))

    @property
    def beta(self):
        r"""Speed, :math:`\beta=|\vec p|/E`."""
        return self.p_mag / self.E

    @property
    def gamma(self):
        r"""Lorentz factor, :math:`\gamma=E/m`.

        Raises
        ------
        ZeroDivisionError
            If the four-vector is massless (no rest frame).
        """
        m = self.mass
        if m == 0.0:
            raise ZeroDivisionError("A massless four-vector has no rest frame / undefined gamma.")
        return self.E / m

    def __add__(self, other):
        return FourVector(self.E + other.E, self.px + other.px, self.py + other.py, self.pz + other.pz)

    def __sub__(self, other):
        return FourVector(self.E - other.E, self.px - other.px, self.py - other.py, self.pz - other.pz)

    def __repr__(self):
        return f"FourVector(E={self.E:.4g}, px={self.px:.4g}, py={self.py:.4g}, pz={self.pz:.4g})"


def boost(four_vector, beta, axis="z"):
    r"""Lorentz-boost a :class:`FourVector` along a Cartesian axis.

    Parameters
    ----------
    four_vector : FourVector
        The four-vector to boost.
    beta : float
        Boost velocity, :math:`-1 < \beta < 1`, in units of :math:`c=1`.
    axis : {"x", "y", "z"}, default="z"
        Cartesian axis of the boost.

    Returns
    -------
    FourVector
        The boosted four-vector.

    Examples
    --------
    >>> p = FourVector(1.0, 0.0, 0.0, 0.0)  # a particle at rest, mass 1
    >>> pb = boost(p, 0.6, axis="z")
    >>> round(pb.E, 6), round(pb.pz, 6)
    (1.25, 0.75)
    """
    if not (-1.0 < beta < 1.0):
        raise ValueError(f"beta={beta} must satisfy -1 < beta < 1.")
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    E, px, py, pz = four_vector.E, four_vector.px, four_vector.py, four_vector.pz
    if axis == "x":
        return FourVector(gamma * (E + beta * px), gamma * (px + beta * E), py, pz)
    elif axis == "y":
        return FourVector(gamma * (E + beta * py), px, gamma * (py + beta * E), pz)
    elif axis == "z":
        return FourVector(gamma * (E + beta * pz), px, py, gamma * (pz + beta * E))
    raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}.")


def boost_generic(four_vector, beta_vec):
    r"""Lorentz-boost a :class:`FourVector` along an arbitrary 3-velocity.

    Generalizes :func:`boost` to a boost direction that need not lie
    along a Cartesian axis -- the transformation used, e.g., to take a
    daughter four-momentum computed in a parent's rest frame back to the
    lab frame when the parent's own flight direction is not one of the
    axes (as happens at every vertex of a branching cascade; see
    :mod:`physicskit.particle.collider`).

    .. math::

        E' = \gamma(E + \vec\beta\cdot\vec p), \qquad
        \vec p' = \vec p + \left[\frac{\gamma-1}{\beta^2}(\vec\beta\cdot\vec p)
        + \gamma E\right]\vec\beta

    which reduces to :func:`boost` when :math:`\vec\beta` is aligned with
    a single axis.

    Parameters
    ----------
    four_vector : FourVector
        The four-vector to boost.
    beta_vec : array_like of shape (3,)
        Boost velocity, :math:`|\vec\beta| < 1`, in units of :math:`c=1`.

    Returns
    -------
    FourVector
        The boosted four-vector.

    Examples
    --------
    >>> p = FourVector(1.0, 0.0, 0.0, 0.0)  # a particle at rest, mass 1
    >>> pb = boost_generic(p, [0.0, 0.0, 0.6])
    >>> round(pb.E, 6), round(pb.pz, 6)
    (1.25, 0.75)
    """
    beta_vec = np.asarray(beta_vec, dtype=float)
    beta2 = float(beta_vec @ beta_vec)
    if beta2 >= 1.0:
        raise ValueError(f"|beta_vec|={np.sqrt(beta2)} must be < 1.")
    if beta2 == 0.0:
        return FourVector(four_vector.E, four_vector.px, four_vector.py, four_vector.pz)
    gamma = 1.0 / np.sqrt(1.0 - beta2)
    E, p_vec = four_vector.E, four_vector.p_vec
    p_dot_beta = float(p_vec @ beta_vec)
    p_new = p_vec + ((gamma - 1.0) * p_dot_beta / beta2 + gamma * E) * beta_vec
    E_new = gamma * (E + p_dot_beta)
    return FourVector(float(E_new), *[float(c) for c in p_new])


def invariant_mass(four_vectors):
    """The invariant mass of a system of four-vectors.

    Parameters
    ----------
    four_vectors : iterable of FourVector
        The particles making up the system.

    Returns
    -------
    float
        The mass of the sum of the four-vectors -- e.g. the reconstructed
        mass of a resonance from its decay products.

    Examples
    --------
    >>> p1 = FourVector(1.0, 0.0, 0.0, 0.6)
    >>> p2 = FourVector(1.0, 0.0, 0.0, -0.6)
    >>> round(invariant_mass([p1, p2]), 6)
    2.0
    """
    total = None
    for p in four_vectors:
        total = p if total is None else total + p
    return total.mass


def rapidity(four_vector, axis="z"):
    r"""Rapidity along a Cartesian axis, :math:`y=\tfrac12\ln\!\big[(E+p_i)/(E-p_i)\big]`.

    Rapidity is additive under boosts along the same axis: boosting a
    particle by velocity :math:`\beta` shifts its rapidity by exactly
    :math:`\operatorname{artanh}\beta`.

    Parameters
    ----------
    four_vector : FourVector
        The particle.
    axis : {"x", "y", "z"}, default="z"
        Cartesian axis.

    Returns
    -------
    float

    Examples
    --------
    >>> p = FourVector(2.0, 0.0, 0.0, 1.0)
    >>> round(rapidity(p), 6)
    0.549306
    """
    E = four_vector.E
    p_i = {"x": four_vector.px, "y": four_vector.py, "z": four_vector.pz}[axis]
    return float(0.5 * np.log((E + p_i) / (E - p_i)))


def boost_to_com(four_vectors):
    """The 3-velocity of the center-of-momentum frame of a system.

    Parameters
    ----------
    four_vectors : iterable of FourVector
        The particles making up the system.

    Returns
    -------
    ndarray of shape (3,)
        The velocity (units of :math:`c=1`) of the system's total
        momentum relative to the current frame, :math:`\\vec\\beta_{\\rm
        com} = \\vec p_{\\rm tot}/E_{\\rm tot}`.

    Examples
    --------
    >>> p1 = FourVector(2.0, 0.0, 0.0, 1.0)
    >>> p2 = FourVector(1.0, 0.0, 0.0, 0.0)
    >>> beta_com = boost_to_com([p1, p2])
    >>> [round(float(b), 6) for b in beta_com]
    [0.0, 0.0, 0.333333]
    """
    total = None
    for p in four_vectors:
        total = p if total is None else total + p
    return total.p_vec / total.E
