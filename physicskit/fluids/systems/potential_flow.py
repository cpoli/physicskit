"""Inviscid, irrotational 2D potential flow: superposition of elementary solutions.

Where the flow has no vorticity (:math:`\\nabla\\times\\mathbf{u}=0`) the
velocity is the gradient of a scalar velocity potential,
:math:`\\mathbf{u}=\\nabla\\phi`, and incompressibility
(:math:`\\nabla\\cdot\\mathbf{u}=0`) then makes :math:`\\phi` harmonic:
:math:`\\nabla^2\\phi=0`. Laplace's equation is linear, so any sum of
solutions is itself a solution -- the entire content of this module is that
observation, applied to four elementary flows (a uniform stream, a
source/sink, a doublet, and a point vortex) whose superpositions reproduce
the classic textbook flow fields, most famously flow past a circular
cylinder with lift.

Every element is expressed through its complex potential
:math:`W(z) = \\phi + i\\psi`, :math:`z = x + iy`, since for these
particular elementary flows the complex-analytic form is both the most
compact way to write the solution and, via the Cauchy-Riemann equations, an
automatic guarantee that :math:`\\phi` and its harmonic conjugate
(streamfunction) :math:`\\psi` are consistent: :math:`u - iv = dW/dz`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physicskit.fluids.exceptions import InvalidParameterError

__all__ = [
    "uniform_flow_potential",
    "source_potential",
    "doublet_potential",
    "point_vortex_potential",
    "PotentialFlow",
    "flow_past_cylinder",
    "pressure_coefficient",
    "kutta_joukowski_lift",
]


def _complex_z(X: ArrayLike, Y: ArrayLike) -> NDArray[np.complex128]:
    return np.asarray(X, dtype=np.float64) + 1j * np.asarray(Y, dtype=np.float64)


def uniform_flow_potential(X: ArrayLike, Y: ArrayLike, U_inf: float, alpha: float = 0.0) -> NDArray[np.complex128]:
    """Complex potential of a uniform stream.

    Parameters
    ----------
    X, Y : array_like of float
        Field points.
    U_inf : float
        Free-stream speed.
    alpha : float, default 0.0
        Angle of attack (radians), measured from the ``+x`` axis.

    Returns
    -------
    ndarray of complex
        Complex potential :math:`W(z) = U_\\infty e^{-i\\alpha} z`.
    """
    z = _complex_z(X, Y)
    return U_inf * np.exp(-1j * alpha) * z


def source_potential(X: ArrayLike, Y: ArrayLike, strength: float, x0: float = 0.0, y0: float = 0.0) -> NDArray[np.complex128]:
    """Complex potential of a point source (or, for negative `strength`, a sink).

    Parameters
    ----------
    X, Y : array_like of float
        Field points.
    strength : float
        Source strength (volume flow rate per unit depth); negative for a sink.
    x0, y0 : float, default 0.0
        Source location.

    Returns
    -------
    ndarray of complex
        Complex potential :math:`W(z) = \\frac{m}{2\\pi}\\ln(z - z_0)`.
    """
    z = _complex_z(X, Y) - (x0 + 1j * y0)
    with np.errstate(divide="ignore"):
        return (strength / (2.0 * np.pi)) * np.log(z)


def doublet_potential(X: ArrayLike, Y: ArrayLike, strength: float, x0: float = 0.0, y0: float = 0.0, alpha: float = 0.0) -> NDArray[np.complex128]:
    """Complex potential of a doublet (a source-sink pair in the zero-separation limit).

    Parameters
    ----------
    X, Y : array_like of float
        Field points.
    strength : float
        Doublet strength :math:`\\kappa` (the limit of ``source_strength *
        separation`` as the separation shrinks to zero).
    x0, y0 : float, default 0.0
        Doublet location.
    alpha : float, default 0.0
        Orientation angle (radians) from sink-to-source.

    Returns
    -------
    ndarray of complex
        Complex potential :math:`W(z) = \\frac{\\kappa e^{-i\\alpha}}{2\\pi (z-z_0)}`.
    """
    z = _complex_z(X, Y) - (x0 + 1j * y0)
    with np.errstate(divide="ignore"):
        return (strength * np.exp(-1j * alpha)) / (2.0 * np.pi * z)


def point_vortex_potential(X: ArrayLike, Y: ArrayLike, circulation: float, x0: float = 0.0, y0: float = 0.0) -> NDArray[np.complex128]:
    """Complex potential of an isolated point vortex.

    Parameters
    ----------
    X, Y : array_like of float
        Field points.
    circulation : float
        Circulation :math:`\\Gamma` (positive is counterclockwise).
    x0, y0 : float, default 0.0
        Vortex location.

    Returns
    -------
    ndarray of complex
        Complex potential :math:`W(z) = \\frac{-i\\Gamma}{2\\pi}\\ln(z - z_0)`.
    """
    z = _complex_z(X, Y) - (x0 + 1j * y0)
    with np.errstate(divide="ignore"):
        return (-1j * circulation / (2.0 * np.pi)) * np.log(z)


class PotentialFlow:
    """A superposition of elementary potential-flow solutions.

    Because Laplace's equation is linear, the complex potential of any
    combination of uniform flow, sources/sinks, doublets, and point vortices
    is simply the sum of their individual complex potentials -- exactly what
    this class accumulates. Call :meth:`add_source`, :meth:`add_doublet`, or
    :meth:`add_vortex` to build up a flow on top of the base uniform stream
    ``(U_inf, alpha)``, then
    :meth:`velocity`, :meth:`streamfunction`, or
    :meth:`pressure_coefficient` to evaluate it on a grid.

    Parameters
    ----------
    U_inf : float, default 1.0
        Free-stream speed used both as the base uniform flow and as the
        reference speed for :meth:`pressure_coefficient`.
    alpha : float, default 0.0
        Free-stream angle of attack (radians).

    Attributes
    ----------
    U_inf, alpha : float
        As above.

    Examples
    --------
    >>> flow = PotentialFlow(U_inf=1.0)
    >>> flow.add_doublet(strength=2 * np.pi * 1.0**2)  # uniform flow + doublet = cylinder
    >>> X, Y = np.meshgrid(np.linspace(-3, 3, 5), np.linspace(-3, 3, 5))
    >>> u, v = flow.velocity(X, Y)
    >>> u.shape
    (5, 5)
    """

    def __init__(self, U_inf: float = 1.0, alpha: float = 0.0):
        self.U_inf = float(U_inf)
        self.alpha = float(alpha)
        self._elements: list = []

    def add_source(self, strength: float, x0: float = 0.0, y0: float = 0.0) -> None:
        """Add a point source (or sink, for negative `strength`).

        Parameters
        ----------
        strength : float
            Source strength; negative for a sink.
        x0, y0 : float, default 0.0
            Source location.
        """
        self._elements.append(("source", (strength, x0, y0)))

    def add_doublet(self, strength: float, x0: float = 0.0, y0: float = 0.0, alpha: float = 0.0) -> None:
        """Add a doublet.

        Parameters
        ----------
        strength : float
            Doublet strength.
        x0, y0 : float, default 0.0
            Doublet location.
        alpha : float, default 0.0
            Orientation angle (radians).
        """
        self._elements.append(("doublet", (strength, x0, y0, alpha)))

    def add_vortex(self, circulation: float, x0: float = 0.0, y0: float = 0.0) -> None:
        """Add a point vortex.

        Parameters
        ----------
        circulation : float
            Circulation :math:`\\Gamma`; positive is counterclockwise.
        x0, y0 : float, default 0.0
            Vortex location.
        """
        self._elements.append(("vortex", (circulation, x0, y0)))

    def complex_potential(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.complex128]:
        """Evaluate the total complex potential :math:`W(z)=\\phi+i\\psi`.

        Parameters
        ----------
        X, Y : array_like of float
            Field points.

        Returns
        -------
        ndarray of complex
            Complex potential at each point.
        """
        X = np.asarray(X, dtype=np.float64)
        W = uniform_flow_potential(X, Y, self.U_inf, self.alpha)
        for kind, args in self._elements:
            if kind == "source":
                W = W + source_potential(X, Y, *args)
            elif kind == "doublet":
                W = W + doublet_potential(X, Y, *args)
            elif kind == "vortex":
                W = W + point_vortex_potential(X, Y, *args)
        return W

    def streamfunction(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the streamfunction :math:`\\psi = \\mathrm{Im}(W)`.

        Parameters
        ----------
        X, Y : array_like of float
            Field points.

        Returns
        -------
        ndarray of float
            Streamfunction values.
        """
        return np.imag(self.complex_potential(X, Y))

    def velocity(self, X: ArrayLike, Y: ArrayLike, delta: float = 1e-6) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate the velocity field :math:`u - iv = dW/dz`.

        Differentiates the complex potential with a central difference in
        `X` (equivalent, by the Cauchy-Riemann equations, to differentiating
        in either direction), which sidesteps deriving and summing a
        separate closed-form velocity expression for every element type.

        Parameters
        ----------
        X, Y : array_like of float
            Field points.
        delta : float, default 1e-6
            Finite-difference step used to differentiate the complex potential.

        Returns
        -------
        u, v : ndarray of float
            Velocity components.
        """
        X = np.asarray(X, dtype=np.float64)
        dW = (self.complex_potential(X + delta, Y) - self.complex_potential(X - delta, Y)) / (2.0 * delta)
        return np.real(dW), -np.imag(dW)

    def pressure_coefficient(self, X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the Bernoulli pressure coefficient :math:`C_p = 1 - (u^2+v^2)/U_\\infty^2`.

        Parameters
        ----------
        X, Y : array_like of float
            Field points.

        Returns
        -------
        ndarray of float
            Pressure coefficient at each point.
        """
        return pressure_coefficient(*self.velocity(X, Y), self.U_inf)


def pressure_coefficient(u: ArrayLike, v: ArrayLike, U_inf: float) -> NDArray[np.float64]:
    """Bernoulli's pressure coefficient for steady, incompressible potential flow.

    Along a streamline of a steady, incompressible, inviscid flow, Bernoulli's
    equation :math:`p + \\tfrac{1}{2}\\rho|\\mathbf{u}|^2 = \\mathrm{const}`
    gives the dimensionless pressure coefficient

    .. math::

        C_p = \\frac{p - p_\\infty}{\\tfrac{1}{2}\\rho U_\\infty^2}
            = 1 - \\frac{u^2+v^2}{U_\\infty^2}.

    Parameters
    ----------
    u, v : array_like of float
        Velocity components.
    U_inf : float
        Free-stream speed.

    Returns
    -------
    ndarray of float
        Pressure coefficient.

    Raises
    ------
    InvalidParameterError
        If `U_inf` is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> bool(np.isclose(pressure_coefficient(u=1.0, v=0.0, U_inf=1.0), 0.0))
    True
    """
    if U_inf <= 0:
        raise InvalidParameterError(f"U_inf must be positive, got {U_inf}")
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return 1.0 - (u**2 + v**2) / U_inf**2


def flow_past_cylinder(U_inf: float, radius: float, circulation: float = 0.0) -> PotentialFlow:
    """Build the classic potential flow past a circular cylinder, with optional lift.

    Superposing a uniform stream with a doublet of strength
    :math:`\\kappa = 2\\pi U_\\infty R^2` places a circular streamline of
    radius `radius` exactly at the origin -- since :math:`\\psi=0` on that
    circle can be shown to be exactly the streamfunction of this
    combination, the circle itself can be interpreted as a solid boundary
    (potential flow does not enforce a no-slip condition, only no
    penetration). Adding a point vortex at the same location keeps the
    circular streamline intact (a vortex centered on the circle induces
    purely tangential velocity on it) while breaking front-back symmetry,
    which is exactly what a real cylinder's boundary layer does when it
    separates asymmetrically -- e.g. from a spinning cylinder (the Magnus
    effect).

    Parameters
    ----------
    U_inf : float
        Free-stream speed.
    radius : float
        Cylinder radius.
    circulation : float, default 0.0
        Circulation :math:`\\Gamma` added at the cylinder center; `0.0` gives
        the symmetric, lift-free flow.

    Returns
    -------
    PotentialFlow
        The assembled flow: uniform stream + doublet (+ vortex if `circulation`
        is nonzero).

    Raises
    ------
    InvalidParameterError
        If `U_inf` or `radius` is not positive.

    See Also
    --------
    kutta_joukowski_lift : The lift force generated by `circulation`.

    Examples
    --------
    >>> flow = flow_past_cylinder(U_inf=1.0, radius=1.0, circulation=4 * np.pi)
    >>> theta = np.linspace(0, 2 * np.pi, 9)
    >>> X, Y = 1.0 * np.cos(theta), 1.0 * np.sin(theta)
    >>> psi = flow.streamfunction(X, Y)
    >>> bool(np.max(np.abs(psi - psi[0])) < 1e-8)  # the cylinder surface is one streamline
    True
    """
    if U_inf <= 0:
        raise InvalidParameterError(f"U_inf must be positive, got {U_inf}")
    if radius <= 0:
        raise InvalidParameterError(f"radius must be positive, got {radius}")
    flow = PotentialFlow(U_inf=U_inf)
    flow.add_doublet(strength=2.0 * np.pi * U_inf * radius**2)
    if circulation != 0.0:
        flow.add_vortex(circulation=circulation)
    return flow


def kutta_joukowski_lift(rho: float, U_inf: float, circulation: float) -> float:
    """The Kutta-Joukowski lift theorem: lift per unit span from bound circulation.

    .. math::

        L' = -\\rho\\,U_\\infty\\,\\Gamma

    Any 2D body generating a net circulation :math:`\\Gamma` around itself in
    a stream of speed :math:`U_\\infty` experiences a lift force per unit
    span of exactly this magnitude, directed perpendicular to the free
    stream -- true regardless of the body's shape, a remarkable consequence
    of potential theory (Kutta 1902, Zhukovsky 1906) that underlies all of
    classical airfoil theory. Real airfoils select `circulation` via the
    Kutta condition (smooth flow off a sharp trailing edge); the cylinder
    with an added point vortex built by :func:`flow_past_cylinder` sets it
    directly. With this module's convention (free stream toward :math:`+x`,
    :math:`\\Gamma>0` counterclockwise, as in :func:`point_vortex_potential`)
    the lift is signed along :math:`+y`: a clockwise circulation
    (:math:`\\Gamma<0`), which speeds up the flow over the top, lifts
    upward, as on a conventional airfoil.

    Parameters
    ----------
    rho : float
        Fluid density.
    U_inf : float
        Free-stream speed.
    circulation : float
        Circulation :math:`\\Gamma` (positive counterclockwise) bound to the body.

    Returns
    -------
    float
        Lift force per unit span along :math:`+y`, :math:`L'`.

    Raises
    ------
    InvalidParameterError
        If `rho` or `U_inf` is not positive.

    Examples
    --------
    >>> round(kutta_joukowski_lift(rho=1.2, U_inf=10.0, circulation=-5.0), 1)
    60.0
    """
    if rho <= 0:
        raise InvalidParameterError(f"rho must be positive, got {rho}")
    if U_inf <= 0:
        raise InvalidParameterError(f"U_inf must be positive, got {U_inf}")
    return -rho * U_inf * circulation
