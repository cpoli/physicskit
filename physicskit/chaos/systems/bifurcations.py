"""Hopf bifurcations: the birth of a limit cycle from a fixed point.

A Hopf bifurcation happens when a complex-conjugate pair of eigenvalues of
a fixed point's Jacobian crosses the imaginary axis,
:math:`\\lambda = \\mu \\pm i\\omega` with :math:`\\mu` changing sign
(Andronov 1929; Hopf, Ber. Math.-Phys. Kl. Sächs. Akad. Wiss. Leipzig 94, 1
(1942)). Near onset every such system reduces to the normal form, in polar
coordinates

.. math::

    \\dot r = \\mu r + a r^3 + c r^5, \\qquad \\dot\\phi = \\omega.

For :math:`a < 0` the bifurcation is *supercritical*: a stable limit cycle
of radius :math:`\\sqrt{-\\mu/a}` grows continuously from zero. For
:math:`a > 0` (with :math:`c < 0` to bound the flow) it is *subcritical*:
the fixed point loses stability by colliding with an unstable cycle, the
state jumps to a large-amplitude cycle, and the two coexist over a window of
:math:`\\mu < 0`, giving hysteresis.

:class:`HopfNormalForm` integrates this normal form, and
:class:`Brusselator` is the chemical oscillator of Prigogine and Lefever
(1968), whose fixed point undergoes a supercritical Hopf bifurcation at
:math:`b_c = 1 + a^2`.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

from physicskit.chaos.core.base_system import DynamicalSystem
from physicskit.chaos.core.integrators import rk4_integrate
from physicskit.chaos.exceptions import InvalidParameterError

__all__ = ["Brusselator", "HopfNormalForm"]


@njit(cache=True)
def _hopf_rhs(state: NDArray[np.float64], t: float, params: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cartesian Hopf normal form; ``params = (mu, omega, a, c)``."""
    mu, omega, a, c = params[0], params[1], params[2], params[3]
    x, y = state[0], state[1]
    r2 = x * x + y * y
    g = mu + a * r2 + c * r2 * r2
    out = np.empty(2)
    out[0] = g * x - omega * y
    out[1] = omega * x + g * y
    return out


@njit(cache=True)
def _brusselator_rhs(state: NDArray[np.float64], t: float, params: NDArray[np.float64]) -> NDArray[np.float64]:
    """Brusselator vector field; ``params = (a, b)``."""
    a, b = params[0], params[1]
    x, y = state[0], state[1]
    out = np.empty(2)
    out[0] = a - (b + 1.0) * x + x * x * y
    out[1] = b * x - x * x * y
    return out


class HopfNormalForm(DynamicalSystem):
    """Normal form of the Hopf bifurcation, :math:`\\dot r = \\mu r + a r^3 + c r^5`, :math:`\\dot\\phi = \\omega`.

    Parameters
    ----------
    mu : float, default 0.1
        Bifurcation parameter: the real part of the fixed point's eigenvalues.
    omega : float, default 1.0
        Rotation frequency at onset.
    a : float, default -1.0
        Cubic (first Lyapunov) coefficient; negative is supercritical,
        positive is subcritical.
    c : float, default 0.0
        Quintic coefficient; must be negative when ``a > 0`` to keep the
        flow bounded.

    Examples
    --------
    >>> HopfNormalForm(mu=0.25).limit_cycle_radii()
    [(0.5, 'stable')]
    >>> [(round(r, 3), s) for r, s in HopfNormalForm(mu=-0.1, a=1.0, c=-1.0).limit_cycle_radii()]
    [(0.336, 'unstable'), (0.942, 'stable')]
    """

    #: State dimension, always 2.
    dim = 2

    def __init__(self, mu: float = 0.1, omega: float = 1.0, a: float = -1.0, c: float = 0.0):
        if a > 0 and c >= 0:
            raise InvalidParameterError("a subcritical normal form (a > 0) needs c < 0 to stay bounded")
        self.mu = float(mu)
        self.omega = float(omega)
        self.a = float(a)
        self.c = float(c)

    @property
    def params(self) -> NDArray[np.float64]:
        """Parameter vector ``(mu, omega, a, c)``."""
        return np.array([self.mu, self.omega, self.a, self.c])

    @property
    def eigenvalues(self) -> NDArray[np.complex128]:
        """Eigenvalues :math:`\\mu \\pm i\\omega` of the fixed point at the origin."""
        return np.array([self.mu + 1j * self.omega, self.mu - 1j * self.omega])

    def rhs(self, state: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        """Evaluate the vector field in Cartesian coordinates ``(x, y)``.

        Parameters
        ----------
        state : ndarray of float, shape (2,)
        t : float
            Current time (unused).

        Returns
        -------
        ndarray of float, shape (2,)
        """
        return np.asarray(_hopf_rhs(np.asarray(state, dtype=np.float64), t, self.params))

    def initial_state(self) -> NDArray[np.float64]:
        """Default initial condition ``(0.01, 0)``, just off the fixed point."""
        return np.array([0.01, 0.0])

    def limit_cycle_radii(self) -> list[tuple[float, str]]:
        """Radii of the limit cycles and their stability.

        The cycles are the positive roots :math:`r = \\sqrt{\\rho}` of
        :math:`\\mu + a\\rho + c\\rho^2 = 0`; a cycle is stable when
        :math:`\\dot r` decreases through it.

        Returns
        -------
        list of (float, str)
            ``(radius, "stable" | "unstable")`` in increasing radius.
        """
        coeffs = [self.c, self.a, self.mu] if self.c != 0.0 else [self.a, self.mu]
        rho = np.roots(coeffs) if self.c != 0.0 or self.a != 0.0 else np.array([])
        out = []
        for root in sorted(rho[np.isreal(rho)].real):
            if root > 0:
                slope = self.a + 2.0 * self.c * root  # d(rdot/r)/d(rho)
                out.append((float(np.sqrt(root)), "stable" if slope < 0 else "unstable"))
        return out

    def trajectory(
        self,
        state0: NDArray[np.float64] | None = None,
        t0: float = 0.0,
        dt: float = 0.01,
        n_steps: int = 10000,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Integrate a trajectory with the Numba-accelerated RK4 integrator.

        Parameters
        ----------
        state0 : array_like of float, shape (2,), optional
            Initial state; defaults to :meth:`initial_state`.
        t0 : float, default 0.0
        dt : float, default 0.01
        n_steps : int, default 10000

        Returns
        -------
        times : ndarray of float, shape (n_steps + 1,)
        states : ndarray of float, shape (n_steps + 1, 2)
        """
        state0 = self.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
        return rk4_integrate(_hopf_rhs, state0, t0, dt, n_steps, self.params)


class Brusselator(DynamicalSystem):
    """The Brusselator chemical oscillator of Prigogine and Lefever (1968).

    .. math::

        \\dot x = a - (b + 1)x + x^2 y, \\qquad \\dot y = bx - x^2 y.

    The unique fixed point :math:`(a, b/a)` has Jacobian trace :math:`b - 1 -
    a^2` and determinant :math:`a^2`, so it undergoes a supercritical Hopf
    bifurcation at :math:`b_c = 1 + a^2`, with frequency :math:`\\omega = a`
    at onset.

    Parameters
    ----------
    a : float, default 1.0
    b : float, default 2.5

    Examples
    --------
    >>> Brusselator(a=1.0).hopf_point
    2.0
    """

    #: State dimension, always 2.
    dim = 2

    def __init__(self, a: float = 1.0, b: float = 2.5):
        if a <= 0 or b < 0:
            raise InvalidParameterError("Brusselator needs a > 0 and b >= 0")
        self.a = float(a)
        self.b = float(b)

    @property
    def params(self) -> NDArray[np.float64]:
        """Parameter vector ``(a, b)``."""
        return np.array([self.a, self.b])

    @property
    def fixed_point(self) -> NDArray[np.float64]:
        """The fixed point :math:`(a, b/a)`."""
        return np.array([self.a, self.b / self.a])

    @property
    def hopf_point(self) -> float:
        """Critical :math:`b_c = 1 + a^2`."""
        return 1.0 + self.a**2

    def jacobian(self) -> NDArray[np.float64]:
        """Jacobian at the fixed point, :math:`\\begin{pmatrix} b - 1 & a^2 \\\\ -b & -a^2 \\end{pmatrix}`."""
        return np.array([[self.b - 1.0, self.a**2], [-self.b, -(self.a**2)]])

    def eigenvalues(self) -> NDArray[np.complex128]:
        """Eigenvalues of :meth:`jacobian`."""
        return np.linalg.eigvals(self.jacobian()).astype(np.complex128)

    def rhs(self, state: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        """Evaluate the Brusselator vector field.

        Parameters
        ----------
        state : ndarray of float, shape (2,)
        t : float
            Current time (unused).

        Returns
        -------
        ndarray of float, shape (2,)
        """
        return np.asarray(_brusselator_rhs(np.asarray(state, dtype=np.float64), t, self.params))

    def initial_state(self) -> NDArray[np.float64]:
        """Default initial condition: the fixed point displaced by 0.01 in :math:`x`."""
        return self.fixed_point + np.array([0.01, 0.0])

    def trajectory(
        self,
        state0: NDArray[np.float64] | None = None,
        t0: float = 0.0,
        dt: float = 0.01,
        n_steps: int = 10000,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Integrate a trajectory with the Numba-accelerated RK4 integrator.

        Parameters
        ----------
        state0 : array_like of float, shape (2,), optional
            Initial state; defaults to :meth:`initial_state`.
        t0 : float, default 0.0
        dt : float, default 0.01
        n_steps : int, default 10000

        Returns
        -------
        times : ndarray of float, shape (n_steps + 1,)
        states : ndarray of float, shape (n_steps + 1, 2)
        """
        state0 = self.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
        return rk4_integrate(_brusselator_rhs, state0, t0, dt, n_steps, self.params)
