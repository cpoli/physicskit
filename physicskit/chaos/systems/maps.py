"""Discrete chaotic maps: the Standard Map, the Henon Map, the Baker's Map, and the Logistic Map."""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

from physicskit.chaos.core.base_system import DiscreteMap
from physicskit.chaos.exceptions import InvalidParameterError

TWO_PI: float = 2.0 * np.pi


@njit(cache=True)
def _standard_map_step(theta: float, p: float, k: float) -> tuple[float, float]:
    """Single Chirikov-Taylor standard map iteration.

    Parameters
    ----------
    theta : float
        Current angle, in radians.
    p : float
        Current momentum.
    k : float
        Kick strength.

    Returns
    -------
    theta_new, p_new : float
        Next angle and momentum, both wrapped into ``[0, 2*pi)``.
    """
    p_new = (p + k * np.sin(theta)) % TWO_PI
    theta_new = (theta + p_new) % TWO_PI
    return theta_new, p_new


@njit(cache=True, nogil=True)
def _standard_map_trajectory(theta0: float, p0: float, k: float, n_iter: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Iterate the standard map ``n_iter`` times from ``(theta0, p0)``.

    Parameters
    ----------
    theta0 : float
        Initial angle, in radians.
    p0 : float
        Initial momentum.
    k : float
        Kick strength.
    n_iter : int
        Number of iterations.

    Returns
    -------
    thetas, ps : ndarray of float, shape (n_iter + 1,)
        Angle and momentum at each iteration, including the initial values.
    """
    thetas = np.empty(n_iter + 1)
    ps = np.empty(n_iter + 1)
    thetas[0], ps[0] = theta0, p0
    theta, p = theta0, p0
    for i in range(n_iter):
        theta, p = _standard_map_step(theta, p, k)
        thetas[i + 1] = theta
        ps[i + 1] = p
    return thetas, ps


class StandardMap(DiscreteMap):
    """The Chirikov-Taylor standard map on the ``(theta, p)`` cylinder.

    ``k=0`` is integrable; chaos onset is around ``k~1``, with global chaos
    for ``k >~ 4-5``.

    Parameters
    ----------
    k : float, default 1.0
        Kick strength.

    Attributes
    ----------
    k : float
        Kick strength.
    """

    #: State dimension, always 2. State is ``(theta, p)``.
    dim = 2

    def __init__(self, k: float = 1.0):
        self.k = float(k)

    def step(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Advance ``(theta, p)`` by one map iteration.

        Parameters
        ----------
        state : array_like of float, shape (2,)
            Current state ``(theta, p)``.

        Returns
        -------
        ndarray of float, shape (2,)
            Next state ``(theta, p)``.
        """
        theta, p = _standard_map_step(float(state[0]), float(state[1]), self.k)
        return np.array([theta, p])

    def initial_state(self) -> NDArray[np.float64]:
        """Default initial condition ``(0.1, 0.1)``.

        Returns
        -------
        ndarray of float, shape (2,)
        """
        return np.array([0.1, 0.1])

    def trajectory(self, state0: NDArray[np.float64] | None = None, n_iter: int = 1000) -> NDArray[np.float64]:
        """Iterate the map ``n_iter`` times starting from ``state0``.

        Parameters
        ----------
        state0 : array_like of float, shape (2,), optional
            Initial state ``(theta, p)``; defaults to :meth:`initial_state`.
        n_iter : int, default 1000
            Number of iterations.

        Returns
        -------
        ndarray of float, shape (n_iter + 1, 2)
            State at each iteration, including ``state0`` as row 0.
        """
        state0 = self.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
        thetas, ps = _standard_map_trajectory(float(state0[0]), float(state0[1]), self.k, n_iter)
        return np.column_stack([thetas, ps])


@njit(cache=True)
def _henon_map_step(x: float, y: float, a: float, b: float) -> tuple[float, float]:
    """Single Henon map iteration.

    Parameters
    ----------
    x, y : float
        Current state.
    a, b : float
        Map parameters.

    Returns
    -------
    x_new, y_new : float
        Next state.
    """
    return 1.0 - a * x * x + y, b * x


@njit(cache=True, nogil=True)
def _henon_map_trajectory(x0: float, y0: float, a: float, b: float, n_iter: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Iterate the Henon map ``n_iter`` times from ``(x0, y0)``.

    Parameters
    ----------
    x0, y0 : float
        Initial state.
    a, b : float
        Map parameters.
    n_iter : int
        Number of iterations.

    Returns
    -------
    xs, ys : ndarray of float, shape (n_iter + 1,)
        State components at each iteration, including the initial values.
    """
    xs = np.empty(n_iter + 1)
    ys = np.empty(n_iter + 1)
    xs[0], ys[0] = x0, y0
    x, y = x0, y0
    for i in range(n_iter):
        x, y = _henon_map_step(x, y, a, b)
        xs[i + 1] = x
        ys[i + 1] = y
    return xs, ys


class HenonMap(DiscreteMap):
    """The Henon map ``x' = 1 - a*x^2 + y, y' = b*x``.

    The classic chaotic parameters are ``a=1.4, b=0.3``.

    Parameters
    ----------
    a : float, default 1.4
        Map parameter.
    b : float, default 0.3
        Map parameter.

    Attributes
    ----------
    a, b : float
        Map parameters.
    """

    #: State dimension, always 2. State is ``(x, y)``.
    dim = 2

    def __init__(self, a: float = 1.4, b: float = 0.3):
        self.a = float(a)
        self.b = float(b)

    def step(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Advance ``(x, y)`` by one map iteration.

        Parameters
        ----------
        state : array_like of float, shape (2,)
            Current state ``(x, y)``.

        Returns
        -------
        ndarray of float, shape (2,)
            Next state ``(x, y)``.
        """
        x, y = _henon_map_step(float(state[0]), float(state[1]), self.a, self.b)
        return np.array([x, y])

    def initial_state(self) -> NDArray[np.float64]:
        """Default initial condition ``(0, 0)``.

        Returns
        -------
        ndarray of float, shape (2,)
        """
        return np.array([0.0, 0.0])

    def trajectory(self, state0: NDArray[np.float64] | None = None, n_iter: int = 1000) -> NDArray[np.float64]:
        """Iterate the map ``n_iter`` times starting from ``state0``.

        Parameters
        ----------
        state0 : array_like of float, shape (2,), optional
            Initial state ``(x, y)``; defaults to :meth:`initial_state`.
        n_iter : int, default 1000
            Number of iterations.

        Returns
        -------
        ndarray of float, shape (n_iter + 1, 2)
            State at each iteration, including ``state0`` as row 0.
        """
        state0 = self.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
        xs, ys = _henon_map_trajectory(float(state0[0]), float(state0[1]), self.a, self.b, n_iter)
        return np.column_stack([xs, ys])


@njit(cache=True)
def _bakers_map_step(x: float, y: float, alpha: float) -> tuple[float, float]:
    """Single (generalized) baker's map iteration.

    Parameters
    ----------
    x, y : float
        Current state, each in ``[0, 1)``.
    alpha : float
        Cut position, in ``(0, 1)``.

    Returns
    -------
    x_new, y_new : float
        Next state, each in ``[0, 1)``.
    """
    if x < alpha:
        return x / alpha, alpha * y
    return (x - alpha) / (1.0 - alpha), alpha + (1.0 - alpha) * y


@njit(cache=True, nogil=True)
def _bakers_map_trajectory(x0: float, y0: float, alpha: float, n_iter: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Iterate the baker's map ``n_iter`` times from ``(x0, y0)``.

    Parameters
    ----------
    x0, y0 : float
        Initial state, each in ``[0, 1)``.
    alpha : float
        Cut position, in ``(0, 1)``.
    n_iter : int
        Number of iterations.

    Returns
    -------
    xs, ys : ndarray of float, shape (n_iter + 1,)
        State components at each iteration, including the initial values.
    """
    xs = np.empty(n_iter + 1)
    ys = np.empty(n_iter + 1)
    xs[0], ys[0] = x0, y0
    x, y = x0, y0
    for i in range(n_iter):
        x, y = _bakers_map_step(x, y, alpha)
        xs[i + 1] = x
        ys[i + 1] = y
    return xs, ys


class BakersMap(DiscreteMap):
    """The (generalized) baker's map on the unit square: the textbook model of chaos.

    The map cuts the unit square ``[0, 1) x [0, 1)`` at ``x = alpha``, stretches
    each piece horizontally back to unit width (contracting it vertically to
    match), and stacks the two pieces -- literally the "stretch, cut, and
    stack" mechanism used to motivate deterministic chaos:

    .. math::

        T(x, y) = \\begin{cases}
            (x / \\alpha,\\ \\alpha y) & 0 \\le x < \\alpha \\\\
            ((x - \\alpha) / (1 - \\alpha),\\ \\alpha + (1 - \\alpha) y) & \\alpha \\le x < 1
        \\end{cases}

    Because each branch is affine with Jacobian determinant exactly 1, the map
    is *area-preserving* (unlike the dissipative Henon map) while still being
    uniformly hyperbolic, ergodic, and mixing: a single long orbit fills the
    unit square uniformly and densely, and its Lyapunov exponents are known
    exactly in closed form (see :meth:`lyapunov_exponents`), making it the
    standard textbook example for validating numerical chaos estimators.
    ``alpha=0.5`` (the default) recovers the classic symmetric baker's map.

    Parameters
    ----------
    alpha : float, default 0.5
        Cut position, in ``(0, 1)``; the classic symmetric map has
        ``alpha=0.5``.

    Attributes
    ----------
    alpha : float
        Cut position.

    Raises
    ------
    ValueError
        If `alpha` does not satisfy ``0 < alpha < 1``.

    Notes
    -----
    ``x = 0`` is an exact fixed point of the map for *every* `alpha` (since
    ``0 / alpha = 0``, always in the first branch). Because each branch is
    expanding, floating-point rounding error is amplified every step, and
    for many choices of `alpha` and the initial condition, a long-enough
    orbit eventually rounds to exactly ``0.0`` in floating point and gets
    permanently trapped there -- silently producing a degenerate, physically
    meaningless trajectory rather than raising an error. This typically
    happens within a few hundred to a few thousand iterations (it depends
    sensitively on `alpha` and the initial condition: it may also never
    happen within a given run). Do not rely on :meth:`trajectory` orbits of
    more than a few hundred iterations to be a faithful sample of the
    invariant measure without first checking the orbit hasn't collapsed;
    :class:`~physicskit.chaos.systems.maps.HenonMap` and continuous systems such as
    :class:`~physicskit.chaos.systems.continuous.Lorenz` do not share this failure
    mode and are better suited to long-trajectory statistics.
    """

    #: State dimension, always 2. State is ``(x, y)``.
    dim = 2

    def __init__(self, alpha: float = 0.5):
        if not 0.0 < alpha < 1.0:
            raise InvalidParameterError("alpha must satisfy 0 < alpha < 1")
        self.alpha = float(alpha)

    def step(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Advance ``(x, y)`` by one map iteration.

        Parameters
        ----------
        state : array_like of float, shape (2,)
            Current state ``(x, y)``, each in ``[0, 1)``.

        Returns
        -------
        ndarray of float, shape (2,)
            Next state ``(x, y)``.
        """
        x, y = _bakers_map_step(float(state[0]), float(state[1]), self.alpha)
        return np.array([x, y])

    def initial_state(self) -> NDArray[np.float64]:
        """Default initial condition ``(0.5, 0.5)``.

        Returns
        -------
        ndarray of float, shape (2,)
        """
        return np.array([0.5, 0.5])

    def trajectory(self, state0: NDArray[np.float64] | None = None, n_iter: int = 1000) -> NDArray[np.float64]:
        """Iterate the map ``n_iter`` times starting from ``state0``.

        Parameters
        ----------
        state0 : array_like of float, shape (2,), optional
            Initial state ``(x, y)``, each in ``[0, 1)``; defaults to
            :meth:`initial_state`.
        n_iter : int, default 1000
            Number of iterations.

        Returns
        -------
        ndarray of float, shape (n_iter + 1, 2)
            State at each iteration, including ``state0`` as row 0.
        """
        state0 = self.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
        xs, ys = _bakers_map_trajectory(float(state0[0]), float(state0[1]), self.alpha, n_iter)
        return np.column_stack([xs, ys])

    def lyapunov_exponents(self) -> tuple[float, float]:
        """Exact Lyapunov exponents of the map, in closed form.

        The baker's map is piecewise-linear and uniformly hyperbolic: almost
        every orbit spends a fraction `alpha` of its time in the expanding-by
        ``1/alpha`` branch and ``1 - alpha`` in the expanding-by
        ``1/(1 - alpha)`` branch, so by the ergodic theorem its Lyapunov
        exponent equals the Shannon entropy of that two-symbol Bernoulli
        process -- no numerical estimation needed. This is exactly what makes
        the baker's map useful as a ground truth for validating
        :func:`physicskit.chaos.utils.metrics.lyapunov_exponent_from_divergence` and
        similar numerical estimators.

        Returns
        -------
        lambda_expanding, lambda_contracting : float
            The two Lyapunov exponents, ``+h`` and ``-h``, where
            ``h = -alpha*ln(alpha) - (1-alpha)*ln(1-alpha)`` (``= ln(2)`` for
            the classic symmetric map, ``alpha=0.5``).
        """
        alpha = self.alpha
        h = -(alpha * np.log(alpha) + (1.0 - alpha) * np.log(1.0 - alpha))
        return h, -h


@njit(cache=True)
def _logistic_map_step(x: float, r: float) -> float:
    """Single logistic map iteration.

    Parameters
    ----------
    x : float
        Current state, in ``[0, 1]``.
    r : float
        Growth rate parameter.

    Returns
    -------
    float
        Next state.
    """
    return r * x * (1.0 - x)


@njit(cache=True, nogil=True)
def _logistic_map_trajectory(x0: float, r: float, n_iter: int) -> NDArray[np.float64]:
    """Iterate the logistic map ``n_iter`` times from ``x0``.

    Parameters
    ----------
    x0 : float
        Initial state, in ``[0, 1]``.
    r : float
        Growth rate parameter.
    n_iter : int
        Number of iterations.

    Returns
    -------
    ndarray of float, shape (n_iter + 1,)
        State at each iteration, including the initial value.
    """
    xs = np.empty(n_iter + 1)
    xs[0] = x0
    x = x0
    for i in range(n_iter):
        x = _logistic_map_step(x, r)
        xs[i + 1] = x
    return xs


class LogisticMap(DiscreteMap):
    """The logistic map ``x' = r*x*(1-x)``: the simplest gateway to chaos.

    Varying the single growth-rate parameter `r` takes this map through the
    complete period-doubling route to chaos: a stable fixed point for
    ``r < 3``, then successive period-doubling bifurcations at ``r`` values
    that accumulate geometrically (ratio converging to the universal
    Feigenbaum constant ``delta ~= 4.669``) onto the onset of chaos at
    ``r ~= 3.5699``, beyond which the map is chaotic for most (but not all --
    note the periodic windows, the largest at ``r ~= 3.8284``) values of `r`
    up to ``r=4``.

    Parameters
    ----------
    r : float, default 3.9
        Growth rate parameter; interesting values range over ``[0, 4]``.

    Attributes
    ----------
    r : float
        Growth rate parameter.
    """

    #: State dimension, always 1. State is ``(x,)``.
    dim = 1

    def __init__(self, r: float = 3.9):
        self.r = float(r)

    def step(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Advance ``x`` by one map iteration.

        Parameters
        ----------
        state : array_like of float, shape (1,)
            Current state ``(x,)``.

        Returns
        -------
        ndarray of float, shape (1,)
            Next state ``(x,)``.
        """
        return np.array([_logistic_map_step(float(state[0]), self.r)])

    def initial_state(self) -> NDArray[np.float64]:
        """Default initial condition ``(0.5,)``.

        Returns
        -------
        ndarray of float, shape (1,)
        """
        return np.array([0.5])

    def trajectory(self, state0: NDArray[np.float64] | None = None, n_iter: int = 1000) -> NDArray[np.float64]:
        """Iterate the map ``n_iter`` times starting from ``state0``.

        Parameters
        ----------
        state0 : array_like of float, shape (1,), optional
            Initial state ``(x,)``; defaults to :meth:`initial_state`.
        n_iter : int, default 1000
            Number of iterations.

        Returns
        -------
        ndarray of float, shape (n_iter + 1, 1)
            State at each iteration, including ``state0`` as row 0.
        """
        state0 = self.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
        xs = _logistic_map_trajectory(float(state0[0]), self.r, n_iter)
        return xs.reshape(-1, 1)
