r"""Feynman path integrals and their reduction to classical mechanics as :math:`\hbar\to 0`.

Dirac's 1933 paper "The Lagrangian in Quantum Mechanics" observed that the
quantum propagator ought to be built from :math:`\exp(iS/\hbar)`, one
factor per infinitesimal time step; Feynman (1948), and later Feynman &
Hibbs (1965) in their textbook *Quantum Mechanics and Path Integrals*,
turned that observation into a full reformulation of quantum mechanics:
the amplitude to go from :math:`(x_0,0)` to :math:`(x_f,T)` is a sum over
*every* continuous path :math:`x(t)` connecting the two endpoints, each
weighted by a phase built from the classical action functional
:math:`S[x(t)]=\int_0^T L(x,\dot x)\,dt` of that path,

.. math::

    K(x_f,T;x_0,0) = \int \mathcal{D}[x(t)]\;
    \exp\!\left(\frac{i}{\hbar}S[x(t)]\right).

This module implements that sum directly, by Monte Carlo sampling a large
ensemble of candidate paths and adding up their phasors
:math:`\exp(iS/\hbar)` one at a time, ordered from the path closest to the
true classical (stationary-action) trajectory outward. Two systems are
exactly solvable end to end here -- the free particle
(:func:`free_particle_classical_path`,
:func:`free_particle_classical_action`) and the harmonic oscillator
(:func:`harmonic_oscillator_classical_path`,
:func:`harmonic_oscillator_classical_action`) -- so every piece of the
machinery can be checked against a closed form.

The physical content is the emergence of the principle of stationary
action itself. Near the classical path, :math:`S` is stationary
(:math:`\delta S=0` to first order in the deformation), so nearby paths
all have nearly the same action and their phasors stay nearly aligned:
they interfere constructively. Far from the classical path, :math:`S`
varies rapidly from one path to a neighboring one, so
:math:`\exp(iS/\hbar)` spins rapidly through many full turns as the path
is varied, and the contributions of an ensemble of such paths cancel
almost completely by destructive interference (this is Feynman's own
"arrow and stopwatch" picture, reproduced here as the phasor spiral built
by :func:`feynman_phasor_partial_sums`). Shrinking :math:`\hbar` makes the
phase :math:`S/\hbar` more sensitive to path deformations, so the
"nearly aligned" neighborhood of the classical path shrinks -- in the
:math:`\hbar\to 0` limit only the classical path itself survives, and
quantum mechanics' sum over histories reduces exactly to the classical
principle of stationary (least) action, :math:`\delta S=0`.

Paths here are represented by their positions at :math:`N+1` uniformly
spaced times over :math:`[0,T]` (:math:`N` time slices,
:math:`\mathrm{d}t=T/N`), and the action of a discretized path is
computed with the standard midpoint-rule time slicing,

.. math::

    S[x] \approx \sum_{i=0}^{N-1}\left[\frac{1}{2}m
    \left(\frac{x_{i+1}-x_i}{\mathrm{d}t}\right)^2
    - V\!\left(\frac{x_i+x_{i+1}}{2}\right)\right]\mathrm{d}t
    \;\xrightarrow[N\to\infty]{}\; \int_0^T\!\left[\frac{1}{2}m\dot x^2
    - V(x)\right]dt.

References
----------
P. A. M. Dirac, "The Lagrangian in Quantum Mechanics," Physikalische
Zeitschrift der Sowjetunion 3, 64-72 (1933).

R. P. Feynman and A. R. Hibbs, *Quantum Mechanics and Path Integrals*,
McGraw-Hill (1965).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = [
    "free_particle_classical_path",
    "free_particle_classical_action",
    "harmonic_oscillator_classical_path",
    "harmonic_oscillator_classical_action",
    "discretized_action",
    "sample_random_paths",
    "feynman_phasor_partial_sums",
    "build_phasor_diagram",
]


def _check_not_conjugate(omega: float, T: float) -> None:
    if np.isclose(np.sin(omega * T), 0.0, atol=1e-9):
        raise ValueError(
            f"omega*T = {omega * T:.6g} is a multiple of pi: the two endpoints are "
            "conjugate points of the harmonic oscillator, where the classical path "
            "connecting them is non-unique (or does not exist). Choose a T not equal "
            "to n*pi/omega for integer n."
        )


def free_particle_classical_path(x0: float, xf: float, T: float, m: float, n_slices: int):
    r"""Exact classical trajectory of a free particle between fixed endpoints.

    With :math:`V=0` the Euler-Lagrange equation :math:`\ddot x=0` has the
    unique solution connecting :math:`(x_0,0)` and :math:`(x_f,T)`: the
    straight line :math:`x_{\rm cl}(t) = x_0 + (x_f-x_0)\,t/T`.

    Parameters
    ----------
    x0 : float
        Initial position, at :math:`t=0`.
    xf : float
        Final position, at :math:`t=T`.
    T : float
        Total elapsed time (must be positive).
    m : float
        Particle mass (unused for the free-particle trajectory itself,
        kept for signature symmetry with :func:`discretized_action` and
        the harmonic-oscillator counterpart).
    n_slices : int
        Number of time slices :math:`N`; the returned arrays have
        :math:`N+1` points.

    Returns
    -------
    t_array : ndarray, shape (n_slices + 1,)
        Uniformly spaced times over :math:`[0,T]`.
    x_array : ndarray, shape (n_slices + 1,)
        Classical trajectory :math:`x_{\rm cl}(t)`.

    Raises
    ------
    ValueError
        If ``T <= 0`` or ``n_slices < 1``.

    Examples
    --------
    >>> t, x = free_particle_classical_path(x0=0.0, xf=2.0, T=1.0, m=1.0, n_slices=4)
    >>> x
    array([0. , 0.5, 1. , 1.5, 2. ])
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")
    if n_slices < 1:
        raise ValueError(f"n_slices must be >= 1, got {n_slices}.")
    t_array = np.linspace(0.0, T, n_slices + 1)
    x_array = x0 + (xf - x0) * t_array / T
    return t_array, x_array


def free_particle_classical_action(x0: float, xf: float, T: float, m: float) -> float:
    r"""Exact classical (Hamilton's principal function) action of a free particle.

    .. math::

        S_{\rm cl} = \frac{m(x_f-x_0)^2}{2T}.

    Parameters
    ----------
    x0, xf : float
        Initial and final positions.
    T : float
        Total elapsed time (must be positive).
    m : float
        Particle mass.

    Returns
    -------
    float
        The classical action :math:`S_{\rm cl}`.

    Raises
    ------
    ValueError
        If ``T <= 0``.

    Examples
    --------
    >>> free_particle_classical_action(x0=0.0, xf=2.0, T=1.0, m=1.0)
    2.0
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")
    return m * (xf - x0) ** 2 / (2.0 * T)


def harmonic_oscillator_classical_path(x0: float, xf: float, T: float, m: float, omega: float, n_slices: int):
    r"""Exact classical trajectory of a harmonic oscillator between fixed endpoints.

    The Euler-Lagrange equation :math:`\ddot x = -\omega^2 x` has the
    unique solution matching both endpoints

    .. math::

        x_{\rm cl}(t) = \frac{x_0\sin(\omega(T-t)) + x_f\sin(\omega t)}
        {\sin(\omega T)},

    valid as long as :math:`\sin(\omega T)\neq 0`; at a *conjugate point*
    (:math:`\omega T` an integer multiple of :math:`\pi`) every path
    through both endpoints has the same action to first order and the
    classical path is not unique, so this raises instead of silently
    dividing by zero.

    Parameters
    ----------
    x0, xf : float
        Initial and final positions.
    T : float
        Total elapsed time (must be positive).
    m : float
        Particle mass (unused for the trajectory itself, kept for
        signature symmetry with :func:`harmonic_oscillator_classical_action`).
    omega : float
        Angular frequency of the oscillator.
    n_slices : int
        Number of time slices :math:`N`; the returned arrays have
        :math:`N+1` points.

    Returns
    -------
    t_array : ndarray, shape (n_slices + 1,)
        Uniformly spaced times over :math:`[0,T]`.
    x_array : ndarray, shape (n_slices + 1,)
        Classical trajectory :math:`x_{\rm cl}(t)`.

    Raises
    ------
    ValueError
        If ``T <= 0``, ``n_slices < 1``, or ``omega*T`` is (numerically) a
        multiple of :math:`\pi` (a conjugate point).

    Examples
    --------
    >>> t, x = harmonic_oscillator_classical_path(x0=0.0, xf=1.0, T=1.0, m=1.0, omega=1.0, n_slices=4)
    >>> bool(np.isclose(x[0], 0.0)) and bool(np.isclose(x[-1], 1.0))
    True
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")
    if n_slices < 1:
        raise ValueError(f"n_slices must be >= 1, got {n_slices}.")
    _check_not_conjugate(omega, T)
    t_array = np.linspace(0.0, T, n_slices + 1)
    x_array = (x0 * np.sin(omega * (T - t_array)) + xf * np.sin(omega * t_array)) / np.sin(omega * T)
    return t_array, x_array


def harmonic_oscillator_classical_action(x0: float, xf: float, T: float, m: float, omega: float) -> float:
    r"""Exact classical (Hamilton's principal function) action of a harmonic oscillator.

    .. math::

        S_{\rm cl} = \frac{m\omega}{2\sin(\omega T)}
        \left[(x_0^2+x_f^2)\cos(\omega T) - 2x_0 x_f\right].

    Parameters
    ----------
    x0, xf : float
        Initial and final positions.
    T : float
        Total elapsed time (must be positive).
    m : float
        Particle mass.
    omega : float
        Angular frequency of the oscillator.

    Returns
    -------
    float
        The classical action :math:`S_{\rm cl}`.

    Raises
    ------
    ValueError
        If ``T <= 0`` or ``omega*T`` is (numerically) a multiple of
        :math:`\pi` (a conjugate point; see
        :func:`harmonic_oscillator_classical_path`).

    Examples
    --------
    >>> round(harmonic_oscillator_classical_action(x0=0.0, xf=0.0, T=1.0, m=1.0, omega=0.5), 6)
    0.0
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")
    _check_not_conjugate(omega, T)
    return float((m * omega / (2.0 * np.sin(omega * T))) * ((x0**2 + xf**2) * np.cos(omega * T) - 2.0 * x0 * xf))


def discretized_action(x_array: np.ndarray, dt: float, m: float, V: Callable[[np.ndarray], np.ndarray]) -> float:
    r"""Midpoint-rule discretized action of a single time-sliced path.

    .. math::

        S[x] = \sum_{i=0}^{N-1}\left[\frac{1}{2}m
        \left(\frac{x_{i+1}-x_i}{\mathrm{d}t}\right)^2
        - V\!\left(\frac{x_i+x_{i+1}}{2}\right)\right]\mathrm{d}t.

    Parameters
    ----------
    x_array : ndarray, shape (N + 1,)
        Path positions :math:`x_0,\dots,x_N` at :math:`N+1` uniformly
        spaced times.
    dt : float
        Time-slice width.
    m : float
        Particle mass.
    V : callable
        Potential energy function ``V(x) -> array_like``, evaluated at
        the midpoint of each slice; must accept a NumPy array.

    Returns
    -------
    float
        The discretized action :math:`S[x]`.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    >>> V = lambda x: 0.0 * x
    >>> round(discretized_action(x, dt=0.25, m=1.0, V=V), 6)
    2.0
    """
    x_array = np.asarray(x_array, dtype=float)
    x_i = x_array[:-1]
    x_ip1 = x_array[1:]
    kinetic = 0.5 * m * ((x_ip1 - x_i) / dt) ** 2
    potential = V(0.5 * (x_i + x_ip1))
    return float(np.sum((kinetic - potential) * dt))


def sample_random_paths(
    x0: float,
    xf: float,
    T: float,
    n_slices: int,
    n_paths: int,
    sigma: float,
    n_modes: int = 8,
    seed: int | None = None,
) -> np.ndarray:
    r"""Sample random paths between fixed endpoints as sine-series Brownian bridges.

    Each path is the straight-line interpolation between the endpoints
    plus a random wiggle built from a discrete sine series,

    .. math::

        x_i = x_0 + (x_f-x_0)\frac{i}{N} + \sigma\sum_{k=1}^{K}
        \frac{a_k}{k}\sin\!\left(\frac{k\pi i}{N}\right),
        \qquad a_k\sim\mathcal{N}(0,1)\ \text{i.i.d. per path},

    where :math:`N` is ``n_slices`` and :math:`K` is ``n_modes``. Every
    term of the sine series vanishes at :math:`i=0` and :math:`i=N`
    identically, so every sampled path exactly satisfies the fixed
    endpoint conditions :math:`x_0` and :math:`x_f` regardless of the
    random coefficients drawn -- this is what makes it a valid "Brownian
    bridge" construction. Paths are built around the straight line (not
    around any particular classical trajectory), so the ensemble
    generically explores the space of continuous paths between the
    endpoints, and any special concentration of the phasor sum near the
    true classical path is a genuine result rather than built in by
    construction.

    Parameters
    ----------
    x0, xf : float
        Fixed initial and final positions shared by every sampled path.
    T : float
        Total elapsed time (must be positive).
    n_slices : int
        Number of time slices :math:`N`; each path has :math:`N+1` points.
    n_paths : int
        Number of random paths to draw.
    sigma : float
        Overall amplitude of the random wiggle.
    n_modes : int, default=8
        Number of sine modes :math:`K` used to build each path's wiggle;
        higher modes are automatically suppressed by the :math:`1/k`
        weighting, so this mainly controls the finest wiggle scale
        resolved.
    seed : int or None, optional
        Seed for the random number generator, for reproducibility.

    Returns
    -------
    ndarray, shape (n_paths, n_slices + 1)
        Sampled paths, each row a full path :math:`x_0,\dots,x_N`.

    Examples
    --------
    >>> paths = sample_random_paths(x0=0.0, xf=1.0, T=1.0, n_slices=10, n_paths=5, sigma=0.3, seed=0)
    >>> paths.shape
    (5, 11)
    >>> bool(np.allclose(paths[:, 0], 0.0)) and bool(np.allclose(paths[:, -1], 1.0))
    True
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")
    if n_slices < 1:
        raise ValueError(f"n_slices must be >= 1, got {n_slices}.")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}.")

    rng = np.random.default_rng(seed)
    N = n_slices
    i = np.arange(N + 1, dtype=float)
    straight = x0 + (xf - x0) * i / N

    k = np.arange(1, n_modes + 1, dtype=float)
    sin_matrix = np.sin(np.outer(k, np.pi * i / N))  # shape (n_modes, N+1)

    a = rng.standard_normal((n_paths, n_modes)) / k[None, :]
    noise = sigma * (a @ sin_matrix)  # shape (n_paths, N+1)

    return straight[None, :] + noise


def feynman_phasor_partial_sums(
    paths: np.ndarray,
    actions: np.ndarray,
    hbar: float,
    classical_action: float | None = None,
) -> np.ndarray:
    r"""Feynman's phasor ("arrow and stopwatch") partial sums, ordered closest-to-classical first.

    Sorts the given ensemble of paths by :math:`|S-S_{\rm cl}|` (ascending)
    and accumulates the running sum of unit phasors
    :math:`\exp(iS/\hbar)`, one path at a time, in that order. Because
    :math:`S` is stationary at the classical path, paths near it have
    nearly equal action and their phasors reinforce (the sum's magnitude
    grows quickly at first); paths far from it have rapidly varying
    action and their phasors point in essentially random directions, so
    later additions contribute a slowly-converging random walk instead
    of steady growth. This is exactly the data behind Feynman's
    arrow-and-stopwatch phasor-spiral diagram.

    Parameters
    ----------
    paths : ndarray, shape (n_paths, n_points)
        Ensemble of paths (only used to determine ``n_paths``; the
        ordering and summation depend only on ``actions``).
    actions : ndarray, shape (n_paths,)
        Discretized action of each path, e.g. from :func:`discretized_action`.
    hbar : float
        Reduced Planck constant (or an effective value thereof); must be
        nonzero.
    classical_action : float or None, optional
        Reference action :math:`S_{\rm cl}` used to sort paths by
        proximity. If None, the minimum action found in ``actions`` is
        used instead.

    Returns
    -------
    ndarray of complex, shape (n_paths + 1,)
        Partial sums :math:`\sum_{j\le n}\exp(iS_j/\hbar)`, starting from
        0 (index 0) up to the full sum over all paths (index ``n_paths``),
        accumulated in closest-to-classical-first order.

    Raises
    ------
    ValueError
        If ``hbar == 0`` or ``actions`` is empty.

    Examples
    --------
    >>> import numpy as np
    >>> actions = np.array([0.0, 0.1, -3.0])
    >>> paths = np.zeros((3, 2))
    >>> partial = feynman_phasor_partial_sums(paths, actions, hbar=1.0, classical_action=0.0)
    >>> partial.shape
    (4,)
    >>> bool(np.isclose(partial[0], 0.0))
    True
    >>> bool(np.isclose(partial[-1], np.sum(np.exp(1j * actions))))
    True
    """
    actions = np.asarray(actions, dtype=float)
    if actions.size == 0:
        raise ValueError("actions must be non-empty.")
    if hbar == 0:
        raise ValueError("hbar must be nonzero.")

    reference = float(np.min(actions)) if classical_action is None else float(classical_action)
    order = np.argsort(np.abs(actions - reference))
    terms = np.exp(1j * actions[order] / hbar)
    cumulative = np.cumsum(terms)
    return np.concatenate([[0.0 + 0.0j], cumulative])


def build_phasor_diagram(
    x0: float,
    xf: float,
    T: float,
    m: float,
    hbar: float,
    potential: str = "free",
    omega: float | None = None,
    n_slices: int = 60,
    n_paths: int = 400,
    sigma: float = 0.5,
    seed: int | None = None,
):
    r"""Build a complete Feynman phasor-spiral dataset for the free particle or harmonic oscillator.

    Ties together the classical trajectory/action, a random ensemble of
    candidate paths (:func:`sample_random_paths`), their discretized
    actions (:func:`discretized_action`), and the resulting phasor
    partial sums (:func:`feynman_phasor_partial_sums`) -- everything
    :func:`~physicskit.semiclassical.visualizers.path_integral.animate_feynman_phasor_spiral`
    needs to draw. The exact classical path is explicitly included as one
    member of the returned ensemble, since it is the single most
    important path in the whole construction.

    Parameters
    ----------
    x0, xf : float
        Fixed initial and final positions.
    T : float
        Total elapsed time (must be positive).
    m : float
        Particle mass.
    hbar : float
        Reduced Planck constant (or an effective value thereof); must be
        nonzero. Smaller ``hbar`` concentrates the phasor sum more
        tightly around the classical path.
    potential : {"free", "harmonic"}, default="free"
        Which exactly-solvable system to use.
    omega : float or None, optional
        Angular frequency; required if ``potential="harmonic"``.
    n_slices : int, default=60
        Number of time slices per path.
    n_paths : int, default=400
        Number of random paths to sample (in addition to the classical
        path itself).
    sigma : float, default=0.5
        Amplitude of the random path wiggles, passed to
        :func:`sample_random_paths`.
    seed : int or None, optional
        Seed for reproducible path sampling.

    Returns
    -------
    paths : ndarray, shape (n_paths + 1, n_slices + 1)
        The random ensemble with the classical path appended as the last row.
    actions : ndarray, shape (n_paths + 1,)
        Discretized action of every path in ``paths`` (including the
        classical path's discretized action, which need not exactly
        equal ``classical_action`` at finite ``n_slices``).
    classical_path : ndarray, shape (n_slices + 1,)
        The exact classical trajectory.
    classical_action : float
        The exact (closed-form) classical action.
    partial_sums : ndarray of complex, shape (n_paths + 2,)
        Phasor partial sums from :func:`feynman_phasor_partial_sums`,
        ordered closest-to-classical-action first.

    Raises
    ------
    ValueError
        If ``potential`` is not one of ``"free"``/``"harmonic"``, or if
        ``potential="harmonic"`` and ``omega`` is not given.

    Examples
    --------
    >>> paths, actions, x_cl, S_cl, partial = build_phasor_diagram(
    ...     x0=0.0, xf=1.0, T=1.0, m=1.0, hbar=0.05, potential="free",
    ...     n_slices=20, n_paths=50, sigma=0.3, seed=0)
    >>> paths.shape[0] == actions.shape[0] == 51
    True
    >>> partial.shape
    (52,)
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")

    if potential == "free":
        _, classical_path = free_particle_classical_path(x0, xf, T, m, n_slices)
        classical_action = free_particle_classical_action(x0, xf, T, m)
        V = lambda x: np.zeros_like(x)
    elif potential == "harmonic":
        if omega is None:
            raise ValueError('omega must be given when potential="harmonic".')
        _, classical_path = harmonic_oscillator_classical_path(x0, xf, T, m, omega, n_slices)
        classical_action = harmonic_oscillator_classical_action(x0, xf, T, m, omega)
        V = lambda x: 0.5 * m * omega**2 * x**2
    else:
        raise ValueError(f'potential must be "free" or "harmonic", got {potential!r}.')

    random_paths = sample_random_paths(x0, xf, T, n_slices, n_paths, sigma, seed=seed)
    paths = np.vstack([random_paths, classical_path[None, :]])

    dt = T / n_slices
    actions = np.array([discretized_action(p, dt, m, V) for p in paths])

    partial_sums = feynman_phasor_partial_sums(paths, actions, hbar, classical_action=classical_action)

    return paths, actions, classical_path, classical_action, partial_sums
