"""Live trajectory-separation tracking to visually demonstrate sensitivity to
initial conditions (a positive largest Lyapunov exponent)."""

from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from physicskit.chaos.core.base_system import BilliardSystem, DynamicalSystem
from physicskit.chaos.utils.metrics import lyapunov_exponent_from_divergence
from physicskit.chaos.visualizers import theme


def trajectory_divergence(
    system: DynamicalSystem,
    state0: ArrayLike | None = None,
    delta_0: float = 1e-8,
    t_max: float = 100.0,
    n_points: int = 2000,
    direction: ArrayLike | None = None,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate two nearby trajectories and track their separation over time.

    Parameters
    ----------
    system : DynamicalSystem
        System to integrate.
    state0 : array_like of float, shape (dim,), optional
        Initial state of the reference trajectory; defaults to
        ``system.initial_state()``.
    delta_0 : float, default 1e-8
        Initial separation between the reference and perturbed trajectories.
    t_max : float, default 100.0
        Total integration time.
    n_points : int, default 2000
        Number of evenly-spaced time points to evaluate.
    direction : array_like of float, shape (dim,), optional
        Direction of the initial perturbation; a random unit vector is used
        if omitted (normalized internally regardless).
    seed : int, optional
        Seed for the random direction generator, for reproducibility.

    Returns
    -------
    t : ndarray of float, shape (n_points,)
        Time points.
    delta_t : ndarray of float, shape (n_points,)
        Euclidean separation between the two trajectories at each time in `t`.
    """
    state0 = system.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=system.dim) if direction is None else np.asarray(direction, dtype=np.float64)
    direction = direction / float(np.linalg.norm(direction))
    state0_perturbed = state0 + delta_0 * direction

    t_eval = np.linspace(0.0, t_max, n_points)

    def _rhs(t: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        return system.rhs(state, t)

    sol1 = solve_ivp(_rhs, (0.0, t_max), state0, t_eval=t_eval, rtol=1e-9, atol=1e-11)
    sol2 = solve_ivp(_rhs, (0.0, t_max), state0_perturbed, t_eval=t_eval, rtol=1e-9, atol=1e-11)

    delta_t = np.asarray(np.linalg.norm(sol1.y - sol2.y, axis=0), dtype=np.float64)
    return sol1.t, delta_t


def plot_lyapunov_divergence(
    system: DynamicalSystem,
    delta_0: float = 1e-8,
    t_max: float = 100.0,
    n_points: int = 2000,
    state0: ArrayLike | None = None,
    fit_fraction: float = 0.5,
    ax: Axes | None = None,
    seed: int | None = None,
) -> tuple[Figure, Axes, float]:
    """Plot ``ln(delta_t / delta_0)`` vs. `t` for two nearby trajectories.

    A straight, positively-sloped line demonstrates exponential divergence
    (``lambda_max > 0``); the slope is estimated by a linear fit over the
    first `fit_fraction` of the time range, before the separation saturates
    at the attractor's scale.

    Parameters
    ----------
    system : DynamicalSystem
        System to integrate.
    delta_0 : float, default 1e-8
        Initial separation between the reference and perturbed trajectories.
    t_max : float, default 100.0
        Total integration time.
    n_points : int, default 2000
        Number of evenly-spaced time points to evaluate.
    state0 : array_like of float, shape (dim,), optional
        Initial state of the reference trajectory; defaults to
        ``system.initial_state()``.
    fit_fraction : float, default 0.5
        Fraction of the time range (from ``t=0``) used to fit the
        exponential-growth slope.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure and axes are created if omitted.
    seed : int, optional
        Seed for the random perturbation direction, for reproducibility.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    lambda_max : float
        Estimated largest Lyapunov exponent from the fitted slope.
    """
    t, delta_t = trajectory_divergence(system, state0=state0, delta_0=delta_0, t_max=t_max, n_points=n_points, seed=seed)
    log_ratio = np.log(np.clip(delta_t, 1e-300, None) / delta_0)

    n_fit = max(2, int(fit_fraction * len(t)))
    lam = lyapunov_exponent_from_divergence(t[:n_fit], delta_t[:n_fit] / delta_0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = cast(Figure, ax.figure)

    ax.plot(t, log_ratio, lw=1.2, color=theme.PRIMARY, label=r"$\ln(\delta_t/\delta_0)$")
    ax.plot(
        t[:n_fit],
        lam * t[:n_fit],
        "--",
        color=theme.MUTED,
        label=rf"fit: $\lambda_{{max}} \approx {lam:.3f}$",
    )
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\ln(\delta_t / \delta_0)$")
    ax.set_title(f"{system.__class__.__name__} trajectory divergence")
    ax.legend()
    return fig, ax, lam


def trajectory_ensemble(
    system: DynamicalSystem,
    state0: ArrayLike | None = None,
    n_members: int = 40,
    spread: float = 1e-4,
    dt: float = 0.01,
    n_steps: int = 10000,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Integrate an ensemble of trajectories launched from a small ball
    around a shared initial state.

    Visualizing the result (e.g. with :func:`plot_trajectory_ensemble`)
    makes sensitivity to initial conditions directly visible: a regular
    orbit's members stay bundled together for the whole run, while a
    chaotic orbit's members visibly fan out, in contrast to
    :func:`trajectory_divergence`, which only tracks a *single* pair's
    separation as a number.

    Parameters
    ----------
    system : DynamicalSystem
        System to integrate.
    state0 : array_like of float, shape (dim,), optional
        Center of the initial ball; defaults to ``system.initial_state()``.
    n_members : int, default 40
        Number of trajectories in the ensemble.
    spread : float, default 1e-4
        Radius of the ball each member's initial state is drawn from.
    dt : float, default 0.01
        Integration step size.
    n_steps : int, default 10000
        Number of integration steps per trajectory.
    seed : int, optional
        Seed for the random initial-state offsets, for reproducibility.

    Returns
    -------
    ndarray of float, shape (n_members, n_steps + 1, dim)
        Every member's full trajectory.
    """
    state0 = system.initial_state() if state0 is None else np.asarray(state0, dtype=np.float64)
    rng = np.random.default_rng(seed)

    all_states = np.empty((n_members, n_steps + 1, system.dim))
    for i in range(n_members):
        offset = rng.normal(size=system.dim)
        offset *= spread / float(np.linalg.norm(offset))
        _, states = system.trajectory(state0=state0 + offset, dt=dt, n_steps=n_steps)
        all_states[i] = states
    return all_states


def plot_trajectory_ensemble(
    system: DynamicalSystem,
    state0: ArrayLike | None = None,
    n_members: int = 40,
    spread: float = 1e-4,
    dt: float = 0.01,
    n_steps: int = 10000,
    x_index: int = 0,
    y_index: int = 1,
    seed: int | None = None,
    ax: Axes | None = None,
    **line_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot an ensemble of nearby trajectories overlaid on the same axes.

    Parameters
    ----------
    system : DynamicalSystem
        System to integrate.
    state0 : array_like of float, shape (dim,), optional
        Center of the initial ball; defaults to ``system.initial_state()``.
    n_members : int, default 40
        Number of trajectories in the ensemble.
    spread : float, default 1e-4
        Radius of the ball each member's initial state is drawn from.
    dt : float, default 0.01
        Integration step size.
    n_steps : int, default 10000
        Number of integration steps per trajectory.
    x_index, y_index : int, default 0, 1
        State indices to use as the plotted x/y coordinates.
    seed : int, optional
        Seed for the random initial-state offsets, for reproducibility.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure and axes are created if omitted.
    **line_kwargs
        Additional keyword arguments forwarded to every ``ax.plot`` call.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    all_states = trajectory_ensemble(system, state0=state0, n_members=n_members, spread=spread, dt=dt, n_steps=n_steps, seed=seed)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = cast(Figure, ax.figure)

    kwargs: dict[str, Any] = {"color": theme.PRIMARY, "lw": 0.5, "alpha": 0.5}
    kwargs.update(line_kwargs)
    for states in all_states:
        ax.plot(states[:, x_index], states[:, y_index], **kwargs)
    ax.set_title(f"{system.__class__.__name__} trajectory ensemble ({n_members} members, spread={spread:g})")
    return fig, ax


def billiard_trajectory_divergence(
    billiard: BilliardSystem,
    pos: ArrayLike,
    vel: ArrayLike,
    delta_0: float = 1e-8,
    n_bounces: int = 200,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Trace two nearby billiard rays and track their bounce-by-bounce separation.

    Unlike :func:`trajectory_divergence` (continuous flows, compared at
    matched *times*), two billiard trajectories are naturally compared at
    matched *bounce number*: the reference ray and a ray launched at a
    slightly perturbed angle both hit the boundary `n_bounces` times, and
    the Euclidean distance between their `k`-th hit points is tracked as `k`
    increases.

    Parameters
    ----------
    billiard : BilliardSystem
        Billiard to trace both rays within.
    pos : array_like of float, shape (2,)
        Shared initial position ``(x, y)``; must lie in the billiard's
        interior.
    vel : array_like of float, shape (2,)
        Initial velocity direction ``(vx, vy)`` of the reference ray;
        normalized internally.
    delta_0 : float, default 1e-8
        Initial angular perturbation (radians) applied to `vel` to launch
        the second ray.
    n_bounces : int, default 200
        Number of reflections to trace for each ray.

    Returns
    -------
    bounce : ndarray of int, shape (n_bounces,)
        Bounce index, ``1, 2, ..., n_bounces``.
    separation : ndarray of float, shape (n_bounces,)
        Euclidean distance between the two rays' hit points at each bounce.
    """
    vel_ref = np.asarray(vel, dtype=np.float64)
    vel_ref = vel_ref / np.linalg.norm(vel_ref)
    angle = float(np.arctan2(vel_ref[1], vel_ref[0])) + delta_0
    vel_perturbed = np.array([np.cos(angle), np.sin(angle)])

    result = billiard.simulate(pos, vel_ref, n_bounces)
    result_perturbed = billiard.simulate(pos, vel_perturbed, n_bounces)

    separation = np.hypot(result["x"] - result_perturbed["x"], result["y"] - result_perturbed["y"])
    bounce = np.arange(1, n_bounces + 1)
    return bounce, separation


def plot_billiard_divergence(
    billiard: BilliardSystem,
    pos: ArrayLike,
    vel: ArrayLike,
    delta_0: float = 1e-8,
    n_bounces: int = 200,
    fit_fraction: float = 0.5,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, float]:
    """Plot ``ln(separation / delta_0)`` vs. bounce number for two nearby billiard rays.

    A straight, positively-sloped line demonstrates exponential divergence
    per bounce; the slope is estimated by a linear fit over the first
    `fit_fraction` of the bounces, before the separation saturates at the
    table's scale.

    Parameters
    ----------
    billiard : BilliardSystem
        Billiard to trace both rays within.
    pos : array_like of float, shape (2,)
        Shared initial position ``(x, y)``; must lie in the billiard's
        interior.
    vel : array_like of float, shape (2,)
        Initial velocity direction ``(vx, vy)`` of the reference ray.
    delta_0 : float, default 1e-8
        Initial angular perturbation (radians) applied to `vel`.
    n_bounces : int, default 200
        Number of reflections to trace for each ray.
    fit_fraction : float, default 0.5
        Fraction of the bounces (from the first) used to fit the
        exponential-growth slope.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure and axes are created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    lambda_per_bounce : float
        Estimated divergence rate, in nats per bounce.
    """
    bounce, separation = billiard_trajectory_divergence(billiard, pos, vel, delta_0=delta_0, n_bounces=n_bounces)
    log_ratio = np.log(np.clip(separation, 1e-300, None) / delta_0)

    n_fit = max(2, int(fit_fraction * len(bounce)))
    lam = lyapunov_exponent_from_divergence(bounce[:n_fit].astype(np.float64), separation[:n_fit] / delta_0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = cast(Figure, ax.figure)

    ax.plot(bounce, log_ratio, lw=1.2, color=theme.PRIMARY, label=r"$\ln(\delta_k/\delta_0)$")
    ax.plot(
        bounce[:n_fit],
        lam * bounce[:n_fit],
        "--",
        color=theme.MUTED,
        label=rf"fit: $\lambda \approx {lam:.3f}$ / bounce",
    )
    ax.set_xlabel("bounce number k")
    ax.set_ylabel(r"$\ln(\delta_k / \delta_0)$")
    ax.set_title(f"{billiard.__class__.__name__} trajectory divergence")
    ax.legend()
    return fig, ax, lam
