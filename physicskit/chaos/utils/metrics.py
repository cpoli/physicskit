"""Metrics for quantifying chaos: Lyapunov exponents, energy drift, phase-volume expansion."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from physicskit.chaos.core.base_system import DiscreteMap, DynamicalSystem


def lyapunov_exponent_from_divergence(t: NDArray[np.float64], delta: NDArray[np.float64]) -> float:
    """Estimate the largest Lyapunov exponent from a trajectory-separation curve.

    The estimate is the slope of ``ln(delta)`` vs. `t`, fit by ordinary least
    squares.

    Parameters
    ----------
    t : ndarray of float, shape (n,)
        Time points.
    delta : ndarray of float, shape (n,)
        The (positive) separation between two initially nearby trajectories
        at each time in `t`, e.g. as produced by
        :func:`physicskit.chaos.visualizers.divergence.trajectory_divergence`.

    Returns
    -------
    float
        Estimated largest Lyapunov exponent ``lambda_max``.
    """
    t = np.asarray(t, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    mask = delta > 0
    slope, _ = np.polyfit(t[mask], np.log(delta[mask]), 1)
    return float(slope)


def numerical_jacobian(system: DynamicalSystem, state: NDArray[np.float64], t: float, eps: float = 1e-6) -> NDArray[np.float64]:
    """Central-difference Jacobian of a dynamical system's vector field.

    Parameters
    ----------
    system : DynamicalSystem
        System whose ``rhs(state, t)`` is to be differentiated.
    state : ndarray of float, shape (dim,)
        State at which to evaluate the Jacobian.
    t : float
        Time at which to evaluate the Jacobian.
    eps : float, default 1e-6
        Finite-difference step size.

    Returns
    -------
    ndarray of float, shape (dim, dim)
        Jacobian matrix ``d(rhs)/d(state)`` at ``(state, t)``.
    """
    state = np.asarray(state, dtype=np.float64)
    dim = state.shape[0]
    jac = np.empty((dim, dim))
    for j in range(dim):
        perturb = np.zeros(dim)
        perturb[j] = eps
        f_plus = system.rhs(state + perturb, t)
        f_minus = system.rhs(state - perturb, t)
        jac[:, j] = (f_plus - f_minus) / (2.0 * eps)
    return jac


def benettin_lyapunov_spectrum(
    system: DynamicalSystem,
    state0: NDArray[np.float64],
    dt: float = 0.01,
    n_steps: int = 5000,
    n_transient: int = 500,
) -> NDArray[np.float64]:
    """Estimate the full Lyapunov spectrum via the Benettin QR method.

    Integrates the reference trajectory and an orthonormal tangent-space
    frame *simultaneously* with matching-order RK4 (the frame's variational
    equation ``dY/dt = J(x(t)) Y`` is evaluated, via the numerical Jacobian,
    at each of RK4's four substages, exactly as the state itself is),
    periodically re-orthonormalizing the frame with QR decomposition and
    accumulating the log-growth of each axis. Integrating the frame at only
    1st-order (e.g. forward Euler) while the reference trajectory uses RK4
    would bias the smallest-magnitude exponents (whose true value is often
    near zero) by an :math:`O(\\mathrm{dt})` error that does not shrink
    merely by integrating for longer -- only by refining `dt` itself.

    Parameters
    ----------
    system : DynamicalSystem
        System whose Lyapunov spectrum is to be estimated.
    state0 : array_like of float, shape (dim,)
        Initial state of the reference trajectory.
    dt : float, default 0.01
        Integration step size.
    n_steps : int, default 5000
        Total number of integration steps.
    n_transient : int, default 500
        Number of initial steps discarded before accumulating exponents, to
        let the frame align with the attractor.

    Returns
    -------
    ndarray of float, shape (dim,)
        Estimated Lyapunov exponents, in descending order.
    """
    state = np.asarray(state0, dtype=np.float64).copy()
    dim = state.shape[0]
    frame = np.eye(dim)
    log_sums = np.zeros(dim)
    t = 0.0
    n_accum = 0

    for i in range(n_steps):
        k1_x = system.rhs(state, t)
        k1_f = numerical_jacobian(system, state, t) @ frame

        state_2 = state + 0.5 * dt * k1_x
        t_2 = t + 0.5 * dt
        k2_x = system.rhs(state_2, t_2)
        k2_f = numerical_jacobian(system, state_2, t_2) @ (frame + 0.5 * dt * k1_f)

        state_3 = state + 0.5 * dt * k2_x
        k3_x = system.rhs(state_3, t_2)
        k3_f = numerical_jacobian(system, state_3, t_2) @ (frame + 0.5 * dt * k2_f)

        state_4 = state + dt * k3_x
        t_4 = t + dt
        k4_x = system.rhs(state_4, t_4)
        k4_f = numerical_jacobian(system, state_4, t_4) @ (frame + dt * k3_f)

        state_next = state + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        frame_next = frame + (dt / 6.0) * (k1_f + 2 * k2_f + 2 * k3_f + k4_f)

        q, r = np.linalg.qr(frame_next)
        signs = np.sign(np.diag(r))
        signs[signs == 0] = 1.0
        q = q * signs
        r = (r.T * signs).T
        frame = q

        if i >= n_transient:
            log_sums += np.log(np.abs(np.diag(r)))
            n_accum += 1

        state = state_next
        t += dt

    elapsed = n_accum * dt
    return log_sums / elapsed


def numerical_jacobian_map(dmap: DiscreteMap, state: NDArray[np.float64], eps: float = 1e-6) -> NDArray[np.float64]:
    """Central-difference Jacobian of a discrete map's step function.

    Parameters
    ----------
    dmap : DiscreteMap
        Map whose ``step(state)`` is to be differentiated.
    state : ndarray of float, shape (dim,)
        State at which to evaluate the Jacobian.
    eps : float, default 1e-6
        Finite-difference step size.

    Returns
    -------
    ndarray of float, shape (dim, dim)
        Jacobian matrix ``d(step)/d(state)`` at `state`.
    """
    state = np.asarray(state, dtype=np.float64)
    dim = state.shape[0]
    jac = np.empty((dim, dim))
    for j in range(dim):
        perturb = np.zeros(dim)
        perturb[j] = eps
        f_plus = dmap.step(state + perturb)
        f_minus = dmap.step(state - perturb)
        jac[:, j] = (f_plus - f_minus) / (2.0 * eps)
    return jac


def map_lyapunov_spectrum(
    dmap: DiscreteMap,
    state0: NDArray[np.float64],
    n_iter: int = 5000,
    n_transient: int = 500,
) -> NDArray[np.float64]:
    """Estimate the full Lyapunov spectrum of a discrete map via the Benettin QR method.

    The discrete-map analog of :func:`benettin_lyapunov_spectrum`: co-evolves
    an orthonormal frame under the linearized (numerically-Jacobian'd) map
    alongside the reference orbit, periodically re-orthonormalizing with QR
    decomposition and accumulating the log-growth of each axis. Unlike the
    continuous-flow version, there is no `dt` to divide by -- exponents come
    out directly in units of "per iteration".

    Parameters
    ----------
    dmap : DiscreteMap
        Map whose Lyapunov spectrum is to be estimated.
    state0 : array_like of float, shape (dim,)
        Initial state of the reference orbit.
    n_iter : int, default 5000
        Total number of map iterations.
    n_transient : int, default 500
        Number of initial iterations discarded before accumulating
        exponents, to let the frame align with the attractor.

    Returns
    -------
    ndarray of float, shape (dim,)
        Estimated Lyapunov exponents, in descending order, in units of "per
        iteration".
    """
    state = np.asarray(state0, dtype=np.float64).copy()
    dim = state.shape[0]
    frame = np.eye(dim)
    log_sums = np.zeros(dim)
    n_accum = 0

    for i in range(n_iter):
        jac = numerical_jacobian_map(dmap, state)
        frame = jac @ frame
        state = dmap.step(state)

        q, r = np.linalg.qr(frame)
        signs = np.sign(np.diag(r))
        signs[signs == 0] = 1.0
        q = q * signs
        r = (r.T * signs).T
        frame = q

        if i >= n_transient:
            log_sums += np.log(np.abs(np.diag(r)))
            n_accum += 1

    return log_sums / n_accum


def energy_drift(
    times: NDArray[np.float64],
    states: NDArray[np.float64],
    energy_func: Callable[[NDArray[np.float64]], float],
) -> NDArray[np.float64]:
    """Relative energy drift along a trajectory.

    Parameters
    ----------
    times : ndarray of float, shape (n,)
        Time points (unused directly, kept for API symmetry with
        :func:`phase_volume_expansion`).
    states : ndarray of float, shape (n, dim)
        State at each time in `times`.
    energy_func : callable
        Function ``energy_func(state) -> float`` computing the conserved (or
        nominally conserved) quantity for a single state; useful for checking
        integrator fidelity (e.g.
        :meth:`physicskit.chaos.systems.continuous.DoublePendulum.energy`).

    Returns
    -------
    ndarray of float, shape (n,)
        Relative drift ``(E(t) - E(0)) / |E(0)|`` at each time in `times`
        (using an absolute, not relative, denominator when ``E(0)`` is
        numerically zero).
    """
    energies = np.array([energy_func(states[i]) for i in range(states.shape[0])])
    e0 = energies[0]
    denom = abs(e0) if abs(e0) > 1e-12 else 1.0
    return np.asarray((energies - e0) / denom, dtype=np.float64)


def phase_volume_expansion(system: DynamicalSystem, times: NDArray[np.float64], states: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cumulative log phase-space-volume expansion along a trajectory.

    Uses Liouville's theorem, ``d(log V)/dt = trace(Jacobian)``, integrated
    with the trapezoidal rule.

    Parameters
    ----------
    system : DynamicalSystem
        System whose phase-space divergence (``trace`` of the Jacobian) is
        evaluated along the trajectory.
    times : ndarray of float, shape (n,)
        Time points.
    states : ndarray of float, shape (n, dim)
        State at each time in `times`.

    Returns
    -------
    ndarray of float, shape (n,)
        Cumulative ``log(V(t) / V(0))`` at each time in `times`, starting at
        ``0.0``. Negative values indicate a dissipative (volume-contracting)
        system.
    """
    n = states.shape[0]
    divergence = np.empty(n)
    for i in range(n):
        jac = numerical_jacobian(system, states[i], times[i])
        divergence[i] = np.trace(jac)
    return np.concatenate(([0.0], np.cumsum(0.5 * (divergence[:-1] + divergence[1:]) * np.diff(times))))
