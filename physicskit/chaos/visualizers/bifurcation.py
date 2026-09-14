"""Generic bifurcation diagrams: sweep a parameter, plot the attractor cross-section.

A bifurcation diagram sweeps one parameter of a system and, for each value,
plots the "surviving" long-term values of some observable after discarding
an initial transient: a single point for a stable fixed point, a handful of
points for a period-``k`` cycle, and a dense band for chaos. The core
:func:`bifurcation_diagram` function is deliberately sampling-agnostic (it
just calls a user-supplied ``sample_fn(param) -> ndarray`` for each parameter
value), so it works equally well for:

- **Discrete maps** (:class:`physicskit.chaos.core.base_system.DiscreteMap`), where
  the natural "sample" is the tail of the map's own iterated trajectory --
  see :func:`map_bifurcation_sampler`.
- **Continuous, periodically-forced flows**
  (:class:`physicskit.chaos.core.base_system.DynamicalSystem`), where the natural
  "sample" is a *stroboscopic* Poincare section: the state sampled once per
  forcing period, which turns the continuous flow into an effective discrete
  map -- see :func:`stroboscopic_bifurcation_sampler`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm

from physicskit.chaos.core.base_system import DiscreteMap, DynamicalSystem
from physicskit.chaos.visualizers import theme

#: A function mapping one parameter value to the array of long-term
#: "surviving" observable values sampled for that parameter.
SampleFunc = Callable[[float], NDArray[np.float64]]


def bifurcation_diagram(
    param_values: ArrayLike,
    sample_fn: SampleFunc,
    n_jobs: int = 1,
    show_progress: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate raw bifurcation-diagram data by sweeping a parameter.

    Parameters
    ----------
    param_values : array_like of float, shape (n_params,)
        Parameter values to sweep.
    sample_fn : callable
        Function ``sample_fn(param) -> ndarray`` returning the array of
        long-term ("surviving", post-transient) observable values for one
        parameter value. See :func:`map_bifurcation_sampler` and
        :func:`stroboscopic_bifurcation_sampler` for ready-made samplers.
    n_jobs : int, default 1
        Number of worker threads to evaluate `sample_fn` with, via
        :class:`concurrent.futures.ThreadPoolExecutor`. Each parameter value
        is fully independent, so this is embarrassingly parallel -- but the
        speedup depends on `sample_fn` releasing the GIL while it runs:
        :func:`stroboscopic_bifurcation_sampler` does (via
        :func:`scipy.integrate.solve_ivp`), and so does
        :func:`map_bifurcation_sampler` (the underlying Numba kernels are
        compiled with ``nogil=True``); an arbitrary pure-Python `sample_fn`
        will not see a speedup. ``1`` (the default) runs sequentially with no
        thread-pool overhead.
    show_progress : bool, default False
        Display a `tqdm` progress bar over parameter values.

    Returns
    -------
    params : ndarray of float, shape (N,)
        Parameter value repeated once per sample.
    samples : ndarray of float, shape (N,)
        The concatenated samples from every parameter value, aligned with
        `params` (``N`` is the total sample count summed across all
        parameter values, not necessarily ``n_params`` times a fixed count).
    """
    params = np.atleast_1d(np.asarray(param_values, dtype=np.float64))

    if n_jobs == 1:
        results = [sample_fn(float(p)) for p in tqdm(params, disable=not show_progress, desc="sweeping")]
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            results = list(
                tqdm(
                    pool.map(lambda p: sample_fn(float(p)), params),
                    total=params.shape[0],
                    disable=not show_progress,
                    desc="sweeping",
                )
            )

    all_params = []
    all_samples = []
    for p, raw_samples in zip(params, results):
        samples = np.atleast_1d(np.asarray(raw_samples, dtype=np.float64))
        all_params.append(np.full(samples.shape[0], p, dtype=np.float64))
        all_samples.append(samples)
    return np.concatenate(all_params), np.concatenate(all_samples)


def map_bifurcation_sampler(
    make_map: Callable[[float], DiscreteMap],
    state0: ArrayLike,
    n_transient: int,
    n_keep: int,
    component: int = 0,
) -> SampleFunc:
    """Build a :func:`bifurcation_diagram` sampler for a :class:`~physicskit.chaos.core.base_system.DiscreteMap`.

    Parameters
    ----------
    make_map : callable
        Function ``make_map(param) -> DiscreteMap`` constructing a fresh map
        instance for one parameter value.
    state0 : array_like of float, shape (dim,)
        Initial state to iterate from, for every parameter value.
    n_transient : int
        Number of initial iterations to discard.
    n_keep : int
        Number of iterations to keep (and return) after the transient.
    component : int, default 0
        Index of the state component to sample.

    Returns
    -------
    callable
        A function ``sample_fn(param) -> ndarray`` of shape ``(n_keep,)``,
        suitable as the `sample_fn` argument to :func:`bifurcation_diagram`.
    """

    def sample_fn(param: float) -> NDArray[np.float64]:
        system = make_map(param)
        trajectory = system.trajectory(np.asarray(state0, dtype=np.float64), n_transient + n_keep)
        return trajectory[n_transient:, component]

    return sample_fn


def stroboscopic_bifurcation_sampler(
    make_system: Callable[[float], DynamicalSystem],
    state0: ArrayLike,
    sample_period: float,
    n_transient_periods: int,
    n_keep_periods: int,
    component: int = 0,
    dt: float = 0.01,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> SampleFunc:
    """Build a :func:`bifurcation_diagram` sampler for a periodically-forced continuous flow.

    Integrates the system and records its state stroboscopically -- once
    every `sample_period` (typically the forcing period of a driven
    oscillator such as :class:`physicskit.chaos.systems.continuous.Duffing`) -- which
    turns the continuous flow into an effective discrete map for bifurcation
    purposes. Uses :func:`scipy.integrate.solve_ivp` against the system's
    plain-Python :meth:`~physicskit.chaos.core.base_system.DynamicalSystem.rhs`
    (rather than the Numba-jitted low-level integrators, which require a
    jitted right-hand side that arbitrary systems may not expose), evaluated
    exactly at the stroboscopic sample times.

    Parameters
    ----------
    make_system : callable
        Function ``make_system(param) -> DynamicalSystem`` constructing a
        fresh system instance for one parameter value.
    state0 : array_like of float, shape (dim,)
        Initial state to integrate from, for every parameter value.
    sample_period : float
        Time between stroboscopic samples (typically the forcing period).
    n_transient_periods : int
        Number of initial stroboscopic samples to discard.
    n_keep_periods : int
        Number of stroboscopic samples to keep (and return) after the
        transient.
    component : int, default 0
        Index of the state component to sample.
    dt : float, default 0.01
        Approximate internal integration step size passed to `solve_ivp` as
        `max_step` (the adaptive solver still lands exactly on each
        stroboscopic sample time via `t_eval`). A parameter sweep calls this
        once per parameter value, so loosening `dt`, `rtol`, and `atol` can
        matter a great deal for the total sweep's running time.
    rtol, atol : float, default 1e-9, 1e-11
        Relative and absolute error tolerances passed to `solve_ivp`.

    Returns
    -------
    callable
        A function ``sample_fn(param) -> ndarray`` of shape
        ``(n_keep_periods,)``, suitable as the `sample_fn` argument to
        :func:`bifurcation_diagram`.
    """
    n_periods = n_transient_periods + n_keep_periods
    sample_times = sample_period * np.arange(1, n_periods + 1)

    def sample_fn(param: float) -> NDArray[np.float64]:
        system = make_system(param)

        def _rhs(t: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
            return system.rhs(state, t)

        solution = solve_ivp(
            _rhs,
            (0.0, sample_times[-1]),
            np.asarray(state0, dtype=np.float64),
            t_eval=sample_times,
            max_step=dt,
            rtol=rtol,
            atol=atol,
        )
        stroboscopic = solution.y[component]
        return np.asarray(stroboscopic[n_transient_periods:], dtype=np.float64)

    return sample_fn


def plot_bifurcation_diagram(
    param_values: ArrayLike,
    sample_fn: SampleFunc,
    ax: Axes | None = None,
    n_jobs: int = 1,
    show_progress: bool = False,
    **scatter_kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a bifurcation diagram.

    Parameters
    ----------
    param_values : array_like of float, shape (n_params,)
        Parameter values to sweep.
    sample_fn : callable
        Function ``sample_fn(param) -> ndarray`` returning the long-term
        surviving observable values for one parameter value; see
        :func:`map_bifurcation_sampler` / :func:`stroboscopic_bifurcation_sampler`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure and axes are created if omitted.
    n_jobs : int, default 1
        See :func:`bifurcation_diagram`.
    show_progress : bool, default False
        Display a `tqdm` progress bar over parameter values.
    **scatter_kwargs
        Additional keyword arguments forwarded to ``ax.plot``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    params, samples = bifurcation_diagram(param_values, sample_fn, n_jobs=n_jobs, show_progress=show_progress)
    param_array = np.asarray(param_values, dtype=np.float64)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = cast(Figure, ax.figure)

    kwargs: dict[str, Any] = {
        "marker": ",",
        "linestyle": "none",
        "color": theme.STRUCTURE,
        "alpha": 0.6,
    }
    kwargs.update(scatter_kwargs)
    ax.plot(params, samples, **kwargs)
    ax.set_xlim(float(np.min(param_array)), float(np.max(param_array)))
    return fig, ax
