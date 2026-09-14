r"""Plotting helpers for Feynman's path-integral phasor ("arrow and stopwatch") construction."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

__all__ = ["animate_feynman_phasor_spiral", "plot_phasor_convergence"]


def animate_feynman_phasor_spiral(
    paths: np.ndarray,
    actions: np.ndarray,
    classical_path: np.ndarray,
    classical_action: float,
    hbar: float,
    t_array: np.ndarray,
    interval: int = 40,
) -> FuncAnimation:
    r"""Animate paths being added one at a time alongside their growing Feynman phasor sum.

    Two side-by-side panels update together, frame by frame, as paths are
    added in order of increasing :math:`|S-S_{\rm cl}|` (closest to the
    classical path first -- the same order used by
    :func:`~physicskit.semiclassical.core.path_integral.feynman_phasor_partial_sums`).
    The left panel shows each path in real space, :math:`x(t)`, faded by
    how far its action is from the classical action; the right panel
    shows the cumulative phasor sum :math:`\sum\exp(iS/\hbar)` in the
    complex plane, traced out as a connected "arrow chain" -- Feynman's
    own picture of how nearby-action paths reinforce while distant-action
    paths spin around and largely cancel.

    Parameters
    ----------
    paths : ndarray, shape (n_paths, n_points)
        Ensemble of paths, e.g. from
        :func:`~physicskit.semiclassical.core.path_integral.build_phasor_diagram`.
    actions : ndarray, shape (n_paths,)
        Discretized action of each path.
    classical_path : ndarray, shape (n_points,)
        The exact classical trajectory, drawn on top of every frame.
    classical_action : float
        The exact classical action, used both as the sort reference and
        as the reference for the color/alpha of each path.
    hbar : float
        Reduced Planck constant (or an effective value thereof) used in
        the phasors :math:`\exp(iS/\hbar)`.
    t_array : ndarray, shape (n_points,)
        Time grid shared by every path (and the classical path).
    interval : int, default=40
        Delay between animation frames, in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object; assign it to a variable to keep it alive,
        and display it with ``plt.show()`` or save it with
        ``anim.save(...)``.

    Examples
    --------
    >>> from physicskit.semiclassical.core.path_integral import build_phasor_diagram
    >>> paths, actions, x_cl, S_cl, _ = build_phasor_diagram(
    ...     x0=0.0, xf=1.0, T=1.0, m=1.0, hbar=0.1, potential="free",
    ...     n_slices=20, n_paths=30, sigma=0.3, seed=0)
    >>> t = np.linspace(0.0, 1.0, x_cl.shape[0])
    >>> anim = animate_feynman_phasor_spiral(paths, actions, x_cl, S_cl, hbar=0.1, t_array=t)
    >>> isinstance(anim, FuncAnimation)
    True
    """
    paths = np.asarray(paths)
    actions = np.asarray(actions)
    n_paths = paths.shape[0]

    order = np.argsort(np.abs(actions - classical_action))
    sorted_paths = paths[order]
    sorted_actions = actions[order]

    dist = np.abs(sorted_actions - classical_action)
    dist_max = dist.max() if dist.max() > 0 else 1.0
    closeness = 1.0 - dist / dist_max  # 1 = right at classical action, 0 = farthest

    phasors = np.exp(1j * sorted_actions / hbar)
    partial_sums = np.concatenate([[0.0 + 0.0j], np.cumsum(phasors)])

    fig, (ax_paths, ax_complex) = plt.subplots(1, 2, figsize=(12, 5.5))

    ax_paths.plot(t_array, classical_path, color="red", lw=2.5, zorder=5, label="classical path")
    ax_paths.set_xlim(t_array.min(), t_array.max())
    x_all = np.concatenate([paths.ravel(), classical_path.ravel()])
    pad = 0.1 * (x_all.max() - x_all.min() + 1e-12)
    ax_paths.set_ylim(x_all.min() - pad, x_all.max() + pad)
    ax_paths.set_xlabel("t")
    ax_paths.set_ylabel("x(t)")
    ax_paths.set_title("Sampled paths (closest to classical first)")
    ax_paths.legend(loc="upper left", fontsize=8)
    path_lines = []

    radius = np.max(np.abs(partial_sums)) * 1.15 + 1e-12
    ax_complex.set_xlim(-radius, radius)
    ax_complex.set_ylim(-radius, radius)
    ax_complex.set_xlabel("Re")
    ax_complex.set_ylabel("Im")
    ax_complex.set_title(r"Phasor sum $\sum e^{iS/\hbar}$")
    ax_complex.axhline(0, color="gray", lw=0.5)
    ax_complex.axvline(0, color="gray", lw=0.5)
    ax_complex.set_aspect("equal")
    (spiral_line,) = ax_complex.plot([], [], color="C0", lw=1.0)
    arrow = ax_complex.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="crimson", lw=2))

    def init():
        spiral_line.set_data([], [])
        arrow.xy = (0, 0)
        arrow.set_position((0, 0))
        return [spiral_line, arrow]

    def update(frame: int):
        for ln in path_lines:
            ln.remove()
        path_lines.clear()
        for j in range(frame + 1):
            c = closeness[j]
            color = (1.0, 0.35 * (1 - c), 0.0, 0.15 + 0.7 * c)
            (ln,) = ax_paths.plot(t_array, sorted_paths[j], color=color, lw=1.0, zorder=2)
            path_lines.append(ln)

        pts = partial_sums[: frame + 2]
        spiral_line.set_data(pts.real, pts.imag)
        arrow.set_position((0, 0))
        arrow.xy = (pts[-1].real, pts[-1].imag)
        ax_complex.set_title(f"Phasor sum after {frame + 1}/{n_paths} paths")
        return path_lines + [spiral_line, arrow]

    anim = FuncAnimation(fig, update, init_func=init, frames=n_paths, interval=interval, blit=False)
    return anim


def plot_phasor_convergence(partial_sums_by_hbar, hbar_values, ax=None):
    r"""Plot the running phasor-sum magnitude for several :math:`\hbar` values on shared axes.

    Shows the core physics claim quantitatively: as :math:`\hbar` shrinks,
    the growing partial sum's magnitude climbs faster and plateaus sooner
    (dominated by an ever-smaller neighborhood of the classical path),
    while larger :math:`\hbar` gives a partial sum that keeps wandering
    randomly as more distant paths are added.

    Parameters
    ----------
    partial_sums_by_hbar : sequence of ndarray of complex
        One partial-sum array per :math:`\hbar` value (e.g. from
        :func:`~physicskit.semiclassical.core.path_integral.feynman_phasor_partial_sums`
        or the last element of
        :func:`~physicskit.semiclassical.core.path_integral.build_phasor_diagram`'s
        return value), each of shape ``(n_paths + 1,)``.
    hbar_values : sequence of float
        The :math:`\hbar` value corresponding to each array in
        ``partial_sums_by_hbar`` (used only for the legend labels).
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> from physicskit.semiclassical.core.path_integral import build_phasor_diagram
    >>> sums, hbars = [], [0.5, 0.05]
    >>> for hbar in hbars:
    ...     _, _, _, _, partial = build_phasor_diagram(
    ...         x0=0.0, xf=1.0, T=1.0, m=1.0, hbar=hbar, potential="free",
    ...         n_slices=20, n_paths=60, sigma=0.3, seed=0)
    ...     sums.append(partial)
    >>> fig, ax = plot_phasor_convergence(sums, hbars)
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    for partial, hbar in zip(partial_sums_by_hbar, hbar_values):
        partial = np.asarray(partial)
        n = partial.shape[0] - 1
        frac = np.arange(n + 1) / n
        ax.plot(frac, np.abs(partial), label=rf"$\hbar={hbar:g}$")

    ax.set_xlabel("fraction of paths included (closest-to-classical first)")
    ax.set_ylabel(r"$\left|\sum e^{iS/\hbar}\right|$")
    ax.set_title("Phasor-sum concentration near the classical path")
    ax.legend(fontsize=8)
    return fig, ax
