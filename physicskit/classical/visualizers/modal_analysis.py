"""Live normal-mode energy diagnostics for lattice chains.

Wraps :meth:`physicskit.classical.systems.chains._ChainBase.modal_energies` in a bar
chart and a Matplotlib ``FuncAnimation`` that updates it frame-by-frame
-- the standard way to visualize the FPUT recurrence: energy initially
concentrated in a single low mode, spreading across many modes, and
then returning almost exactly to the starting mode.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

__all__ = ["plot_modal_energy_bars", "animate_modal_energies"]


def plot_modal_energy_bars(system, q: np.ndarray, p: np.ndarray, ax=None, **kwargs):
    """Static bar chart of E_k for each normal mode k at one instant.

    Parameters
    ----------
    system : object
        A lattice-chain system exposing ``modal_energies(q, p)``, e.g.
        :class:`~physicskit.classical.systems.chains.HarmonicChain` or
        :class:`~physicskit.classical.systems.chains.FPUTChain`.
    q, p : ndarray
        State to evaluate.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    **kwargs
        Forwarded to ``ax.bar``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    energies = system.modal_energies(q, p)
    modes = np.arange(1, len(energies) + 1)
    ax.bar(modes, energies, **kwargs)
    ax.set_xlabel("normal mode k")
    ax.set_ylabel("$E_k$")
    ax.set_title("Modal energy distribution")
    return ax


def animate_modal_energies(system, result, stride: int = 1, interval: int = 30, fig=None, ax=None):
    """Animate the modal-energy bar chart across a full trajectory.

    Parameters
    ----------
    system : object
        A lattice-chain system exposing ``modal_energies(q, p)``.
    result : physicskit.classical.core.base_system.SimulationResult
        Output of ``system.integrate(...)``.
    stride : int
        Render only every ``stride``-th recorded sample (for long runs).
    interval : int
        Delay between frames, in milliseconds.
    fig : matplotlib.figure.Figure, optional
        Figure to draw on; inferred from ``ax`` or created if omitted.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    if ax is None:
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.figure

    n_modes = result.q.shape[1]
    modes = np.arange(1, n_modes + 1)
    e0 = system.modal_energies(result.q[0], result.p[0])
    bars = ax.bar(modes, e0)
    ax.set_xlabel("normal mode k")
    ax.set_ylabel("$E_k$")
    y_max = max(1e-12, np.sum(e0)) * 1.1
    ax.set_ylim(0, y_max)
    title = ax.set_title(f"Modal energy at t = {result.t[0]:.3f}")

    frame_indices = np.arange(0, len(result.t), stride)

    def update(frame_i):
        idx = frame_indices[frame_i]
        energies = system.modal_energies(result.q[idx], result.p[idx])
        for bar, e in zip(bars, energies):
            bar.set_height(e)
        title.set_text(f"Modal energy at t = {result.t[idx]:.3f}")
        return list(bars) + [title]

    return FuncAnimation(fig, update, frames=len(frame_indices), interval=interval, blit=False)
