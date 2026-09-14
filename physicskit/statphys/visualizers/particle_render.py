"""Visualizations of the Lennard-Jones gas: particle snapshots and velocity statistics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from physicskit.statphys.utils.thermodynamics import maxwell_boltzmann_speed_pdf

__all__ = ["animate_gas", "plot_particle_snapshot", "plot_velocity_histogram"]


def plot_particle_snapshot(gas, ax=None, show_velocities=False, color_by_speed=True):
    """Draw a single snapshot of the particle gas's positions.

    Parameters
    ----------
    gas : LennardJonesGas
        Simulation instance to render.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created if not given.
    show_velocities : bool, default=False
        Whether to overlay velocity vectors as arrows.
    color_by_speed : bool, default=True
        Whether to color particles by their instantaneous speed.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    speeds = gas.speeds()
    c = speeds if color_by_speed else "steelblue"
    scatter = ax.scatter(gas.positions[:, 0], gas.positions[:, 1], c=c, cmap="plasma", s=25)
    if color_by_speed:
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="speed")
    if show_velocities:
        ax.quiver(
            gas.positions[:, 0],
            gas.positions[:, 1],
            gas.velocities[:, 0],
            gas.velocities[:, 1],
            color="black",
            alpha=0.5,
            scale_units="xy",
            angles="xy",
            scale=1.0,
        )
    ax.set_xlim(0, gas.box_size)
    ax.set_ylim(0, gas.box_size)
    ax.set_aspect("equal")
    ax.set_title(f"t = {gas.time:.2f}, T = {gas.temperature():.2f}")
    return ax


def plot_velocity_histogram(gas, ax=None, bins=40, show_theory=True):
    """Plot the current speed histogram against the theoretical Maxwell-Boltzmann curve.

    Parameters
    ----------
    gas : LennardJonesGas
        Simulation instance to render.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.
    bins : int, default=40
        Histogram bin count.
    show_theory : bool, default=True
        Whether to overlay the equilibrium Maxwell-Boltzmann density at the
        gas's instantaneous kinetic temperature.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    speeds = gas.speeds()
    ax.hist(speeds, bins=bins, density=True, alpha=0.6, color="steelblue", label="simulation")
    if show_theory:
        v = np.linspace(0, speeds.max() * 1.2, 200)
        pdf = maxwell_boltzmann_speed_pdf(v, gas.temperature(), mass=gas.mass, kB=gas.kB, dim=2)
        ax.plot(v, pdf, color="crimson", linewidth=2, label="Maxwell-Boltzmann")
    ax.set_xlabel("speed")
    ax.set_ylabel("probability density")
    ax.legend()
    return ax


def animate_gas(gas, n_frames=200, steps_per_frame=5):
    """Animate the particle gas evolving under Velocity Verlet dynamics.

    Parameters
    ----------
    gas : LennardJonesGas
        Simulation instance to animate; advanced in place.
    n_frames : int, default=200
        Number of animation frames.
    steps_per_frame : int, default=5
        Integration steps performed between consecutive frames.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    scatter = ax.scatter(gas.positions[:, 0], gas.positions[:, 1], c=gas.speeds(), cmap="plasma", s=25)
    ax.set_xlim(0, gas.box_size)
    ax.set_ylim(0, gas.box_size)
    ax.set_aspect("equal")
    title = ax.set_title(f"t = {gas.time:.2f}")

    def update(_frame):
        gas.step(steps_per_frame)
        scatter.set_offsets(gas.positions)
        scatter.set_array(gas.speeds())
        title.set_text(f"t = {gas.time:.2f}, T = {gas.temperature():.2f}")
        return scatter, title

    return FuncAnimation(fig, update, frames=n_frames, blit=False, interval=40)
