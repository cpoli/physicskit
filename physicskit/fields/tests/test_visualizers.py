"""Fast (no anim.save(), no file I/O) coverage of physicskit.fields.visualizers:
static plots and the ax=None/ax=<given> branch of every function, plus
building each animation and drawing a frame without ever writing a GIF
(see physicskit/fields/tests/test_field_animations.py for the real
save()-to-disk smoke tests, marked slow)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from physicskit.fields.quantum_fields import harmonic_trap_grid
from physicskit.fields.visualizers import (
    animate_density_2d,
    animate_field_1d,
    animate_field_2d,
    animate_flux_tube,
    plot_bec_density,
    plot_bec_phase,
    plot_field_1d,
    plot_poynting_field,
)


def test_plot_field_1d_with_and_without_label_and_given_ax():
    x = np.linspace(-10, 10, 50)
    u = np.sin(x)

    fig, ax = plot_field_1d(x, u)
    assert isinstance(fig, plt.Figure)
    assert not ax.get_legend()

    fig2, ax2 = plot_field_1d(x, u, label="u(x)")
    assert ax2.get_legend() is not None

    _, ax_given = plot_field_1d(x, u, ax=ax)
    assert ax_given is ax


def test_plot_poynting_field_given_ax():
    X, Y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    Sx, Sy = -Y, X
    fig, ax = plot_poynting_field(X, Y, Sx, Sy)
    assert isinstance(fig, plt.Figure)

    _, ax_given = plot_poynting_field(X, Y, Sx, Sy, ax=ax)
    assert ax_given is ax


def test_plot_bec_density_and_phase_given_ax():
    X, Y, _, _, _ = harmonic_trap_grid(16, 10.0)
    psi = np.exp(-0.5 * (X**2 + Y**2)).astype(complex)

    fig, ax = plot_bec_density(X, Y, psi)
    assert isinstance(fig, plt.Figure)
    _, ax_given = plot_bec_density(X, Y, psi, ax=ax)
    assert ax_given is ax

    fig2, ax2 = plot_bec_phase(X, Y, psi)
    assert isinstance(fig2, plt.Figure)
    _, ax2_given = plot_bec_phase(X, Y, psi, ax=ax2)
    assert ax2_given is ax2


def test_animate_field_1d_builds_and_draws_a_frame_given_ax():
    x = np.linspace(-10, 10, 30)
    frames = np.array([np.sin(x + phase) for phase in np.linspace(0, 1, 4)])
    times = np.linspace(0, 1, 4)

    fig, ax = plt.subplots()
    anim = animate_field_1d(x, frames, times, ax=ax)
    assert anim._fig is fig
    anim._draw_frame(1)
    assert ax.get_title() == f"t = {times[1]:.4g}"


def test_animate_field_2d_builds_and_draws_a_frame_given_ax():
    X, Y = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    frames = np.random.default_rng(0).normal(size=(3, 8, 8))

    fig, ax = plt.subplots()
    anim = animate_field_2d(X, Y, frames, ax=ax)
    assert anim._fig is fig
    anim._draw_frame(1)
    assert ax.get_title() == "frame 1"  # times=None branch


def test_animate_density_2d_builds_and_draws_a_frame_given_ax():
    frames = np.random.default_rng(0).random(size=(3, 8, 8))

    fig, ax = plt.subplots()
    anim = animate_density_2d(frames, ax=ax)
    assert anim._fig is fig
    anim._draw_frame(1)


def test_animate_flux_tube_builds_and_draws_a_frame_given_ax():
    fig, ax = plt.subplots()
    anim = animate_flux_tube((20, 10), dx=0.25, dy=0.25, separations=np.linspace(4.0, 8.0, 3), ax=ax)
    assert anim._fig is fig
    anim._draw_frame(1)
