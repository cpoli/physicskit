"""Coverage for previously entirely untested plotting/interactive modules in
physicskit.relativity.visualizers: spacetime_3d, spacetime_diagrams,
wave_plots, interactive.interactive_shadow_image, and
shadow_render.animate_shadow_spin_sweep. All built/drawn directly without
GIF-saving, so fast except where noted @pytest.mark.slow (ray tracing)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from physicskit.relativity.chapters.gw_merger import BinaryMerger
from physicskit.relativity.visualizers.interactive import interactive_shadow_image
from physicskit.relativity.visualizers.shadow_render import animate_shadow_spin_sweep, render_black_hole_image
from physicskit.relativity.visualizers.spacetime_3d import flamm_paraboloid, plot_flamm_paraboloid
from physicskit.relativity.visualizers.spacetime_diagrams import (
    kruskal_coordinates,
    penrose_carter_coordinates,
    plot_kruskal_diagram,
    plot_penrose_diagram,
)
from physicskit.relativity.visualizers.wave_plots import animate_wave_ripple, plot_strain_waveform, plot_wave_ripple


def test_flamm_paraboloid_and_plot_given_and_default_ax():
    M = 1.0
    X, Y, Z = flamm_paraboloid(M, n_r=10, n_phi=10)
    assert X.shape == Y.shape == Z.shape == (10, 10)
    assert np.all(Z >= 0.0)

    ax0 = plot_flamm_paraboloid(M)
    assert isinstance(ax0, Axes)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax_given = plot_flamm_paraboloid(M, ax=ax)
    assert ax_given is ax


def test_kruskal_and_penrose_coordinates_map_horizon_to_light_cone():
    M = 1.0
    X, T = kruskal_coordinates(t=0.0, r=2.0 * M, M=M)
    assert X == pytest.approx(0.0, abs=1e-8)
    assert T == pytest.approx(0.0, abs=1e-8)

    space, time = penrose_carter_coordinates(t=0.0, r=2.0 * M, M=M)
    assert space == pytest.approx(0.0, abs=1e-8)
    assert time == pytest.approx(0.0, abs=1e-8)


def test_plot_kruskal_diagram_default_and_given_ax():
    ax0 = plot_kruskal_diagram(M=1.0)
    assert isinstance(ax0, Axes)

    fig, ax = plt.subplots()
    ax_given = plot_kruskal_diagram(M=1.0, ax=ax)
    assert ax_given is ax


def test_plot_penrose_diagram_default_and_given_ax():
    ax0 = plot_penrose_diagram(M=1.0)
    assert isinstance(ax0, Axes)

    fig, ax = plt.subplots()
    ax_given = plot_penrose_diagram(M=1.0, ax=ax)
    assert ax_given is ax


def test_plot_strain_waveform_default_ax_with_cross_polarization_and_merger_marker():
    t = np.linspace(-10, 10, 50)
    h_plus = np.sin(t)
    h_cross = np.cos(t)
    ax = plot_strain_waveform(t, h_plus, h_cross=h_cross, t_merger=0.0)
    assert isinstance(ax, Axes)
    assert len(ax.lines) == 3  # h_plus, h_cross, merger marker


def test_plot_wave_ripple_default_and_given_ax():
    merger = BinaryMerger(m1=30.0, m2=30.0, distance=1000.0)
    ax0 = plot_wave_ripple(merger, t=0.0, t_merger=0.0, grid_size=20)
    assert isinstance(ax0, Axes)

    fig, ax = plt.subplots()
    ax_given = plot_wave_ripple(merger, t=0.0, t_merger=0.0, ax=ax, grid_size=20)
    assert ax_given is ax


def test_animate_wave_ripple_builds_and_draws_a_frame():
    merger = BinaryMerger(m1=30.0, m2=30.0, distance=1000.0)
    t_values = np.linspace(-5.0, 5.0, 3)
    anim = animate_wave_ripple(merger, t_merger=0.0, t_values=t_values, grid_size=15)
    anim._draw_frame(1)
    assert anim._fig.axes[0].get_title() == f"t={t_values[1]:.1f}"


def test_interactive_shadow_image_returns_heatmap_figure():
    result = render_black_hole_image(M=1.0, ny=10, nx=10)
    fig = interactive_shadow_image(result)
    assert fig.data[0].type == "heatmap"


@pytest.mark.slow
def test_animate_shadow_spin_sweep_builds_and_draws_a_frame():
    a_values = np.array([0.0, 0.3])
    anim = animate_shadow_spin_sweep(M=1.0, a_values=a_values, ny=10, nx=10)
    anim._draw_frame(1)
    assert f"a = {a_values[1]:.2f}" in anim._fig.axes[0].get_title()


@pytest.mark.slow
def test_animate_shadow_spin_sweep_default_a_values():
    anim = animate_shadow_spin_sweep(M=1.0, ny=6, nx=6)
    anim._draw_frame(0)
    assert anim._fig.axes[0].get_title().startswith("a = 0.00")
