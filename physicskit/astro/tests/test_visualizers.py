import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import PillowWriter
from matplotlib.collections import LineCollection

from physicskit.astro.cosmic_web import lagrangian_grid, random_displacement_potential
from physicskit.astro.galactic_dynamics import circular_velocity
from physicskit.astro.nbody import NBodySystem, figure_eight_initial_conditions
from physicskit.astro.stellar_dynamo import simulate_alpha_omega_dynamo, simulate_stellar_convection
from physicskit.astro.stellar_structure import lane_emden
from physicskit.astro.visualizers import (
    animate_dynamo_wave,
    animate_nbody_trajectories,
    animate_stellar_convection,
    animate_zeldovich_collapse,
    plot_dynamo_butterfly_diagram,
    plot_lane_emden,
    plot_nbody_trajectories,
    plot_rotation_curve,
    plot_zeldovich_snapshot,
)


@pytest.mark.slow
def test_animate_nbody_trajectories_figure_eight_saves_gif(tmp_path):
    positions, velocities, masses = figure_eight_initial_conditions()
    system = NBodySystem(positions, velocities, masses)
    history = system.simulate(dt=0.002, n_steps=300)

    anim = animate_nbody_trajectories(history, skip=5)
    out = tmp_path / "nbody_figure_eight.gif"
    anim.save(out, writer=PillowWriter(fps=10))
    assert out.exists() and out.stat().st_size > 0


def test_animate_nbody_trajectories_uses_one_collection_and_marker_per_body():
    history = np.random.default_rng(1).normal(size=(50, 3, 3))

    anim = animate_nbody_trajectories(history, title="N-body test")
    ax = anim._fig.axes[0]
    collections = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(collections) == 3
    assert len(ax.lines) == 3
    assert ax.get_title() == "N-body test"

    anim._draw_frame(0)  # a single point per body: the <2-point trail branch
    assert collections[0].get_segments() == []


def test_animate_nbody_trajectories_trail_fades_with_alpha():
    """The trail must actually fade: older segments more transparent than
    the most recent one."""
    history = np.cumsum(np.random.default_rng(2).normal(size=(100, 3, 3)), axis=0)

    anim = animate_nbody_trajectories(history, trail=100)
    anim._draw_frame(99)
    ax = anim._fig.axes[0]
    lc = next(c for c in ax.collections if isinstance(c, LineCollection))
    alphas = np.asarray(lc.get_colors())[:, 3]
    assert alphas[0] < alphas[-1]


def test_plot_nbody_trajectories_returns_one_line_per_body():
    # A synthetic array is enough here -- plot_nbody_trajectories only
    # cares about (n_steps+1, N, 3) shape, not real dynamics.
    history = np.random.default_rng(0).normal(size=(10, 3, 3))

    fig, ax = plot_nbody_trajectories(history)
    assert isinstance(fig, plt.Figure)
    assert len(ax.lines) == 3

    _, ax_given = plot_nbody_trajectories(history, ax=ax)
    assert ax_given is ax  # reuses the supplied axes rather than creating a new figure


def test_plot_lane_emden_returns_figure():
    xi, theta = lane_emden(1.5, n_points=50)
    fig, ax = plot_lane_emden(xi, theta)
    assert isinstance(fig, plt.Figure)
    assert len(ax.lines) == 2  # theta(xi) curve + the axhline(0) reference line
    assert len(ax.collections) == 1  # the surface-point scatter

    _, ax_given = plot_lane_emden(xi, theta, ax=ax)
    assert ax_given is ax


def test_plot_rotation_curve_with_and_without_observed_data():
    r = np.linspace(1.0, 10.0, 10)
    v_model = circular_velocity(r, lambda rr: 1.0)

    fig1, ax1 = plot_rotation_curve(r, v_model)
    assert isinstance(fig1, plt.Figure)
    assert len(ax1.collections) == 0  # no scatter without observed data

    fig2, ax2 = plot_rotation_curve(r, v_model, v_observed=v_model * 1.1)
    assert isinstance(fig2, plt.Figure)
    assert len(ax2.collections) == 1  # scatter added for observed data

    _, ax_given = plot_rotation_curve(r, v_model, ax=ax1)
    assert ax_given is ax1


def _small_zeldovich_field():
    q = lagrangian_grid(5, 1.0)
    k_vectors, amplitudes, phases = random_displacement_potential(4, k_min=2 * np.pi, k_max=6 * np.pi, amplitude_scale=0.02, seed=0)
    return q, k_vectors, amplitudes, phases


def test_animate_zeldovich_collapse_builds_and_draws_a_frame():
    q, k_vectors, amplitudes, phases = _small_zeldovich_field()
    D_values = np.linspace(0.0, 0.1, 3)
    anim = animate_zeldovich_collapse(q, D_values, k_vectors, amplitudes, phases)
    anim._draw_frame(1)
    assert anim._fig.axes[0].collections  # the density-colored scatter exists


def test_plot_zeldovich_snapshot_returns_figure():
    q, k_vectors, amplitudes, phases = _small_zeldovich_field()
    fig, ax = plot_zeldovich_snapshot(q, 0.05, k_vectors, amplitudes, phases)
    assert isinstance(fig, plt.Figure)
    assert ax.collections

    _, ax_given = plot_zeldovich_snapshot(q, 0.05, k_vectors, amplitudes, phases, ax=ax)
    assert ax_given is ax


@pytest.mark.slow
def test_animate_stellar_convection_builds_and_draws_a_frame():
    times, omega_snaps, T_snaps = simulate_stellar_convection(16, 16, 2 * np.pi, 2 * np.pi, n_steps=20, save_every=10, seed=0)
    anim = animate_stellar_convection(times, omega_snaps, T_snaps)
    anim._draw_frame(1)
    ax_omega, ax_T = anim._fig.axes[0], anim._fig.axes[1]
    assert len(ax_omega.images) == 1 and len(ax_T.images) == 1


def test_animate_dynamo_wave_builds_and_draws_a_frame():
    Lx = 2 * np.pi
    x = np.linspace(0, Lx, 16, endpoint=False)
    A0 = 1e-3 * np.cos(x)
    B0 = np.zeros_like(A0)
    times, A_snaps, B_snaps = simulate_alpha_omega_dynamo(A0, B0, alpha=1.0, shear=5.0, eta=0.05, Lx=Lx, dt=0.01, n_steps=20, save_every=5)
    anim = animate_dynamo_wave(times, A_snaps, B_snaps, x)
    anim._draw_frame(1)
    assert len(anim._fig.axes[0].lines) == 2  # A(x) and B(x)


def test_plot_dynamo_butterfly_diagram_returns_figure():
    Lx = 2 * np.pi
    x = np.linspace(0, Lx, 16, endpoint=False)
    A0 = 1e-3 * np.cos(x)
    B0 = np.zeros_like(A0)
    times, A_snaps, B_snaps = simulate_alpha_omega_dynamo(A0, B0, alpha=1.0, shear=5.0, eta=0.05, Lx=Lx, dt=0.01, n_steps=20, save_every=5)
    fig, ax = plot_dynamo_butterfly_diagram(times, B_snaps, x)
    assert isinstance(fig, plt.Figure)
    assert ax.collections

    _, ax_given = plot_dynamo_butterfly_diagram(times, B_snaps, x, ax=ax)
    assert ax_given is ax
