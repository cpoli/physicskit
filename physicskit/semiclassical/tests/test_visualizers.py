"""Coverage for physicskit.semiclassical.visualizers's ax=<given> branch
(previously untested for every function here): plot_density_of_states,
plot_wkb_wavefunction, plot_classical_trajectory_on_wigner, plot_scar_map,
plot_husimi_1d. All pure matplotlib, small inputs -- fast."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from physicskit.semiclassical.core.gutzwiller import gutzwiller_density_of_states
from physicskit.semiclassical.core.path_integral import build_phasor_diagram
from physicskit.semiclassical.core.propagators import frozen_gaussian_1d
from physicskit.semiclassical.core.wkb import bohr_sommerfeld_energies, wkb_wavefunction
from physicskit.semiclassical.systems.scarring import bouncing_ball_orbit_points, husimi_projection_1d
from physicskit.semiclassical.visualizers.gutzwiller import plot_density_of_states
from physicskit.semiclassical.visualizers.path_integral import animate_feynman_phasor_spiral, plot_phasor_convergence
from physicskit.semiclassical.visualizers.propagators import plot_classical_trajectory_on_wigner
from physicskit.semiclassical.visualizers.scarring import plot_husimi_1d, plot_scar_map
from physicskit.semiclassical.visualizers.wkb import plot_wkb_wavefunction


def test_plot_density_of_states_given_ax():
    V = lambda x: 0.5 * x**2
    E_grid = np.linspace(0.2, 3.5, 50)
    dos = gutzwiller_density_of_states(E_grid, V, m=1.0, x_min=-20, x_max=20)
    fig, ax = plt.subplots()
    _, ax_given = plot_density_of_states(E_grid, dos, ax=ax)
    assert ax_given is ax


def test_plot_wkb_wavefunction_given_ax():
    V = lambda x: 0.5 * x**2
    E2 = bohr_sommerfeld_energies(V, m=1.0, x_min=-20, x_max=20, n_max=3)[2]
    x = np.linspace(-6, 6, 200)
    psi = wkb_wavefunction(x, E2, V)
    fig, ax = plt.subplots()
    _, ax_given = plot_wkb_wavefunction(x, psi, V=V, ax=ax)
    assert ax_given is ax


def test_plot_classical_trajectory_on_wigner_given_ax():
    x = np.linspace(-8, 8, 100)
    psi = frozen_gaussian_1d(x, qc=1.0, pc=0.5, gamma=1.0)
    theta = np.linspace(0, 2 * np.pi, 30)
    q_hist, p_hist = np.cos(theta), np.sin(theta)
    fig, ax = plt.subplots()
    _, ax_given = plot_classical_trajectory_on_wigner(x, psi, q_hist, p_hist, ax=ax)
    assert ax_given is ax


def test_plot_scar_map_given_ax():
    x = np.linspace(-2, 2, 30)
    y = np.linspace(-1, 1, 20)
    X, Y = np.meshgrid(x, y, indexing="ij")
    density = np.exp(-((X - 0.3) ** 2) / (2 * 0.2**2))
    orbit_x, orbit_y = bouncing_ball_orbit_points(x0=0.3, R=1.0)
    fig, ax = plt.subplots()
    _, ax_given = plot_scar_map(X, Y, density, orbit_x=orbit_x, orbit_y=orbit_y, ax=ax)
    assert ax_given is ax


def test_plot_husimi_1d_given_ax():
    s = np.linspace(-10, 10, 200)
    psi = np.exp(-((s - 2.0) ** 2) / 2.0) * np.exp(1j * 3.0 * s)
    S0, P0, H = husimi_projection_1d(psi, s, sigma=1.0, resolution=20, s0_range=(-2, 6), p0_range=(-2, 8))
    fig, ax = plt.subplots()
    _, ax_given = plot_husimi_1d(S0, P0, H, ax=ax)
    assert ax_given is ax


def test_animate_feynman_phasor_spiral_builds_and_draws_a_frame():
    paths, actions, x_cl, S_cl, _ = build_phasor_diagram(x0=0.0, xf=1.0, T=1.0, m=1.0, hbar=0.1, potential="free", n_slices=20, n_paths=10, sigma=0.3, seed=0)
    t = np.linspace(0.0, 1.0, x_cl.shape[0])
    anim = animate_feynman_phasor_spiral(paths, actions, x_cl, S_cl, hbar=0.1, t_array=t)
    anim._init_draw()
    anim._draw_frame(0)
    anim._draw_frame(1)  # a second frame exercises removing the previous frame's path lines
    assert "2/" in anim._fig.axes[1].get_title()


def test_plot_phasor_convergence_given_ax():
    sums, hbars = [], [0.5, 0.05]
    for hbar in hbars:
        _, _, _, _, partial = build_phasor_diagram(x0=0.0, xf=1.0, T=1.0, m=1.0, hbar=hbar, potential="free", n_slices=10, n_paths=20, sigma=0.3, seed=0)
        sums.append(partial)
    fig, ax = plt.subplots()
    _, ax_given = plot_phasor_convergence(sums, hbars, ax=ax)
    assert ax_given is ax
