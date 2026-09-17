"""Coverage for physicskit.plasma.visualizers's static plotting functions
(previously untested): the ax=None/ax=<given> branch of each. All pure
matplotlib, small inputs -- fast."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from physicskit.plasma.kinetic import landau_damping_ic
from physicskit.plasma.mhd import safety_factor_large_aspect_ratio, solve_grad_shafranov
from physicskit.plasma.single_particle import MP, QE, boris_integrate
from physicskit.plasma.visualizers import (
    plot_cma_diagram,
    plot_drift_trajectory,
    plot_field_energy_history,
    plot_flux_surfaces,
    plot_particle_orbit_3d,
    plot_phase_space,
    plot_q_profile,
)
from physicskit.plasma.waves import cma_coordinates


def test_plot_particle_orbit_3d_given_ax():
    pos_hist, _ = boris_integrate(np.zeros(3), np.array([1e5, 0.0, 0.0]), QE, MP, np.zeros(3), np.array([0.0, 0.0, 1.0]), 1e-10, steps=20)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    _, ax_given = plot_particle_orbit_3d(pos_hist, ax=ax)
    assert ax_given is ax
    assert fig is ax.figure


def test_plot_drift_trajectory_given_ax():
    E = np.array([0.0, 1e3, 0.0])
    B = np.array([0.0, 0.0, 1.0])
    pos_hist, _ = boris_integrate(np.zeros(3), np.zeros(3), QE, MP, E, B, 1e-10, steps=20)
    fig, ax = plt.subplots()
    _, ax_given = plot_drift_trajectory(pos_hist, ax=ax)
    assert ax_given is ax


def test_plot_flux_surfaces_given_ax():
    R = np.linspace(0.5, 1.5, 11)
    Z = np.linspace(-0.5, 0.5, 11)
    psi = solve_grad_shafranov(R, Z, c1=1.0, c2=-2.0)
    fig, ax = plt.subplots()
    _, ax_given = plot_flux_surfaces(R, Z, psi, ax=ax)
    assert ax_given is ax


def test_plot_q_profile_given_ax():
    r = np.linspace(0.05, 0.5, 10)
    q = np.array([safety_factor_large_aspect_ratio(ri, R0=1.0, Bt=2.0, Bp=0.2) for ri in r])
    fig, ax = plt.subplots()
    _, ax_given = plot_q_profile(r, q, ax=ax)
    assert ax_given is ax


def test_plot_phase_space_given_ax():
    x, v = landau_damping_ic(200, L=4 * np.pi, k_mode=0.5, alpha=0.1, v_th=1.0, seed=0)
    fig, ax = plt.subplots()
    _, ax_given = plot_phase_space(x, v, ax=ax)
    assert ax_given is ax


def test_plot_field_energy_history_given_ax():
    t = np.linspace(0, 10, 20)
    field_energy = np.exp(-0.1 * t)
    fig, ax = plt.subplots()
    _, ax_given = plot_field_energy_history(t, field_energy, ax=ax)
    assert ax_given is ax


def test_plot_cma_diagram_given_ax():
    omega = np.linspace(0.1, 5.0, 10)
    X, Y = cma_coordinates(omega, wpe=2.0, wce=3.0)
    fig, ax = plt.subplots()
    _, ax_given = plot_cma_diagram(X, Y, ax=ax)
    assert ax_given is ax
