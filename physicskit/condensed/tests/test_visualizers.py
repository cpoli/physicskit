"""Smoke tests for physicskit.condensed.visualizers: every plot returns a
populated figure, covering both the ax=None and ax=<given> branches. All
pure matplotlib/plotly plotting -- no numba, no file I/O -- so kept small
and fast."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from physicskit.condensed.models import graphene_hamiltonian, haldane_model, ssh_lattice_hamiltonian
from physicskit.condensed.tight_binding import build_finite_cluster, build_ribbon
from physicskit.condensed.topology import compute_berry_curvature
from physicskit.condensed.visualizers import (
    plot_band_structure,
    plot_berry_curvature,
    plot_edge_state_density,
    plot_fermi_surface_3d,
    plot_lattice_structure,
)


def test_plot_band_structure_with_and_without_labels_and_given_ax():
    path = [(0, 0), (2 * np.pi / 3, 4 * np.pi / 3), (0, 0)]
    fig, ax = plot_band_structure(lambda k1, k2: graphene_hamiltonian(k1, k2), path)
    assert isinstance(fig, plt.Figure)

    fig2, ax2 = plot_band_structure(lambda k1, k2: graphene_hamiltonian(k1, k2), path, labels=["G", "K", "G"])
    assert [t.get_text() for t in ax2.get_xticklabels()] == ["G", "K", "G"]

    _, ax_given = plot_band_structure(lambda k1, k2: graphene_hamiltonian(k1, k2), path, ax=ax)
    assert ax_given is ax


def test_plot_berry_curvature_returns_heatmap():
    H = lambda k1, k2: haldane_model(k1, k2, phi=np.pi / 2)
    F = compute_berry_curvature(H, grid_size=8, band_index=0)
    fig, ax = plot_berry_curvature(F)
    assert isinstance(fig, plt.Figure)
    assert len(ax.images) == 1

    _, ax_given = plot_berry_curvature(F, ax=ax)
    assert ax_given is ax


def test_plot_edge_state_density_returns_density_array():
    H = ssh_lattice_hamiltonian(v=0.5, w=1.0)
    H_wire = build_ribbon(H, open_direction=0, n_cells=6)
    fig, ax, density = plot_edge_state_density(H_wire, k_parallel=[])
    assert isinstance(fig, plt.Figure)
    assert density[0] > density[len(density) // 2]

    _, ax_given, _ = plot_edge_state_density(H_wire, k_parallel=[], ax=ax)
    assert ax_given is ax


def test_plot_lattice_structure_with_and_without_weights():
    H, positions, bonds = build_finite_cluster(ssh_lattice_hamiltonian(v=0.5, w=1.0), n_cells=3)
    _, states = np.linalg.eigh(H)
    density = np.abs(states[:, 0]) ** 2

    fig, ax = plot_lattice_structure(positions, bonds)
    assert isinstance(fig, plt.Figure)
    assert len(ax.collections) == 1  # unweighted scatter, no colorbar

    fig2, ax2 = plot_lattice_structure(positions, bonds, weights=density)
    assert len(fig2.axes) == 2  # main axes + colorbar axes
    assert len(ax2.collections) == 1

    _, ax_given = plot_lattice_structure(positions, bonds, ax=ax)
    assert ax_given is ax


def test_plot_lattice_structure_1d_positions_and_zero_max_weight():
    """1D positions (dim=1) get y=0, and an all-zero weights array takes
    the max_w<=0 fallback branch for marker sizing."""
    positions = np.array([[0.0], [1.0], [2.0]])
    bonds = [(0, 1, 1.0), (1, 2, 1.0)]
    fig, ax = plot_lattice_structure(positions, bonds, weights=np.zeros(3))
    assert isinstance(fig, plt.Figure)
    assert len(ax.collections) == 1


def test_plot_fermi_surface_3d_returns_isosurface_trace():
    eps = lambda kx, ky, kz: -2 * (np.cos(kx) + np.cos(ky) + np.cos(kz))
    fig = plot_fermi_surface_3d(eps, mu=0.0, grid_size=6)
    assert fig.data[0].type == "isosurface"
