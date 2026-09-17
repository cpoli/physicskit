"""Coverage for physicskit.fluids.visualizers's static plotting functions
(previously entirely untested): plot_shock_tube_profiles, plot_streamlines,
plot_vorticity_field, plot_pressure_coefficient, plot_energy_spectrum --
including the ax=<given> branch and the optional-overlay branches. All pure
matplotlib, small inputs -- fast."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from physicskit.fluids.systems.compressible_flow import sod_shock_tube
from physicskit.fluids.systems.potential_flow import pressure_coefficient
from physicskit.fluids.utils.spectral_analysis import energy_spectrum
from physicskit.fluids.visualizers.compressible import plot_shock_tube_profiles
from physicskit.fluids.visualizers.flow_fields import plot_streamlines, plot_vorticity_field
from physicskit.fluids.visualizers.potential_flow import plot_pressure_coefficient
from physicskit.fluids.visualizers.spectra import plot_energy_spectrum


def test_plot_shock_tube_profiles_returns_three_axes():
    result = sod_shock_tube(nx=50, t_final=0.15)
    fig, axes = plot_shock_tube_profiles(result["x"], result["rho"], result["u"], result["p"])
    assert isinstance(fig, Figure)
    assert len(axes) == 3
    for ax in axes:
        assert len(ax.lines) == 1


def test_plot_streamlines_default_and_given_ax():
    x = np.linspace(-1, 1, 10)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u, v = -Y, X
    fig0, ax0 = plot_streamlines(X, Y, u, v)
    assert isinstance(fig0, Figure)

    fig, ax = plt.subplots()
    _, ax_given = plot_streamlines(X, Y, u, v, ax=ax)
    assert ax_given is ax


def test_plot_vorticity_field_given_ax_and_with_streamline_overlay():
    x = np.linspace(-1, 1, 10)
    X, Y = np.meshgrid(x, x, indexing="ij")
    omega = np.sin(X) * np.cos(Y)
    u, v = -Y, X

    fig0, ax0 = plot_vorticity_field(X, Y, omega)
    assert isinstance(fig0, Figure)

    fig, ax = plt.subplots()
    _, ax_given = plot_vorticity_field(X, Y, omega, ax=ax)
    assert ax_given is ax
    assert len(ax.lines) == 0  # no streamlines without u, v

    fig2, ax2 = plt.subplots()
    _, ax2_given = plot_vorticity_field(X, Y, omega, u=u, v=v, ax=ax2)
    assert ax2_given is ax2
    assert ax2.lines or ax2.collections  # streamplot overlay was drawn


def test_plot_pressure_coefficient_given_ax():
    theta = np.linspace(0, 2 * np.pi, 30)
    Cp = pressure_coefficient(u=np.cos(theta), v=np.sin(theta), U_inf=1.0)
    fig0, ax0 = plot_pressure_coefficient(theta, Cp)
    assert isinstance(fig0, Figure)

    fig, ax = plt.subplots()
    _, ax_given = plot_pressure_coefficient(theta, Cp, ax=ax)
    assert ax_given is ax


def test_plot_energy_spectrum_given_ax_and_without_kolmogorov_line():
    n, length = 32, 2 * np.pi
    x = np.linspace(0, length, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u, v = np.sin(3 * Y), np.sin(3 * X)
    k, E = energy_spectrum(u, v, length)

    fig0, ax0 = plot_energy_spectrum(k, E)
    assert isinstance(fig0, Figure)

    fig, ax = plt.subplots()
    _, ax_given = plot_energy_spectrum(k, E, ax=ax)
    assert ax_given is ax
    assert len(ax.lines) == 2  # spectrum + Kolmogorov reference

    fig2, ax2 = plt.subplots()
    plot_energy_spectrum(k, E, ax=ax2, show_kolmogorov=False)
    assert len(ax2.lines) == 1  # spectrum only
