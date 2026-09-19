"""Tests for physicskit.quantum.chapters.potentials: the parts
test_physics_checks.py and test_animations.py don't already exercise --
the asymmetric step well, the gravitational bouncer (both the Numerov
solver and its closed-form Airy-function check), DoubleWellSimulator's
localized_state, FiniteSquareWell's bound_states/transmission_spectrum/
evanescent_wavefunction, and the 2D rectangular/circular boxes.
"""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.quantum._compat import trapz
from physicskit.quantum.chapters.potentials import (
    Box2DEigenstate,
    CircularBox2D,
    DoubleWellSimulator,
    FiniteSquareWell,
    RectangularBox2D,
    airy_bouncer_energies,
    airy_wavefunction,
    asymmetric_well_states,
    gravitational_bouncer_states,
)

# --- Asymmetric step well -----------------------------------------------------


def test_asymmetric_well_states_are_bound_below_the_lower_wall():
    res = asymmetric_well_states(width=2.0, V_left=40.0, V_right=15.0, n_points=800, n_states=3)
    assert np.all(np.diff(res.energies) > 0)  # ascending
    assert np.all(res.energies < 15.0)  # below the lower (right) wall
    for psi in res.wavefunctions:
        assert trapz(psi**2, res.x) == pytest.approx(1.0, abs=1e-2)


# --- Gravitational bouncer -----------------------------------------------------


def test_gravitational_bouncer_numerov_matches_airy_analytic_energies():
    n_states = 4
    numeric = gravitational_bouncer_states(alpha=1.0, x_extent=12.0, n_points=1500, n_states=n_states)
    analytic = airy_bouncer_energies(alpha=1.0, n_states=n_states)
    np.testing.assert_allclose(numeric.energies, analytic, rtol=2e-2)


def test_airy_bouncer_energies_are_ascending_and_positive():
    energies = airy_bouncer_energies(alpha=2.0, n_states=5)
    assert np.all(energies > 0)
    assert np.all(np.diff(energies) > 0)


def test_airy_wavefunction_is_normalized_and_vanishes_below_floor():
    x = np.linspace(-2, 30, 4000)
    psi = airy_wavefunction(x, alpha=1.0, n=0)
    assert trapz(psi**2, x) == pytest.approx(1.0, abs=1e-3)
    np.testing.assert_allclose(psi[x < 0], 0.0)


def test_airy_wavefunction_overlaps_numerov_ground_state():
    numeric = gravitational_bouncer_states(alpha=1.0, x_extent=12.0, n_points=1500, n_states=1)
    analytic_psi = airy_wavefunction(numeric.x, alpha=1.0, n=0)
    overlap = abs(trapz(analytic_psi * numeric.wavefunctions[0], numeric.x))
    assert overlap == pytest.approx(1.0, abs=1e-2)


# --- Double well: localized_state ---------------------------------------------


def test_double_well_localized_state_is_normalized_and_side_asymmetric():
    dw = DoubleWellSimulator(lam=0.3, a=2.0, n_points=800)
    result = dw.solve(n_states=2)
    left = dw.localized_state(result, side="left")
    right = dw.localized_state(result, side="right")
    assert trapz(left**2, result.x) == pytest.approx(1.0, abs=1e-2)
    assert trapz(right**2, result.x) == pytest.approx(1.0, abs=1e-2)
    left_mask = result.x < 0
    # a state built to localize on the left should weight the left lobe more
    assert trapz(left[left_mask] ** 2, result.x[left_mask]) > trapz(right[left_mask] ** 2, result.x[left_mask])


# --- Finite square well: bound_states / transmission_spectrum / evanescent ----


def test_finite_square_well_bound_states_are_all_negative_energy():
    well = FiniteSquareWell(V0=20.0, width=2.0)
    res = well.bound_states(n_points=800, n_states=6)
    assert res.energies.size > 0
    assert np.all(res.energies < 0)
    assert np.all(np.diff(res.energies) > 0)


def test_finite_square_well_evanescent_wavefunction_matches_bound_states_entry():
    well = FiniteSquareWell(V0=20.0, width=2.0)
    res = well.bound_states(n_points=800, n_states=3)
    np.testing.assert_allclose(well.evanescent_wavefunction(res, n=1), res.wavefunctions[1])


def test_transmission_spectrum_matches_pointwise_scattering():
    well = FiniteSquareWell(V0=10.0, width=1.0)
    E_values = np.array([1.0, 5.0, 12.0])
    spectrum = well.transmission_spectrum(E_values)
    expected = np.array([well.scattering(E).T for E in E_values])
    np.testing.assert_allclose(spectrum, expected)
    assert np.all((spectrum >= 0) & (spectrum <= 1))


# --- 2D rectangular box ---------------------------------------------------------


def test_rectangular_box_energy_matches_formula():
    box = RectangularBox2D(Lx=2.0, Ly=3.0, hbar=1.0, m=1.0)
    nx, ny = 2, 3
    expected = (np.pi**2 * box.hbar**2 / (2 * box.m)) * ((nx / box.Lx) ** 2 + (ny / box.Ly) ** 2)
    assert box.energy(nx, ny) == pytest.approx(expected)


def test_rectangular_box_spectrum_is_sorted_by_energy():
    box = RectangularBox2D(Lx=1.0, Ly=1.7)
    states = box.spectrum(n_max=4)
    energies = [s.energy for s in states]
    assert energies == sorted(energies)
    assert len(states) == 16


def test_rectangular_box_square_has_degeneracies():
    box = RectangularBox2D(Lx=1.0, Ly=1.0)
    degeneracies = box.degeneracies(n_max=4)
    # (1,2) and (2,1) are exactly degenerate for a square box (E ~ nx^2+ny^2)
    assert any({(1, 2), (2, 1)} <= set(pairs) for pairs in degeneracies.values())
    for pairs in degeneracies.values():
        assert len(pairs) >= 2


def test_rectangular_box_grid_has_expected_shape_and_bounds():
    box = RectangularBox2D(Lx=2.0, Ly=3.0)
    X, Y = box.grid(n_points=10)
    assert X.shape == (10, 10)
    assert X.min() == pytest.approx(0.0)
    assert X.max() == pytest.approx(2.0)
    assert Y.max() == pytest.approx(3.0)


def test_box2d_eigenstate_psi_is_normalized_over_the_box():
    box = RectangularBox2D(Lx=1.0, Ly=1.0)
    state = Box2DEigenstate(nx=2, ny=1, energy=box.energy(2, 1))
    X, Y = box.grid(n_points=200)
    psi = state.psi(X, Y, box.Lx, box.Ly)
    integral = trapz(trapz(psi**2, Y[0], axis=1), X[:, 0])
    assert integral == pytest.approx(1.0, abs=1e-2)


# --- 2D circular quantum dot ------------------------------------------------------


def test_circular_box_eigenstate_energy_matches_bessel_zero_formula():
    from scipy.special import jn_zeros

    dot = CircularBox2D(R=1.5, hbar=1.0, m_mass=1.0)
    state = dot.eigenstate(m=1, n=2)
    expected_k = jn_zeros(1, 2)[-1] / dot.R
    assert state.k == pytest.approx(expected_k)
    assert state.energy == pytest.approx(dot.hbar**2 * state.k**2 / (2 * dot.m_mass))


def test_circular_box_spectrum_is_sorted_and_ground_state_is_m_zero_n_one():
    dot = CircularBox2D(R=1.0)
    states = dot.spectrum(m_max=3, n_max=3)
    energies = [s.energy for s in states]
    assert energies == sorted(energies)
    assert states[0].m == 0
    assert states[0].n == 1


def test_circular_box_polar_grid_spans_radius_and_full_turn():
    dot = CircularBox2D(R=2.0)
    r, phi = dot.polar_grid(n_r=20, n_phi=30)
    assert r.shape == (20, 30)
    assert r.max() == pytest.approx(2.0)
    assert phi.max() == pytest.approx(2 * np.pi)


def test_circular_eigenstate_psi_vanishes_at_the_boundary():
    dot = CircularBox2D(R=1.0)
    state = dot.eigenstate(m=0, n=1)
    phi = np.linspace(0, 2 * np.pi, 16)
    r_boundary = np.full_like(phi, dot.R)
    psi_boundary = state.psi(r_boundary, phi, dot.R)
    np.testing.assert_allclose(psi_boundary, 0.0, atol=1e-8)
