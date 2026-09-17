"""Tests for physicskit.quantum.chapters.harmonic_spin: the parts of
HarmonicOscillator/ThermalState not already exercised by
test_physics_checks.py or test_animations.py -- time-evolved stationary
states, squeezed states, thermal populations/distributions/variance, and
the ThermalState convenience wrapper.
"""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.quantum._compat import trapz
from physicskit.quantum.chapters.harmonic_spin import HarmonicOscillator, ThermalState


@pytest.fixture
def osc():
    return HarmonicOscillator(m=1.0, omega=1.0, hbar=1.0)


# --- Time-evolved stationary states ---------------------------------------


def test_eigenfunction_t_has_time_independent_density(osc):
    x = np.linspace(-6, 6, 200)
    psi_t0 = osc.eigenfunction_t(2, x, t=0.0)
    psi_t1 = osc.eigenfunction_t(2, x, t=1.7)
    np.testing.assert_allclose(np.abs(psi_t0) ** 2, np.abs(psi_t1) ** 2, atol=1e-10)
    np.testing.assert_allclose(psi_t0.real, osc.eigenfunction(2, x))


def test_eigenfunction_t_accumulates_correct_phase(osc):
    x = np.linspace(-6, 6, 200)
    t = 0.3
    psi_t = osc.eigenfunction_t(1, x, t)
    expected_phase = np.exp(-1j * osc.energy(1) * t / osc.hbar)
    np.testing.assert_allclose(psi_t, osc.eigenfunction(1, x) * expected_phase)


# --- Squeezed states ---------------------------------------------------------


def test_squeezed_vacuum_wavefunction_is_normalized(osc):
    x = np.linspace(-10, 10, 4000)
    psi = osc.squeezed_vacuum_wavefunction(x, r=0.5)
    assert trapz(np.abs(psi) ** 2, x) == pytest.approx(1.0, abs=1e-3)


def test_squeezed_vacuum_wavefunction_is_real_when_phi_is_zero(osc):
    x = np.linspace(-10, 10, 100)
    psi = osc.squeezed_vacuum_wavefunction(x, r=0.8, phi=0.0)
    np.testing.assert_allclose(psi.imag, 0.0, atol=1e-12)


def test_squeezed_vacuum_wavefunction_has_complex_chirp_when_phi_nonzero(osc):
    x = np.linspace(-10, 10, 100)
    psi = osc.squeezed_vacuum_wavefunction(x, r=0.8, phi=0.5)
    assert np.any(np.abs(psi.imag) > 1e-12)


def test_squeezed_uncertainties_matches_variance_formula(osc):
    r = 0.6
    dx, dp = osc.squeezed_uncertainties(r)
    assert dx == pytest.approx(np.sqrt(osc.hbar / (2 * osc.m * osc.omega) * np.exp(-2 * r)))
    assert dp == pytest.approx(np.sqrt(osc.hbar * osc.m * osc.omega / 2 * np.exp(2 * r)))


def test_squeezed_uncertainties_saturates_heisenberg_bound_for_any_r(osc):
    for r in (0.0, 0.5, 1.5):
        dx, dp = osc.squeezed_uncertainties(r)
        assert dx * dp == pytest.approx(osc.hbar / 2)


def test_zero_point_uncertainties_matches_squeezed_at_r_zero(osc):
    assert osc.zero_point_uncertainties() == osc.squeezed_uncertainties(r=0.0)


# --- Thermal (mixed) states ---------------------------------------------------


def test_thermal_populations_at_zero_temperature_is_ground_state(osc):
    pops = osc.thermal_populations(T=0.0, n_max=5)
    np.testing.assert_allclose(pops, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_thermal_populations_sum_to_one_and_decay_with_n(osc):
    pops = osc.thermal_populations(T=2.0, n_max=50)
    assert pops.sum() == pytest.approx(1.0)
    assert np.all(np.diff(pops) < 0)  # monotonically decreasing Boltzmann weights


def test_thermal_position_distribution_is_normalized_and_symmetric(osc):
    x = np.linspace(-15, 15, 3000)
    P = osc.thermal_position_distribution(x, T=1.0, n_max=60)
    assert trapz(P, x) == pytest.approx(1.0, abs=1e-2)
    np.testing.assert_allclose(P, P[::-1], atol=1e-6)


def test_thermal_position_variance_matches_numerical_integral(osc):
    x = np.linspace(-20, 20, 6000)
    T = 1.5
    P = osc.thermal_position_distribution(x, T=T, n_max=80)
    numeric_var = trapz(x**2 * P, x)
    assert osc.thermal_position_variance_analytic(T) == pytest.approx(numeric_var, rel=1e-2)


def test_thermal_position_variance_reduces_to_zero_point_as_t_to_zero(osc):
    assert osc.thermal_position_variance_analytic(T=0.0) == pytest.approx(osc.hbar / (2 * osc.m * osc.omega))


def test_thermal_position_variance_grows_with_temperature(osc):
    var_cold = osc.thermal_position_variance_analytic(T=0.1)
    var_hot = osc.thermal_position_variance_analytic(T=10.0)
    assert var_hot > var_cold


# --- ThermalState convenience wrapper -----------------------------------------


def test_thermal_state_populations_delegates_to_oscillator(osc):
    ts = ThermalState(oscillator=osc, T=2.0, n_max=30)
    np.testing.assert_allclose(ts.populations(), osc.thermal_populations(2.0, 30, 1.0))


def test_thermal_state_position_distribution_delegates_to_oscillator(osc):
    x = np.linspace(-10, 10, 200)
    ts = ThermalState(oscillator=osc, T=2.0, n_max=30)
    np.testing.assert_allclose(ts.position_distribution(x), osc.thermal_position_distribution(x, 2.0, 1.0, 30))


def test_thermal_state_variance_delegates_to_oscillator(osc):
    ts = ThermalState(oscillator=osc, T=2.0)
    assert ts.variance() == osc.thermal_position_variance_analytic(2.0, 1.0)


def test_thermal_state_check_normalization_is_close_to_one(osc):
    x = np.linspace(-15, 15, 3000)
    ts = ThermalState(oscillator=osc, T=1.0, n_max=60)
    assert ts.check_normalization(x) == pytest.approx(1.0, abs=1e-2)
