"""Tests for physicskit.quantum.chapters.scattering: Rutherford from the Born
approximation, hard-sphere and square-well phase shifts, the optical
theorem, and Born vs. partial-wave agreement for weak potentials."""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.particle.scattering import rutherford_dsigma_domega
from physicskit.quantum.chapters.scattering import (
    born_amplitude,
    born_phase_shifts,
    hard_sphere_phase_shifts,
    momentum_transfer,
    partial_wave_amplitude,
    partial_wave_cross_section,
    partial_wave_phase_shifts,
    yukawa_born_amplitude,
)


def test_unscreened_yukawa_born_is_rutherford():
    mass, k, Z1, Z2, alpha = 1.0, 2.0, 2, 79, 1 / 137.035999084
    theta = np.linspace(0.2, np.pi, 30)
    q = momentum_transfer(k, theta)
    dsigma = yukawa_born_amplitude(q, g=Z1 * Z2 * alpha, mu=0.0, mass=mass) ** 2
    E_kin = k**2 / (2 * mass)
    np.testing.assert_allclose(dsigma, rutherford_dsigma_domega(theta, Z1, Z2, E_kin, alpha), rtol=1e-12)


def test_numerical_born_matches_screened_coulomb_closed_form():
    g, mu = 0.7, 0.3
    q = np.array([0.0, 0.1, 0.5, 1.0, 3.0])
    f = born_amplitude(lambda r: g * np.exp(-mu * r) / r, q)
    np.testing.assert_allclose(f, yukawa_born_amplitude(q, g, mu), rtol=1e-7)


def test_weakly_screened_numerical_born_approaches_rutherford():
    g, mu, k = 1.0, 1e-3, 3.0
    theta = np.array([0.5, 1.0, 2.0, 3.0])
    f = born_amplitude(lambda r: g * np.exp(-mu * r) / r, momentum_transfer(k, theta))
    np.testing.assert_allclose(f**2, rutherford_dsigma_domega(theta, 1, 1, k**2 / 2, alpha=g), rtol=1e-4)


def test_hard_sphere_s_wave_and_cross_section_limits():
    a = 1.0
    for k in (0.1, 0.5, 2.0):
        # phase shifts are defined modulo pi
        assert np.exp(2j * hard_sphere_phase_shifts(k, a, 0)[0]) == pytest.approx(np.exp(-2j * k * a))
    # low energy: sigma -> 4 pi a^2
    k = 1e-3
    assert partial_wave_cross_section(k, hard_sphere_phase_shifts(k, a, 3)) == pytest.approx(4 * np.pi * a**2, rel=1e-5)
    # high energy: sigma -> 2 pi a^2 [1 + 0.9962 (ka)^(-2/3)] (Nussenzveig 1969)
    for k in (200.0, 800.0):
        sigma = partial_wave_cross_section(k, hard_sphere_phase_shifts(k, a, int(k * a) + 60))
        assert sigma == pytest.approx(2 * np.pi * a**2 * (1 + 0.9962 * (k * a) ** (-2 / 3)), rel=5e-4)


def test_hard_sphere_p_wave_closed_form():
    # tan(delta_1) = (ka - tan ka)/(1 + ka tan ka)  (Sakurai Eq. 7.6.33 rearranged)
    x = 0.9
    d1 = hard_sphere_phase_shifts(x, 1.0, 1)[1]
    assert np.tan(d1) == pytest.approx((x - np.tan(x)) / (1 + x * np.tan(x)))


def test_high_barrier_approaches_hard_sphere():
    k, a = 1.0, 1.0
    wall = lambda r: np.where(r < a, 2e4, 0.0)  # noqa: E731
    d = partial_wave_phase_shifts(wall, k, 3, r_max=1.5, breakpoints=[a])
    # penetration depth 1/sqrt(2 V0) = 0.005 shifts the effective radius
    np.testing.assert_allclose(np.exp(2j * d), np.exp(2j * hard_sphere_phase_shifts(k, a - 0.005, 3)), atol=5e-3)


@pytest.mark.parametrize("V0, k", [(0.8, 0.7), (3.0, 0.4), (10.0, 1.5)])
def test_square_well_s_wave(V0, k):
    a = 1.0
    well = lambda r: np.where(r < a, -V0, 0.0)  # noqa: E731
    d0 = partial_wave_phase_shifts(well, k, 0, r_max=a + 0.5, breakpoints=[a])[0]
    K = np.sqrt(k**2 + 2 * V0)
    exact = -k * a + np.arctan(k / K * np.tan(K * a))
    assert np.exp(2j * d0) == pytest.approx(np.exp(2j * exact), abs=1e-7)


def test_weak_potential_phase_shifts_match_born():
    V = lambda r: 0.02 * np.exp(-r / 0.8) / r  # noqa: E731
    d_exact = partial_wave_phase_shifts(V, 1.2, 4, r_max=25.0)
    d_born = born_phase_shifts(V, 1.2, 4, r_max=25.0)
    np.testing.assert_allclose(d_exact, d_born, rtol=0.03, atol=1e-7)


def test_optical_theorem_and_born_amplitude_agreement():
    g, mu, k = 0.02, 1.0, 1.0
    V = lambda r: g * np.exp(-mu * r) / r  # noqa: E731
    deltas = partial_wave_phase_shifts(V, k, 12, r_max=30.0)
    sigma = partial_wave_cross_section(k, deltas)
    assert partial_wave_amplitude(0.0, k, deltas).imag == pytest.approx(k * sigma / (4 * np.pi), rel=1e-10)
    theta = np.linspace(0, np.pi, 9)
    f_pw = partial_wave_amplitude(theta, k, deltas)
    f_born = yukawa_born_amplitude(momentum_transfer(k, theta), g, mu)
    np.testing.assert_allclose(f_pw.real, f_born, rtol=0.03)
