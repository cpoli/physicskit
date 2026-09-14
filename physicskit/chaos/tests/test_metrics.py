import numpy as np
import pytest

from physicskit.chaos.systems.continuous import DoublePendulum, Lorenz
from physicskit.chaos.systems.maps import HenonMap, StandardMap
from physicskit.chaos.utils.metrics import (
    benettin_lyapunov_spectrum,
    energy_drift,
    lyapunov_exponent_from_divergence,
    map_lyapunov_spectrum,
    numerical_jacobian,
    numerical_jacobian_map,
    phase_volume_expansion,
)


def test_numerical_jacobian_matches_known_linear_system():
    """For dx/dt = A x, the Jacobian is exactly A everywhere."""

    class Linear:
        dim = 2

        def rhs(self, state, t):
            a = np.array([[0.0, 1.0], [-2.0, -3.0]])
            return a @ state

    jac = numerical_jacobian(Linear(), np.array([1.0, 2.0]), t=0.0)
    np.testing.assert_allclose(jac, [[0.0, 1.0], [-2.0, -3.0]], atol=1e-6)


def test_numerical_jacobian_map_matches_known_henon_jacobian():
    """The Henon map's Jacobian at (x, y) is exactly [[-2*a*x, 1], [b, 0]]."""
    henon = HenonMap(a=1.4, b=0.3)
    x, y = 0.3, 0.1
    jac = numerical_jacobian_map(henon, np.array([x, y]))
    expected = np.array([[-2.0 * 1.4 * x, 1.0], [0.3, 0.0]])
    np.testing.assert_allclose(jac, expected, atol=1e-5)


def test_lyapunov_exponent_from_divergence_recovers_known_exponential_rate():
    t = np.linspace(0.0, 10.0, 200)
    true_rate = 0.7
    delta = 1e-6 * np.exp(true_rate * t)
    estimate = lyapunov_exponent_from_divergence(t, delta)
    assert estimate == pytest.approx(true_rate, rel=1e-3)


def test_benettin_lyapunov_spectrum_matches_known_lorenz_values():
    """The classic Lorenz parameters have well-documented Lyapunov exponents
    of approximately (0.906, 0, -14.57)."""
    system = Lorenz()
    spectrum = benettin_lyapunov_spectrum(system, system.initial_state(), dt=0.005, n_steps=10000, n_transient=2000)
    assert spectrum[0] == pytest.approx(0.906, abs=0.15)
    assert abs(spectrum[1]) < 0.1
    assert spectrum[2] == pytest.approx(-14.57, abs=1.5)


def test_benettin_lyapunov_spectrum_sum_matches_exact_lorenz_divergence():
    """The Lorenz system's phase-space divergence (trace of the Jacobian) is
    exactly -(sigma + 1 + beta) everywhere, so the sum of the Lyapunov
    spectrum -- estimated with no knowledge of that identity -- should match
    it closely, regardless of how accurately the individual exponents (which
    are harder to pin down individually) are estimated."""
    system = Lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
    spectrum = benettin_lyapunov_spectrum(system, system.initial_state(), dt=0.005, n_steps=10000, n_transient=2000)
    expected_sum = -(system.sigma + 1.0 + system.beta)
    assert spectrum.sum() == pytest.approx(expected_sum, abs=0.05)


def test_map_lyapunov_spectrum_sum_matches_exact_henon_jacobian_determinant():
    """The Henon map's Jacobian determinant is exactly -b everywhere, so the
    sum of its Lyapunov exponents must equal ln|b| to high precision --
    regardless of how well the individual exponents are estimated."""
    henon = HenonMap(a=1.4, b=0.3)
    spectrum = map_lyapunov_spectrum(henon, henon.initial_state(), n_iter=10000, n_transient=1000)
    assert spectrum.sum() == pytest.approx(np.log(0.3), abs=1e-4)
    assert spectrum[0] > 0.0 > spectrum[1]


def test_map_lyapunov_spectrum_sums_to_zero_for_area_preserving_map():
    """The standard map is exactly area-preserving, so its Lyapunov spectrum
    must sum to zero."""
    system = StandardMap(k=2.0)
    spectrum = map_lyapunov_spectrum(system, system.initial_state(), n_iter=10000, n_transient=1000)
    assert spectrum.sum() == pytest.approx(0.0, abs=1e-4)


def test_energy_drift_is_zero_at_t0():
    system = DoublePendulum()
    t, states = system.trajectory(n_steps=500, dt=0.005)
    drift = energy_drift(t, states, system.energy)
    assert drift[0] == pytest.approx(0.0)


def test_energy_drift_stays_small_for_a_good_integration():
    system = DoublePendulum()
    t, states = system.trajectory(n_steps=2000, dt=0.005)
    drift = energy_drift(t, states, system.energy)
    assert np.max(np.abs(drift)) < 1e-3


def test_phase_volume_expansion_starts_at_zero():
    system = Lorenz()
    t, states = system.trajectory(n_steps=500, dt=0.01)
    log_volume = phase_volume_expansion(system, t, states)
    assert log_volume[0] == pytest.approx(0.0)


def test_phase_volume_expansion_is_dissipative_for_lorenz():
    """Lorenz phase-space volume must contract (log volume strictly
    decreasing on average) since the system is dissipative."""
    system = Lorenz(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
    t, states = system.trajectory(n_steps=1000, dt=0.01)
    log_volume = phase_volume_expansion(system, t, states)
    assert log_volume[-1] < 0.0
