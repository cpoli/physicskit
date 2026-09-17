"""Tests for physicskit.quantum.chapters.entanglement: Bell states, EPR/CHSH
correlations, Ising-coupling entanglement generation, and the
Aharonov-Bohm ring -- covering the parts test_physics_checks.py and
test_animations.py don't already exercise (measurement_probabilities,
monte_carlo_correlation, the reduced-density-matrix/purity/trajectory
helpers, and the ring's spectrum/eigenstate/persistent_current).
"""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.quantum.chapters.entanglement import (
    AharonovBohmRing,
    BellCorrelations,
    IsingEntangler,
    bell_state,
)


# --- Bell states -------------------------------------------------------------


@pytest.mark.parametrize("kind", ["phi+", "phi-", "psi+", "psi-"])
def test_bell_state_is_normalized(kind):
    state = bell_state(kind)
    assert np.vdot(state, state).real == pytest.approx(1.0)


def test_bell_state_rejects_unknown_kind():
    with pytest.raises(ValueError):
        bell_state("not-a-bell-state")


# --- BellCorrelations ---------------------------------------------------------


def test_measurement_probabilities_of_singlet_at_matched_angles_are_anticorrelated():
    bc = BellCorrelations()
    probs = bc.measurement_probabilities(0.0, 0.0)
    assert probs["++"] == pytest.approx(0.0, abs=1e-12)
    assert probs["--"] == pytest.approx(0.0, abs=1e-12)
    assert probs["+-"] == pytest.approx(0.5)
    assert probs["-+"] == pytest.approx(0.5)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_measurement_probabilities_sum_to_one_at_arbitrary_angles():
    bc = BellCorrelations()
    probs = bc.measurement_probabilities(0.7, -1.3)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_monte_carlo_correlation_converges_to_exact_correlation():
    bc = BellCorrelations()
    rng = np.random.default_rng(0)
    empirical = bc.monte_carlo_correlation(0.0, 0.0, n_trials=20000, rng=rng)
    assert empirical == pytest.approx(bc.correlation(0.0, 0.0), abs=0.03)


# --- IsingEntangler ------------------------------------------------------------


def test_reduced_density_matrix_and_purity_of_initial_product_state_are_pure():
    ising = IsingEntangler(J=1.0)
    psi0 = ising.initial_state()
    rho = ising.reduced_density_matrix(psi0)
    assert rho.shape == (2, 2)
    assert np.trace(rho).real == pytest.approx(1.0)
    assert ising.purity(psi0) == pytest.approx(1.0)


def test_purity_drops_to_one_half_at_maximal_entanglement():
    ising = IsingEntangler(J=1.0)
    t_max_entangle = np.pi / 4
    psi = ising.state(t_max_entangle)
    assert ising.purity(psi) == pytest.approx(0.5, abs=1e-9)
    assert ising.concurrence(psi) == pytest.approx(1.0, abs=1e-9)


def test_state_trajectory_matches_pointwise_state_calls():
    ising = IsingEntangler(J=1.0)
    t_values = np.linspace(0, 1, 5)
    trajectory = ising.state_trajectory(t_values)
    assert trajectory.shape == (5, 4)
    for i, t in enumerate(t_values):
        np.testing.assert_allclose(trajectory[i], ising.state(t))


def test_concurrence_trajectory_starts_at_zero_and_peaks_at_one():
    ising = IsingEntangler(J=1.0)
    t_values = np.array([0.0, np.pi / 4])
    concurrences = ising.concurrence_trajectory(t_values)
    assert concurrences[0] == pytest.approx(0.0, abs=1e-9)
    assert concurrences[1] == pytest.approx(1.0, abs=1e-9)


# --- Aharonov-Bohm ring --------------------------------------------------------


def test_spectrum_is_sorted_and_minimal_at_n_zero_for_zero_flux():
    ring = AharonovBohmRing(R=1.0)
    spectrum = ring.spectrum(Phi=0.0, n_range=5)
    assert spectrum.shape == (11,)
    np.testing.assert_allclose(spectrum, np.sort(spectrum))
    assert spectrum[0] == pytest.approx(0.0)


def test_eigenstate_has_constant_magnitude():
    ring = AharonovBohmRing(R=2.0)
    phi = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    psi = ring.eigenstate(n=3, phi=phi)
    np.testing.assert_allclose(np.abs(psi), 1.0 / np.sqrt(2 * np.pi * ring.R))


def test_persistent_current_matches_analytic_derivative_of_energy():
    ring = AharonovBohmRing(R=1.0, m=1.0, hbar=1.0, q=1.0)
    n, Phi = 1, 0.0
    expected = (ring.hbar**2 / (ring.m * ring.R**2 * ring.flux_quantum)) * (n - Phi / ring.flux_quantum)
    assert ring.persistent_current(n, Phi) == pytest.approx(expected, rel=1e-6)


def test_persistent_current_vanishes_by_symmetry_for_n_zero_at_zero_flux():
    ring = AharonovBohmRing(R=1.0)
    assert ring.persistent_current(n=0, Phi=0.0) == pytest.approx(0.0, abs=1e-9)
