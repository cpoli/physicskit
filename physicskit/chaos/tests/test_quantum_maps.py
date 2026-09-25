import numpy as np
import pytest

from physicskit.chaos.quantum.husimi import husimi_function
from physicskit.chaos.quantum.maps import QuantumBakersMap, QuantumKickedRotor
from physicskit.chaos.systems.maps import BakersMap, StandardMap


def _is_unitary(u, atol=1e-10):
    n = u.shape[0]
    return np.allclose(u.conj().T @ u, np.eye(n), atol=atol)


@pytest.mark.parametrize("dim", [8, 33, 64])
def test_kicked_rotor_floquet_operator_is_unitary(dim):
    qkr = QuantumKickedRotor(k=1.7, dim=dim)
    assert _is_unitary(qkr.floquet_operator())


def test_kicked_rotor_rejects_too_small_dim():
    with pytest.raises(ValueError):
        QuantumKickedRotor(dim=1)


def test_kicked_rotor_default_hbar_matches_torus_quantization():
    qkr = QuantumKickedRotor(dim=100)
    assert qkr.hbar == pytest.approx(2.0 * np.pi / 100)


def test_kicked_rotor_evolve_preserves_norm_and_dimension():
    qkr = QuantumKickedRotor(k=2.0, dim=50)
    psi0 = qkr.coherent_state(1.0, 2.0)
    states = qkr.evolve(psi0, n_steps=10)
    assert states.shape == (11, 50)
    assert np.allclose(np.linalg.norm(states, axis=1), 1.0)


def test_kicked_rotor_eigenphases_are_on_unit_circle():
    qkr = QuantumKickedRotor(k=3.0, dim=40)
    phases = qkr.eigenphases()
    assert phases.shape == (40,)
    assert np.all(phases >= -np.pi) and np.all(phases <= np.pi)
    assert np.all(np.diff(phases) >= 0.0)


def test_kicked_rotor_coherent_state_husimi_peaks_at_its_own_center():
    qkr = QuantumKickedRotor(k=1.0, dim=300)
    theta0, p0 = 2.3, 4.1
    psi = qkr.coherent_state(theta0, p0)
    theta_grid, p_grid, husimi = qkr.husimi(psi, resolution=150)
    idx = np.unravel_index(np.argmax(husimi), husimi.shape)
    assert theta_grid[idx] == pytest.approx(theta0, abs=0.1)
    assert p_grid[idx] == pytest.approx(p0, abs=0.1)


@pytest.mark.slow
def test_kicked_rotor_husimi_peak_tracks_classical_orbit_at_weak_kick():
    """At small k the classical map is near-integrable, so a narrow wavepacket's
    Husimi peak should closely track the classical StandardMap orbit for a
    handful of iterations, before it has a chance to spread appreciably."""
    k = 0.3
    dim = 200
    theta0, p0 = 1.0, 1.0
    qkr = QuantumKickedRotor(k=k, dim=dim)
    psi = qkr.coherent_state(theta0, p0)
    states = qkr.evolve(psi, n_steps=4)

    classical = StandardMap(k=k).trajectory(np.array([theta0, p0]), n_iter=4)

    for i in range(5):
        theta_grid, p_grid, husimi = qkr.husimi(states[i], resolution=80)
        idx = np.unravel_index(np.argmax(husimi), husimi.shape)
        assert theta_grid[idx] == pytest.approx(classical[i, 0], abs=0.15)
        assert p_grid[idx] == pytest.approx(classical[i, 1], abs=0.15)


@pytest.mark.parametrize("dim", [8, 32, 100])
def test_bakers_map_floquet_operator_is_unitary_symmetric(dim):
    qbm = QuantumBakersMap(dim=dim, alpha=0.5)
    assert _is_unitary(qbm.floquet_operator())


@pytest.mark.parametrize("dim,alpha", [(10, 0.3), (100, 0.7), (99, 1.0 / 3.0)])
def test_bakers_map_floquet_operator_is_unitary_asymmetric(dim, alpha):
    qbm = QuantumBakersMap(dim=dim, alpha=alpha)
    assert _is_unitary(qbm.floquet_operator())


def test_bakers_map_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        QuantumBakersMap(dim=10, alpha=0.0)
    with pytest.raises(ValueError):
        QuantumBakersMap(dim=10, alpha=1.0)


def test_bakers_map_rejects_dim_alpha_combo_with_empty_block():
    with pytest.raises(ValueError):
        QuantumBakersMap(dim=10, alpha=0.02)


def test_bakers_map_evolve_preserves_norm_and_dimension():
    qbm = QuantumBakersMap(dim=48)
    psi0 = qbm.coherent_state(0.2, 0.6)
    states = qbm.evolve(psi0, n_steps=6)
    assert states.shape == (7, 48)
    assert np.allclose(np.linalg.norm(states, axis=1), 1.0)


def test_bakers_map_coherent_state_husimi_peaks_at_its_own_center():
    qbm = QuantumBakersMap(dim=250)
    q0, p0 = 0.35, 0.72
    psi = qbm.coherent_state(q0, p0)
    q_grid, p_grid, husimi = qbm.husimi(psi, resolution=150)
    idx = np.unravel_index(np.argmax(husimi), husimi.shape)
    assert q_grid[idx] == pytest.approx(q0, abs=0.03)
    assert p_grid[idx] == pytest.approx(p0, abs=0.03)


def test_bakers_map_hbar_matches_classical_area_quantization():
    # dim = 1/h = 1/(2 pi hbar) states on the unit torus.
    qbm = QuantumBakersMap(dim=40)
    assert qbm.hbar == pytest.approx(1.0 / (2.0 * np.pi * 40))


def test_bakers_map_coherent_state_has_requested_momentum():
    # Momentum eigenstates on the unit torus are exp(2 pi i k q), p = k/dim; a coherent state
    # at p0 must peak at k = p0*dim (previously it peaked at p0*dim/(2 pi)).
    dim = 64
    qbm = QuantumBakersMap(dim=dim)
    psi = qbm.coherent_state(0.5, 0.25)
    momentum_amplitudes = np.fft.fft(psi)
    assert int(np.argmax(np.abs(momentum_amplitudes))) == round(0.25 * dim)
    position_spread = np.sqrt(np.sum(np.abs(psi) ** 2 * (np.arange(dim) / dim - 0.5) ** 2))
    assert position_spread == pytest.approx(np.sqrt(qbm.hbar / 2.0), rel=1e-3)  # minimum uncertainty


def test_bakers_map_eigenphases_shape_and_range():
    qbm = QuantumBakersMap(dim=40)
    phases = qbm.eigenphases()
    assert phases.shape == (40,)
    assert np.all(np.diff(phases) >= 0.0)


def test_husimi_function_is_normalized_to_unit_peak():
    n = 64
    psi = np.zeros(n, dtype=complex)
    psi[0] = 1.0
    _, _, husimi = husimi_function(psi, hbar=2.0 * np.pi / n, q_period=2.0 * np.pi, resolution=40)
    assert husimi.max() == pytest.approx(1.0)
    assert husimi.min() >= 0.0


def test_classical_bakers_map_default_alpha_matches_quantum_default():
    """physicskit.chaos's classical BakersMap defaults to alpha=0.5; the quantum
    version's default should match, so QuantumBakersMap(dim) quantizes
    BakersMap() out of the box."""
    assert BakersMap().alpha == QuantumBakersMap(dim=10).alpha
