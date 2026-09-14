import numpy as np
import pytest

from physicskit.optics.quantum_optics import (
    JaynesCummingsModel,
    coherent_state,
    compute_wigner_function,
    fock_state,
    squeezed_state,
    wigner_negativity,
)


def test_fock_state_is_normalized_and_correctly_placed():
    psi = fock_state(3, 8)
    assert np.linalg.norm(psi) == pytest.approx(1.0)
    assert psi[3] == pytest.approx(1.0)
    assert np.sum(np.abs(psi) ** 2) - np.abs(psi[3]) ** 2 == pytest.approx(0.0)


def test_fock_state_raises_when_n_exceeds_cutoff():
    with pytest.raises(ValueError):
        fock_state(5, 5)


def test_coherent_state_is_normalized():
    psi = coherent_state(1.5 + 0.5j, cutoff=40)
    assert np.linalg.norm(psi) == pytest.approx(1.0, rel=1e-6)


def test_coherent_state_mean_photon_number_matches_alpha_squared():
    alpha = 1.2
    cutoff = 40
    psi = coherent_state(alpha, cutoff)
    n = np.arange(cutoff)
    mean_n = np.sum(n * np.abs(psi) ** 2)
    assert mean_n == pytest.approx(alpha**2, rel=1e-2)


def test_squeezed_state_is_normalized():
    psi = squeezed_state(xi=0.5 + 0.2j, alpha=0.3, cutoff=30)
    assert np.linalg.norm(psi) == pytest.approx(1.0, rel=1e-6)


def test_squeezed_state_reduces_to_vacuum_when_xi_and_alpha_zero():
    psi = squeezed_state(xi=0.0, alpha=0.0, cutoff=10)
    expected = fock_state(0, 10)
    assert np.allclose(psi, expected)


def test_squeezed_state_reduces_to_coherent_state_when_xi_zero():
    alpha = 0.8 - 0.3j
    cutoff = 30
    psi = squeezed_state(xi=0.0, alpha=alpha, cutoff=cutoff)
    expected = coherent_state(alpha, cutoff)
    assert np.allclose(psi, expected, atol=1e-6)


def test_wigner_function_integrates_to_one_for_coherent_state():
    cutoff = 25
    psi = coherent_state(1.0 + 0.5j, cutoff)
    xg = np.linspace(-6, 6, 121)
    pg = np.linspace(-6, 6, 121)
    W = compute_wigner_function(psi, xg, pg)
    integral = np.trapezoid(np.trapezoid(W, pg, axis=1), xg)
    assert integral == pytest.approx(1.0, rel=0.03)


def test_wigner_function_integrates_to_one_for_fock_state_one():
    cutoff = 15
    psi = fock_state(1, cutoff)
    xg = np.linspace(-6, 6, 121)
    pg = np.linspace(-6, 6, 121)
    W = compute_wigner_function(psi, xg, pg)
    integral = np.trapezoid(np.trapezoid(W, pg, axis=1), xg)
    assert integral == pytest.approx(1.0, rel=0.03)
    assert W.shape == (len(xg), len(pg))


def test_wigner_negativity_zero_for_coherent_state_positive_for_fock_one():
    xg = np.linspace(-6, 6, 121)
    pg = np.linspace(-6, 6, 121)

    psi_coh = coherent_state(1.0, 25)
    W_coh = compute_wigner_function(psi_coh, xg, pg)
    neg_coh = wigner_negativity(W_coh, xg, pg)
    assert neg_coh == pytest.approx(0.0, abs=1e-9)

    psi_fock1 = fock_state(1, 15)
    W_fock1 = compute_wigner_function(psi_fock1, xg, pg)
    neg_fock1 = wigner_negativity(W_fock1, xg, pg)
    assert neg_fock1 > 0.0


def test_fock_state_one_wigner_function_is_negative_at_origin():
    cutoff = 10
    psi = fock_state(1, cutoff)
    xg = np.linspace(-4, 4, 81)
    pg = np.linspace(-4, 4, 81)
    W = compute_wigner_function(psi, xg, pg)
    origin = (np.abs(xg).argmin(), np.abs(pg).argmin())
    assert W[origin] == pytest.approx(-1.0 / np.pi, rel=1e-3)


def test_jaynes_cummings_hamiltonian_is_hermitian():
    jc = JaynesCummingsModel(omega_c=1.0, omega_a=1.2, g=0.3, cutoff=6)
    H = jc.hamiltonian()
    assert H.shape == (12, 12)
    assert np.allclose(H, H.conj().T)


def test_resonant_vacuum_rabi_oscillation_matches_cos_squared():
    omega0 = 2.0
    g = 0.4
    cutoff = 12
    jc = JaynesCummingsModel(omega_c=omega0, omega_a=omega0, g=g, cutoff=cutoff)

    t_array = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    Pe = jc.excited_state_population(t_array, n_photons=0)
    expected = np.cos(g * t_array) ** 2

    assert np.allclose(Pe, expected, atol=1e-3)


def test_excited_state_population_conserves_probability():
    jc = JaynesCummingsModel(omega_c=1.0, omega_a=1.5, g=0.2, cutoff=8)
    psi0 = fock_state(0, 2 * jc.cutoff)  # |e, 0>
    t_array = np.linspace(0.0, 20.0, 15)
    psi_t = jc.evolve(psi0, t_array)
    norms = np.sum(np.abs(psi_t) ** 2, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-8)


def test_off_resonant_jaynes_cummings_reduces_rabi_amplitude():
    # Far off resonance, the atom barely exchanges excitation with the
    # cavity: the excited-state population should stay close to 1.
    g = 0.1
    cutoff = 10
    jc = JaynesCummingsModel(omega_c=1.0, omega_a=1.0 + 50.0 * g, g=g, cutoff=cutoff)
    t_array = np.linspace(0.0, 20.0, 50)
    Pe = jc.excited_state_population(t_array, n_photons=0)
    assert Pe.min() > 0.9
