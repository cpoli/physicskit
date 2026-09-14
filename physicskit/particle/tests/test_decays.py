import numpy as np
import pytest

from physicskit.particle.decays import (
    activity,
    bateman_decay_chain,
    decay_constant,
    half_life,
    michel_spectrum,
    muon_decay_event,
    radioactive_decay_number,
    sample_michel_electron_energies,
    two_body_decay,
    two_body_decay_momentum,
)
from physicskit.particle.kinematics import invariant_mass


def test_two_body_decay_momentum_forbidden_raises():
    with pytest.raises(ValueError):
        two_body_decay_momentum(1.0, 0.6, 0.6)


def test_two_body_decay_reconstructs_parent_mass_and_is_back_to_back():
    cases = [
        (1.0, 0.4, 0.4, 1.0, 0.0),
        (1.0, 0.4, 0.4, 0.0, 1.5),
        (2.0, 0.5, 1.0, -0.3, 2.0),
        (1.5, 0.1, 0.2, 0.7, 4.0),
    ]
    for M, m1, m2, cos_theta, phi in cases:
        p1, p2 = two_body_decay(M, m1, m2, cos_theta, phi)
        assert invariant_mass([p1, p2]) == pytest.approx(M)
        assert (p1.p_vec + p2.p_vec) == pytest.approx([0.0, 0.0, 0.0], abs=1e-10)
        assert p1.mass == pytest.approx(m1, abs=1e-8)
        assert p2.mass == pytest.approx(m2, abs=1e-8)


def test_decay_constant_half_life_roundtrip():
    t_half = 5.0
    lam = decay_constant(t_half)
    assert half_life(lam) == pytest.approx(t_half)


def test_radioactive_decay_number_matches_exponential_law():
    N0, lam, t = 1000.0, 0.05, np.linspace(0, 20, 5)
    N = radioactive_decay_number(N0, lam, t)
    assert N == pytest.approx(N0 * np.exp(-lam * t))
    assert radioactive_decay_number(N0, lam, 0.0) == pytest.approx(N0)


def test_activity():
    assert activity(200.0, 0.02) == pytest.approx(4.0)


def test_bateman_chain_species_one_matches_simple_exponential():
    N0, lam1, lam2 = 500.0, 0.3, 0.1
    t = np.linspace(0, 10, 6)
    N = bateman_decay_chain(N0, [lam1, lam2], t)
    assert N[0] == pytest.approx(radioactive_decay_number(N0, lam1, t))


def test_bateman_two_species_matches_textbook_closed_form():
    N0, lam1, lam2 = 500.0, 0.3, 0.1
    t = np.linspace(0, 10, 6)
    N = bateman_decay_chain(N0, [lam1, lam2], t)
    N2_expected = N0 * lam1 / (lam2 - lam1) * (np.exp(-lam1 * t) - np.exp(-lam2 * t))
    assert N[1] == pytest.approx(N2_expected)


def test_bateman_three_species_chain_shape_and_positivity():
    N = bateman_decay_chain(1000.0, [0.5, 0.3, 0.1], np.linspace(0, 5, 10))
    assert N.shape == (3, 10)
    assert np.all(N >= -1e-8)


def test_michel_spectrum_normalized_and_bounds():
    x = np.linspace(0, 1, 200001)
    assert np.trapezoid(michel_spectrum(x), x) == pytest.approx(1.0, abs=1e-4)
    assert michel_spectrum(0.0) == pytest.approx(0.0)
    assert michel_spectrum(1.0) == pytest.approx(2.0)
    assert np.all(michel_spectrum(x) >= -1e-12)


def test_sample_michel_electron_energies_matches_theoretical_mean():
    rng = np.random.default_rng(42)
    xs = sample_michel_electron_energies(50_000, rng=rng)
    assert xs.min() >= 0.0
    assert xs.max() <= 1.0
    # <x> = integral x * 2x^2(3-2x) dx from 0 to 1 = 2*(3/4-2/5) = 7/10
    assert xs.mean() == pytest.approx(0.7, abs=0.02)


def test_muon_decay_event_conserves_energy_momentum_and_is_massless_neutrinos():
    m_mu = 105.658
    rng = np.random.default_rng(7)
    for _ in range(20):
        p_e, p_numu_bar, p_nue = muon_decay_event(m_mu, rng=rng)
        total = p_e + p_numu_bar + p_nue
        assert total.E == pytest.approx(m_mu, abs=1e-6)
        assert total.p_vec == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
        assert p_numu_bar.mass == pytest.approx(0.0, abs=1e-6)
        assert p_nue.mass == pytest.approx(0.0, abs=1e-6)
        assert 0.0 <= p_e.E <= m_mu / 2.0 + 1e-6


def test_muon_decay_event_electron_energy_spectrum_matches_michel_mean():
    m_mu = 105.658
    rng = np.random.default_rng(3)
    E_e = np.array([muon_decay_event(m_mu, rng=rng)[0].E for _ in range(20_000)])
    x = 2.0 * E_e / m_mu
    assert x.mean() == pytest.approx(0.7, abs=0.02)
