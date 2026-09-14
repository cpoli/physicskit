import numpy as np
import pytest

from physicskit.particle.neutrinos import oscillation_probability, survival_probability


def test_oscillation_probability_vanishes_at_zero_baseline():
    assert oscillation_probability(0.0, 1.0, theta=0.4, delta_m2=2.5e-3) == pytest.approx(0.0)


def test_oscillation_probability_reaches_maximal_value_at_maximal_mixing():
    theta = np.pi / 4  # sin^2(2 theta) = 1
    delta_m2 = 2.5e-3
    E = 1.0
    L = (np.pi / 2) / (1.267 * delta_m2 / E)  # argument = pi/2 -> sin^2 = 1
    assert oscillation_probability(L, E, theta, delta_m2) == pytest.approx(1.0, abs=1e-8)


def test_probabilities_sum_to_one():
    L = np.linspace(0, 2000, 50)
    P_mu = oscillation_probability(L, E=1.0, theta=0.5, delta_m2=2.5e-3)
    P_e = survival_probability(L, E=1.0, theta=0.5, delta_m2=2.5e-3)
    assert P_mu + P_e == pytest.approx(np.ones_like(L))


def test_no_mixing_gives_no_oscillation():
    L = np.linspace(0, 1000, 20)
    P_mu = oscillation_probability(L, E=1.0, theta=0.0, delta_m2=2.5e-3)
    assert P_mu == pytest.approx(np.zeros_like(L))


def test_oscillation_probability_is_periodic_in_L():
    theta, delta_m2, E = 0.5, 2.5e-3, 1.0
    period = np.pi * E / (1.267 * delta_m2)
    L0 = 137.0
    assert oscillation_probability(L0, E, theta, delta_m2) == pytest.approx(oscillation_probability(L0 + period, E, theta, delta_m2))
