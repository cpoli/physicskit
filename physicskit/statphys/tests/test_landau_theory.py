import numpy as np
import pytest

from physicskit.statphys.utils.landau_theory import (
    landau_equilibrium_magnetization,
    landau_free_energy,
    landau_susceptibility,
)


def test_equilibrium_magnetization_zero_above_tc():
    m = landau_equilibrium_magnetization(T=3.0, Tc=2.0, a=1.0, b=1.0, h=0.0)
    assert m == pytest.approx(0.0)


def test_equilibrium_magnetization_matches_closed_form_below_tc():
    T, Tc, a, b = 1.0, 2.0, 1.0, 1.0
    m = landau_equilibrium_magnetization(T=T, Tc=Tc, a=a, b=b, h=0.0)
    expected = np.sqrt(a * (Tc - T) / (2 * b))
    assert abs(m) == pytest.approx(expected, rel=1e-6)


def test_equilibrium_magnetization_vectorized_over_temperature():
    T = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    m = landau_equilibrium_magnetization(T, Tc=2.0)
    assert m.shape == T.shape
    assert np.all(m[T >= 2.0] == pytest.approx(0.0))
    assert np.all(np.abs(m[T < 2.0]) > 0.0)


def test_susceptibility_diverges_approaching_tc_from_both_sides():
    chi_far_above = landau_susceptibility(3.0, Tc=2.0)
    chi_near_above = landau_susceptibility(2.01, Tc=2.0)
    chi_far_below = landau_susceptibility(1.0, Tc=2.0)
    chi_near_below = landau_susceptibility(1.99, Tc=2.0)
    assert chi_near_above > chi_far_above
    assert chi_near_below > chi_far_below


def test_free_energy_at_zero_field_zero_magnetization_is_zero():
    assert landau_free_energy(m=0.0, T=1.0, Tc=2.0, h=0.0) == pytest.approx(0.0)


def test_equilibrium_magnetization_minimizes_free_energy():
    T, Tc, a, b, h = 1.0, 2.0, 1.0, 1.0, 0.0
    m_eq = landau_equilibrium_magnetization(T, Tc, a, b, h)
    F_eq = landau_free_energy(m_eq, T, Tc, a, b, h)
    F_zero = landau_free_energy(0.0, T, Tc, a, b, h)
    assert F_eq <= F_zero
