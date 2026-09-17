import numpy as np
import pytest
from scipy.integrate import trapezoid

from physicskit.statphys.utils.thermodynamics import (
    bec_condensate_fraction,
    binder_cumulant,
    bose_einstein_occupation,
    fermi_dirac_occupation,
    maxwell_boltzmann_component_pdf,
    maxwell_boltzmann_speed_pdf,
    specific_heat,
    spin_correlation_function,
    susceptibility,
)


def test_specific_heat_zero_for_constant_energy():
    E = np.full(1000, -50.0)
    assert specific_heat(E, temperature=2.0, n_sites=64) == pytest.approx(0.0)


def test_specific_heat_positive_for_fluctuating_energy():
    rng = np.random.default_rng(0)
    E = rng.normal(-100.0, 5.0, size=5000)
    assert specific_heat(E, temperature=2.0, n_sites=64) > 0


def test_susceptibility_scales_with_variance():
    rng = np.random.default_rng(1)
    M_low = rng.normal(0.0, 1.0, size=5000)
    M_high = rng.normal(0.0, 5.0, size=5000)
    assert susceptibility(M_high, 1.0, 64) > susceptibility(M_low, 1.0, 64)


def test_binder_cumulant_gaussian_limit():
    rng = np.random.default_rng(2)
    m = rng.normal(0.0, 1.0, size=200_000)
    # for a zero-mean Gaussian, <m^4>/<m^2>^2 = 3, so U4 -> 0
    assert binder_cumulant(m) == pytest.approx(0.0, abs=0.05)


def test_binder_cumulant_delta_distribution():
    m = np.full(1000, 2.5)
    assert binder_cumulant(m) == pytest.approx(2.0 / 3.0)


def test_spin_correlation_function_zero_lag_equals_variance():
    rng = np.random.default_rng(3)
    spins = rng.choice([-1, 1], size=(20, 20))
    r, G = spin_correlation_function(spins)
    assert r[0] == 0
    assert G[0] == pytest.approx(1.0 - spins.mean() ** 2, abs=1e-9)


def test_maxwell_boltzmann_speed_pdf_normalizes_to_one():
    v = np.linspace(0, 20, 200_000)
    pdf = maxwell_boltzmann_speed_pdf(v, temperature=2.0, mass=1.0, dim=2)
    integral = trapezoid(pdf, v)
    assert integral == pytest.approx(1.0, abs=1e-3)


def test_maxwell_boltzmann_speed_pdf_invalid_dim_raises():
    with pytest.raises(ValueError):
        maxwell_boltzmann_speed_pdf(np.array([1.0]), temperature=1.0, dim=4)


@pytest.mark.parametrize("dim", [1, 3])
def test_maxwell_boltzmann_speed_pdf_normalizes_to_one_in_1d_and_3d(dim):
    v = np.linspace(0, 20, 200_000)
    pdf = maxwell_boltzmann_speed_pdf(v, temperature=2.0, mass=1.0, dim=dim)
    integral = trapezoid(pdf, v)
    assert integral == pytest.approx(1.0, abs=1e-3)


def test_maxwell_boltzmann_component_pdf_is_gaussian():
    vx = np.linspace(-10, 10, 200_000)
    pdf = maxwell_boltzmann_component_pdf(vx, temperature=1.5, mass=1.0)
    integral = trapezoid(pdf, vx)
    assert integral == pytest.approx(1.0, abs=1e-3)


def test_bose_einstein_occupation_diverges_near_ground_state():
    n_close = bose_einstein_occupation(1.001, mu=1.0, temperature=1.0)
    n_far = bose_einstein_occupation(5.0, mu=1.0, temperature=1.0)
    assert n_close > n_far > 0


def test_fermi_dirac_occupation_bounded_between_zero_and_one():
    energies = np.linspace(-5, 5, 50)
    n = fermi_dirac_occupation(energies, mu=0.0, temperature=1.0)
    assert np.all(n > 0.0) and np.all(n < 1.0)


def test_fermi_dirac_occupation_step_function_at_zero_temperature():
    # small enough to sharpen into a step, large enough that exp(x) doesn't overflow
    n_below = fermi_dirac_occupation(-1.0, mu=0.0, temperature=0.05)
    n_above = fermi_dirac_occupation(1.0, mu=0.0, temperature=0.05)
    assert n_below == pytest.approx(1.0, abs=1e-6)
    assert n_above == pytest.approx(0.0, abs=1e-6)


def test_fermi_dirac_occupation_half_at_the_fermi_energy():
    n = fermi_dirac_occupation(2.0, mu=2.0, temperature=1.0)
    assert n == pytest.approx(0.5)


def test_bec_condensate_fraction_full_at_zero_temperature():
    f = bec_condensate_fraction(0.0, critical_temperature=1.0)
    assert float(f) == pytest.approx(1.0)


def test_bec_condensate_fraction_zero_above_tc():
    f = bec_condensate_fraction(2.0, critical_temperature=1.0)
    assert float(f) == pytest.approx(0.0)


def test_bec_condensate_fraction_monotonically_decreasing():
    T = np.linspace(0.0, 1.5, 20)
    f = bec_condensate_fraction(T, critical_temperature=1.0)
    assert np.all(np.diff(f) <= 1e-12)
