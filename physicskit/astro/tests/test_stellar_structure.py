import numpy as np
import pytest

from physicskit.astro.stellar_structure import (
    PolytropicStar,
    chandrasekhar_mass,
    lane_emden,
    main_sequence_luminosity,
)


def test_lane_emden_n0_matches_analytic_solution():
    xi, theta = lane_emden(0.0)
    assert xi[-1] == pytest.approx(np.sqrt(6.0), rel=1e-3)
    # analytic: theta(xi) = 1 - xi^2/6
    analytic = 1.0 - xi**2 / 6.0
    assert theta == pytest.approx(analytic, abs=2e-3)


def test_lane_emden_n1_matches_analytic_solution():
    xi, theta = lane_emden(1.0)
    assert xi[-1] == pytest.approx(np.pi, rel=1e-3)
    # analytic: theta(xi) = sin(xi)/xi
    analytic = np.sin(xi) / xi
    assert theta == pytest.approx(analytic, abs=2e-3)


def test_polytropic_star_n0_matches_uniform_sphere_mass():
    star = PolytropicStar(n=0.0, K=2.5, rho_c=3.0)
    uniform_mass = 4.0 / 3.0 * np.pi * star.radius**3 * star.rho_c
    assert star.mass == pytest.approx(uniform_mass, rel=1e-2)


def test_polytropic_star_positive_physical_quantities():
    star = PolytropicStar(n=1.5, K=1.0, rho_c=1.0)
    assert star.radius > 0
    assert star.mass > 0
    assert star.xi1 > 0


def test_chandrasekhar_mass_near_known_value():
    M = chandrasekhar_mass(2.0)
    assert 1.2 < M < 1.5


def test_chandrasekhar_mass_scales_as_inverse_mu_e_squared():
    assert chandrasekhar_mass(4.0) == pytest.approx(chandrasekhar_mass(2.0) / 4.0)


def test_main_sequence_luminosity_power_law():
    assert main_sequence_luminosity(2.0) == pytest.approx(2.0**3.5)
    assert main_sequence_luminosity(1.0) == pytest.approx(1.0)
