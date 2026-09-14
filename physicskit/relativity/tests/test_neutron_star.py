import numpy as np
import pytest

from physicskit.relativity.chapters.neutron_star import NeutronStar


def test_solve_gives_positive_mass_and_radius():
    star = NeutronStar(K=100.0, Gamma=2.0)
    M, R, sol = star.solve(central_density=1.28e-3)
    assert M > 0.0
    assert R > 0.0
    assert sol.success


def test_known_k100_gamma2_benchmark():
    # Standard numerical-relativity textbook benchmark (e.g. Baumgarte &
    # Shapiro): a K=100, Gamma=2 polytrope has a maximum mass of
    # approximately 1.6-1.7 (geometrized units matching this convention),
    # near a central density of a few x 1e-3.
    star = NeutronStar(K=100.0, Gamma=2.0)
    M, R, _ = star.solve(central_density=1.28e-3)
    assert pytest.approx(1.4, abs=0.1) == M
    assert pytest.approx(9.6, abs=1.0) == R


def test_mass_radius_curve_has_a_maximum_mass_turning_point():
    star = NeutronStar(K=100.0, Gamma=2.0)
    central_densities = np.logspace(-3.3, -2.2, 14)
    masses, radii = star.mass_radius_curve(central_densities)
    assert np.all(masses > 0.0)
    assert np.all(radii > 0.0)
    # radius should decrease monotonically as central density increases
    # (denser, more compact stars)
    assert np.all(np.diff(radii) < 0.0)
    # a genuine maximum mass exists strictly inside the scanned range
    assert 0 < np.argmax(masses) < len(masses) - 1


def test_higher_central_density_gives_smaller_radius():
    star = NeutronStar(K=100.0, Gamma=2.0)
    _M_low, R_low, _ = star.solve(central_density=5.0e-4)
    _M_high, R_high, _ = star.solve(central_density=1.0e-3)
    assert R_high < R_low


def test_energy_density_reduces_to_rest_mass_at_zero_pressure():
    star = NeutronStar(K=100.0, Gamma=2.0)
    assert star.energy_density(0.0) == pytest.approx(0.0)
