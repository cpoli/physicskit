import numpy as np
import pytest
from scipy.integrate import quad

from physicskit.astro.galactic_dynamics import (
    circular_velocity,
    nfw_density,
    nfw_enclosed_mass,
    nfw_potential,
)


def test_nfw_enclosed_mass_matches_numerical_integral_of_density():
    rho_s, r_s = 2.0, 3.0
    for r in [1.0, 3.0, 10.0]:
        expected, _ = quad(lambda rp: 4 * np.pi * rp**2 * nfw_density(rp, rho_s, r_s), 0, r)
        assert nfw_enclosed_mass(r, rho_s, r_s) == pytest.approx(expected, rel=1e-4)


def test_nfw_enclosed_mass_monotonically_increasing():
    rho_s, r_s = 1.0, 1.0
    r = np.linspace(0.1, 20, 50)
    M = nfw_enclosed_mass(r, rho_s, r_s)
    assert np.all(np.diff(M) > 0)


def test_nfw_potential_derivative_matches_enclosed_mass():
    rho_s, r_s, G = 1.5, 2.0, 1.0
    r0 = 4.0
    dr = 1e-5
    dPhi_dr = (nfw_potential(r0 + dr, rho_s, r_s, G) - nfw_potential(r0 - dr, rho_s, r_s, G)) / (2 * dr)
    expected = G * nfw_enclosed_mass(r0, rho_s, r_s) / r0**2
    assert dPhi_dr == pytest.approx(expected, rel=1e-4)


def test_circular_velocity_point_mass_is_keplerian():
    M0, G, r = 5.0, 1.0, 3.0
    v = circular_velocity(r, lambda rr: M0, G=G)
    assert v == pytest.approx(np.sqrt(G * M0 / r))


def test_nfw_rotation_curve_is_roughly_flat_at_large_radius():
    rho_s, r_s = 1.0, 1.0
    r = np.linspace(5 * r_s, 30 * r_s, 50)
    v = circular_velocity(r, lambda rr: nfw_enclosed_mass(rr, rho_s, r_s))
    assert np.std(v) / np.mean(v) < 0.2
