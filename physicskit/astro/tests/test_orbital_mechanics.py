import numpy as np
import pytest

from physicskit.astro.orbital_mechanics import (
    hohmann_transfer,
    orbital_elements_from_state,
    orbital_period,
    state_from_orbital_elements,
    vis_viva_speed,
)


def test_orbital_period_matches_kepler_third_law():
    mu = 1.0
    a = 2.0
    T = orbital_period(a, mu)
    assert T == pytest.approx(2 * np.pi * np.sqrt(a**3 / mu))


def test_vis_viva_parabolic_limit():
    v = vis_viva_speed(r=1.0, a=np.inf, mu=1.0)
    assert v == pytest.approx(np.sqrt(2.0))


def test_vis_viva_circular_orbit_matches_period():
    mu = 1.0
    r = 3.0
    v = vis_viva_speed(r, a=r, mu=mu)
    period_from_v = 2 * np.pi * r / v
    assert period_from_v == pytest.approx(orbital_period(r, mu))


ELEMENT_CASES = [
    (1.0, 0.3, 0.5, 1.0, 0.7, 1.2),
    (2.0, 0.1, 1.2, 2.5, 0.3, 4.0),
    (1.5, 0.5, 0.9, 0.2, 2.1, 0.5),
]


@pytest.mark.parametrize("a,e,i,raan,argp,nu", ELEMENT_CASES)
def test_orbital_elements_roundtrip(a, e, i, raan, argp, nu):
    mu = 1.0
    r_vec, v_vec = state_from_orbital_elements(a, e, i, raan, argp, nu, mu)
    a2, e2, i2, raan2, argp2, nu2 = orbital_elements_from_state(r_vec, v_vec, mu)
    assert a2 == pytest.approx(a, rel=1e-8)
    assert e2 == pytest.approx(e, rel=1e-8)
    assert i2 == pytest.approx(i, rel=1e-8)
    assert raan2 == pytest.approx(raan % (2 * np.pi), rel=1e-8)
    assert argp2 == pytest.approx(argp % (2 * np.pi), rel=1e-8)
    assert nu2 == pytest.approx(nu % (2 * np.pi), rel=1e-8)


def test_hohmann_transfer_outward_deltav_matches_speed_differences():
    mu, r1, r2 = 1.0, 1.0, 4.0
    a_t = (r1 + r2) / 2.0
    dv1, dv2, t = hohmann_transfer(r1, r2, mu)
    expected_dv1 = vis_viva_speed(r1, a_t, mu) - vis_viva_speed(r1, r1, mu)
    expected_dv2 = vis_viva_speed(r2, r2, mu) - vis_viva_speed(r2, a_t, mu)
    assert dv1 == pytest.approx(abs(expected_dv1))
    assert dv2 == pytest.approx(abs(expected_dv2))
    assert dv1 > 0
    assert dv2 > 0
    assert t == pytest.approx(orbital_period(a_t, mu) / 2.0)


def test_hohmann_transfer_inward_is_positive_magnitudes():
    dv1, dv2, t = hohmann_transfer(4.0, 1.0, mu=1.0)
    assert dv1 > 0
    assert dv2 > 0
    assert t > 0
