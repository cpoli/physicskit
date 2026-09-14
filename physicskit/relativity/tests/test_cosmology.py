import numpy as np
import pytest

from physicskit.relativity.chapters.cosmology import FLRWCosmology


def test_flat_universe_omega_k_zero():
    cosmo = FLRWCosmology(H0=70.0, Omega_m=0.3, Omega_r=0.0, Omega_Lambda=0.7)
    assert cosmo.Omega_k == pytest.approx(0.0)


def test_hubble_parameter_today_equals_H0():
    cosmo = FLRWCosmology(H0=67.4, Omega_m=0.315, Omega_r=0.0, Omega_Lambda=0.685)
    assert cosmo.hubble_parameter(1.0) == pytest.approx(67.4, rel=1e-6)


def test_expansion_rate_decreases_going_back_from_matter_domination():
    cosmo = FLRWCosmology(H0=70.0, Omega_m=0.3, Omega_r=0.0, Omega_Lambda=0.7)
    # E(a) should be larger in the past (smaller a) for an expanding universe
    assert cosmo.E(0.5) > cosmo.E(1.0)


def test_redshift_scale_factor_roundtrip():
    cosmo = FLRWCosmology()
    z = 2.5
    a = cosmo.redshift_to_scale_factor(z)
    assert cosmo.scale_factor_to_redshift(a) == pytest.approx(z)


def test_age_of_universe_is_physically_reasonable():
    cosmo = FLRWCosmology(H0=70.0, Omega_m=0.3, Omega_r=0.0, Omega_Lambda=0.7)
    age = cosmo.age_gyr()
    assert 12.0 < age < 15.0


def test_age_increases_with_scale_factor():
    cosmo = FLRWCosmology(H0=70.0, Omega_m=0.3, Omega_r=0.0, Omega_Lambda=0.7)
    assert cosmo.age_gyr(0.5) < cosmo.age_gyr(1.0) < cosmo.age_gyr(1.5)


def test_comoving_distance_increases_with_redshift():
    cosmo = FLRWCosmology(H0=70.0, Omega_m=0.3, Omega_r=0.0, Omega_Lambda=0.7)
    d1 = cosmo.comoving_distance_mpc(0.5)
    d2 = cosmo.comoving_distance_mpc(1.0)
    d3 = cosmo.comoving_distance_mpc(2.0)
    assert 0.0 < d1 < d2 < d3


def test_luminosity_distance_exceeds_comoving_distance_at_nonzero_redshift():
    cosmo = FLRWCosmology(H0=70.0, Omega_m=0.3, Omega_r=0.0, Omega_Lambda=0.7)
    z = 1.0
    assert cosmo.luminosity_distance_mpc(z) == pytest.approx((1.0 + z) * cosmo.comoving_distance_mpc(z))
    assert cosmo.luminosity_distance_mpc(z) > cosmo.comoving_distance_mpc(z)


def test_scale_factor_history_shape():
    cosmo = FLRWCosmology(H0=70.0, Omega_m=0.3, Omega_r=0.0, Omega_Lambda=0.7)
    t_gyr, a = cosmo.scale_factor_history(n_points=50)
    assert t_gyr.shape == (50,)
    assert a.shape == (50,)
    assert np.all(np.diff(t_gyr) > 0.0)  # age monotonically increases with scale factor
