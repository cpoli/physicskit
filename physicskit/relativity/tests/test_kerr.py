import numpy as np
import pytest

from physicskit.relativity.chapters.kerr import KerrBlackHole


def test_schwarzschild_limit_a_zero():
    bh = KerrBlackHole(M=2.0, a=0.0)
    assert bh.outer_horizon_radius == pytest.approx(4.0)
    assert bh.inner_horizon_radius == pytest.approx(0.0)
    assert bh.ergosphere_radius(np.pi / 2.0) == pytest.approx(4.0)
    assert bh.isco_radius(prograde=True) == pytest.approx(12.0)
    assert bh.isco_radius(prograde=False) == pytest.approx(12.0)


def test_invalid_spin_raises():
    with pytest.raises(ValueError):
        KerrBlackHole(M=1.0, a=1.0)
    with pytest.raises(ValueError):
        KerrBlackHole(M=1.0, a=-0.1)


def test_prograde_isco_smaller_than_retrograde():
    bh = KerrBlackHole(M=1.0, a=0.9)
    assert bh.isco_radius(prograde=True) < 6.0
    assert bh.isco_radius(prograde=False) > 6.0
    assert bh.isco_radius(prograde=True) < bh.isco_radius(prograde=False)


def test_isco_approaches_horizon_for_extremal_spin():
    # convergence to r_ISCO -> M is slow (a power law in 1-a/M) near extremality,
    # so this only checks the qualitative trend, not tight numerical convergence.
    moderate = KerrBlackHole(M=1.0, a=0.9)
    near_extremal = KerrBlackHole(M=1.0, a=0.9999)
    assert moderate.isco_radius(prograde=True) > near_extremal.isco_radius(prograde=True) > 1.0
    assert near_extremal.isco_radius(prograde=True) == pytest.approx(1.0, abs=0.15)


def test_ergosphere_contains_horizon_and_widest_at_equator():
    bh = KerrBlackHole(M=1.0, a=0.8)
    assert bh.ergosphere_radius(np.pi / 2.0) == pytest.approx(2.0)
    assert bh.ergosphere_radius(0.0) == pytest.approx(bh.outer_horizon_radius)
    assert bh.ergosphere_radius(np.pi / 2.0) > bh.outer_horizon_radius


def test_is_inside_ergosphere():
    bh = KerrBlackHole(M=1.0, a=0.8)
    r_mid = (bh.outer_horizon_radius + bh.ergosphere_radius(np.pi / 2.0)) / 2.0
    assert bh.is_inside_ergosphere(r_mid, np.pi / 2.0)
    assert not bh.is_inside_ergosphere(bh.ergosphere_radius(np.pi / 2.0) + 1.0, np.pi / 2.0)


def test_frame_dragging_positive_and_decays_at_large_r():
    bh = KerrBlackHole(M=1.0, a=0.9)
    r = np.array([3.0, 10.0, 100.0, 1000.0])
    omega = bh.frame_dragging_angular_velocity(r)
    assert np.all(omega > 0.0)
    assert np.all(np.diff(omega) < 0.0)  # monotonically decreasing with r
    # asymptotic Lense-Thirring rate omega ~ 2Ma/r^3
    assert omega[-1] == pytest.approx(2.0 * bh.M * bh.a / r[-1] ** 3, rel=1e-3)


def test_max_penrose_efficiency_bounds():
    schwarzschild_limit = KerrBlackHole(M=1.0, a=0.0)
    assert schwarzschild_limit.max_penrose_efficiency() == pytest.approx(0.0)

    extremal = KerrBlackHole(M=1.0, a=0.999999)
    assert extremal.max_penrose_efficiency() == pytest.approx(1.0 - 1.0 / np.sqrt(2.0), abs=1e-3)


def test_penrose_energy_gain():
    bh = KerrBlackHole(M=1.0, a=0.9)
    e_out = bh.penrose_energy_gain(initial_energy=1.0, fragment_energy_infalling=-0.1)
    assert e_out == pytest.approx(1.1)
    with pytest.raises(ValueError):
        bh.penrose_energy_gain(initial_energy=1.0, fragment_energy_infalling=0.1)


def test_circular_orbit_conserved_quantities_reduce_to_schwarzschild():
    bh = KerrBlackHole(M=1.0, a=1e-8)
    E, L = bh.circular_orbit_conserved_quantities(r0=10.0, prograde=True)
    r = 10.0
    # Schwarzschild circular-orbit specific energy and angular momentum
    E_schw = (1.0 - 2.0 / r) / np.sqrt(1.0 - 3.0 / r)
    L_schw = np.sqrt(r) / np.sqrt(1.0 - 3.0 / r)
    assert pytest.approx(E_schw, rel=1e-5) == E
    assert pytest.approx(L_schw, rel=1e-5) == L


def test_keplerian_angular_velocity_reduces_to_schwarzschild_at_zero_spin():
    bh = KerrBlackHole(M=1.0, a=0.0)
    r = 10.0
    assert bh.keplerian_angular_velocity(r, prograde=True) == pytest.approx(np.sqrt(1.0 / r**3))


def test_keplerian_angular_velocity_retrograde_requires_larger_magnitude():
    # Frame dragging effectively supplies "free" angular velocity to a
    # prograde orbiter, so it needs a smaller coordinate Omega to stay on a
    # circular geodesic at the same r; a retrograde orbiter fights the
    # dragging and needs a larger |Omega|.
    bh = KerrBlackHole(M=1.0, a=0.9)
    omega_pro = bh.keplerian_angular_velocity(10.0, prograde=True)
    omega_retro = bh.keplerian_angular_velocity(10.0, prograde=False)
    assert omega_pro > 0.0
    assert omega_retro < 0.0
    assert abs(omega_retro) > omega_pro


def test_kerr_disk_redshift_factor_doppler_asymmetry():
    bh = KerrBlackHole(M=1.0, a=0.7)
    r = 10.0
    g_receding = bh.disk_redshift_factor(r, impact_parameter=-5.0)
    g_none = bh.disk_redshift_factor(r, impact_parameter=0.0)
    g_approaching = bh.disk_redshift_factor(r, impact_parameter=5.0)
    assert g_receding < g_none < g_approaching
    assert g_none < 1.0
