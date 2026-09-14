import numpy as np
import pytest

from physicskit.relativity.chapters.schwarzschild import SchwarzschildBlackHole


def test_key_radii():
    bh = SchwarzschildBlackHole(M=2.0)
    assert bh.horizon_radius == pytest.approx(4.0)
    assert bh.photon_sphere_radius == pytest.approx(6.0)
    assert bh.isco_radius == pytest.approx(12.0)
    assert bh.critical_impact_parameter == pytest.approx(2.0 * 3.0 * np.sqrt(3.0))


def test_circular_orbit_angular_velocity_is_keplerian():
    bh = SchwarzschildBlackHole(M=1.0)
    r = 100.0
    assert bh.circular_orbit_angular_velocity(r) == pytest.approx(np.sqrt(1.0 / r**3))


def test_circular_orbit_stays_at_constant_radius():
    bh = SchwarzschildBlackHole(M=1.0)
    y0 = bh.circular_orbit_initial_state(r0=10.0)
    traj = bh.integrate_geodesic(y0, dtau=0.02, n_steps=2000)
    assert np.allclose(traj["r"], 10.0, atol=1e-6)


def test_light_deflection_converges_to_weak_field_formula():
    bh = SchwarzschildBlackHole(M=1.0)
    b = 50.0
    y0 = bh.null_geodesic_initial_state(r0=2.0e5, impact_parameter=b, ingoing=True)
    traj = bh.integrate_geodesic(y0, dtau=2.0, n_steps=300000)
    measured = (traj["phi"][-1] - traj["phi"][0]) - np.pi
    assert measured == pytest.approx(bh.light_deflection_angle(b), rel=0.1)


def test_light_capture_near_photon_sphere():
    bh = SchwarzschildBlackHole(M=1.0)
    # impact parameter well below the critical value: photon should plunge in
    y0 = bh.null_geodesic_initial_state(r0=100.0, impact_parameter=3.0, ingoing=True)
    traj = bh.integrate_geodesic(y0, dtau=0.05, n_steps=20000)
    assert traj["r"].min() < bh.horizon_radius * 1.05


def test_light_escape_above_critical_impact_parameter():
    bh = SchwarzschildBlackHole(M=1.0)
    y0 = bh.null_geodesic_initial_state(r0=1000.0, impact_parameter=30.0, ingoing=True)
    traj = bh.integrate_geodesic(y0, dtau=0.5, n_steps=20000)
    assert traj["r"].min() > bh.photon_sphere_radius
    assert traj["r"][-1] > traj["r"].min()


def test_weak_field_precession_converges_for_wide_low_eccentricity_orbit():
    bh = SchwarzschildBlackHole(M=1.0)
    y0 = bh.eccentric_orbit_initial_state(r0=100.0, eccentricity_boost=0.03)
    traj = bh.integrate_geodesic(y0, dtau=0.05, n_steps=300000)
    prec = bh.perihelion_precession(traj)
    assert len(prec) > 0
    e_est = (traj["r"].max() - traj["r"].min()) / (traj["r"].max() + traj["r"].min())
    a_semi = (traj["r"].max() + traj["r"].min()) / 2.0
    weak_field = bh.weak_field_precession_per_orbit(a_semi, e_est)
    assert np.mean(prec) == pytest.approx(weak_field, rel=0.15)
    assert np.mean(prec) > 0.0  # perihelion precesses forward, as GR predicts


def test_shapiro_delay_positive_and_increases_with_smaller_impact_parameter():
    bh = SchwarzschildBlackHole(M=1.0)
    delay_far = bh.shapiro_delay(r1=1000.0, r2=1000.0, impact_parameter=100.0)
    delay_close = bh.shapiro_delay(r1=1000.0, r2=1000.0, impact_parameter=10.0)
    assert delay_far > 0.0
    assert delay_close > delay_far


def test_effective_potential_has_stable_and_unstable_extrema_above_isco_momentum():
    bh = SchwarzschildBlackHole(M=1.0)
    # for L modestly above the ISCO value L^2=12M^2, V_eff should show both an
    # unstable maximum (between the photon sphere and the ISCO) and a stable
    # minimum (outside the ISCO) -- the two circular-orbit branches merging
    # into the single ISCO inflection as L -> sqrt(12)*M from above.
    L = np.sqrt(12.0) * 1.05
    r = np.linspace(3.5, 20.0, 2000)
    V = bh.effective_potential(r, L)
    idx_max = np.argmax(V[r < 6.0])
    idx_min = np.argmin(V[r > 6.0])
    r_unstable = r[r < 6.0][idx_max]
    r_stable = r[r > 6.0][idx_min]
    assert bh.photon_sphere_radius < r_unstable < bh.isco_radius
    assert r_stable > bh.isco_radius


def test_disk_redshift_factor_doppler_asymmetry():
    bh = SchwarzschildBlackHole(M=1.0)
    r = 10.0
    g_receding = bh.disk_redshift_factor(r, impact_parameter=-5.0)
    g_none = bh.disk_redshift_factor(r, impact_parameter=0.0)
    g_approaching = bh.disk_redshift_factor(r, impact_parameter=5.0)
    assert g_receding < g_none < g_approaching
    assert g_none < 1.0  # b=0: purely gravitational + orbital time dilation, no Doppler boost


def test_disk_redshift_factor_approaches_one_far_from_the_black_hole():
    bh = SchwarzschildBlackHole(M=1.0)
    assert bh.disk_redshift_factor(1.0e6, impact_parameter=0.0) == pytest.approx(1.0, abs=1.0e-4)


def test_tidal_acceleration_stretches_radially_compresses_transversely():
    bh = SchwarzschildBlackHole(M=1.0)
    radial, transverse = bh.tidal_acceleration(r=6.0, proper_separation=1.0)
    assert radial > 0.0
    assert transverse < 0.0
    assert radial == pytest.approx(-2.0 * transverse)  # 2M/r^3 vs -M/r^3: exactly a factor of 2


def test_tidal_acceleration_grows_closer_to_the_black_hole():
    bh = SchwarzschildBlackHole(M=1.0)
    radial_far, _ = bh.tidal_acceleration(r=20.0, proper_separation=1.0)
    radial_near, _ = bh.tidal_acceleration(r=4.0, proper_separation=1.0)
    assert radial_near > radial_far
