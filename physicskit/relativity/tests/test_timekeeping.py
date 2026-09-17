import pytest

from physicskit.relativity.chapters.timekeeping import (
    EARTH_RADIUS_M,
    GPS_ORBITAL_RADIUS_M,
    clock_rate_factor,
    earth_mass_geometrized,
    gps_relativistic_offset_per_day,
)


def test_earth_mass_geometrized_is_a_few_millimeters():
    # GM_earth/c^2 is about 4.4 mm
    M = earth_mass_geometrized()
    assert pytest.approx(0.004435, abs=0.0002) == M


def test_clock_rate_factor_defaults_m_to_earth_mass_geometrized():
    rate_default = clock_rate_factor(EARTH_RADIUS_M, angular_velocity_si=0.0)
    rate_explicit = clock_rate_factor(EARTH_RADIUS_M, angular_velocity_si=0.0, M=earth_mass_geometrized())
    assert rate_default == pytest.approx(rate_explicit)


def test_clock_rate_factor_static_is_pure_gravitational_redshift():
    M = earth_mass_geometrized()
    rate = clock_rate_factor(EARTH_RADIUS_M, angular_velocity_si=0.0, M=M)
    expected = (1.0 - 2.0 * M / EARTH_RADIUS_M) ** 0.5
    assert rate == pytest.approx(expected)


def test_clock_rate_factor_less_than_one_and_decreases_with_rotation():
    M = earth_mass_geometrized()
    rate_static = clock_rate_factor(GPS_ORBITAL_RADIUS_M, 0.0, M=M)
    rate_rotating = clock_rate_factor(GPS_ORBITAL_RADIUS_M, angular_velocity_si=1.0e-4, M=M)
    assert rate_static < 1.0
    assert rate_rotating < rate_static


def test_gps_offset_matches_the_well_known_38_microseconds_per_day():
    offset = gps_relativistic_offset_per_day()
    assert offset == pytest.approx(38.6, abs=1.0)


def test_gps_offset_is_positive_satellite_runs_fast():
    assert gps_relativistic_offset_per_day() > 0.0


def test_higher_orbit_increases_the_gravitational_advantage():
    offset_low = gps_relativistic_offset_per_day(r_satellite=1.0e7)
    offset_high = gps_relativistic_offset_per_day(r_satellite=4.0e7)
    assert offset_high > offset_low
