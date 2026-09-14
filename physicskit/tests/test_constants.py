"""Tests for physicskit.constants."""

import pytest

import physicskit.constants as const


def test_speed_of_light_exact():
    assert const.C == 299_792_458.0


def test_energy_temperature_roundtrip():
    T = 300.0
    E = const.temperature_to_energy(T)
    assert const.energy_to_temperature(E) == pytest.approx(T)


def test_ev_joule_roundtrip():
    E_ev = 13.6
    assert const.joules_to_ev(const.ev_to_joules(E_ev)) == pytest.approx(E_ev)


def test_electronvolt_matches_elementary_charge():
    # 1 eV is, by definition, the elementary charge times 1 volt.
    assert const.ELECTRONVOLT == pytest.approx(const.ELEMENTARY_CHARGE)


def test_gravitational_unit_system_consistent_with_G():
    units = const.gravitational_unit_system(length_m=const.PARSEC_M, mass_kg=const.SOLAR_MASS_KG)
    # G_SI = length_m^3 / (mass_kg * time_s^2) should reproduce G by construction.
    recovered_G = units["length_m"] ** 3 / (units["mass_kg"] * units["time_s"] ** 2)
    assert recovered_G == pytest.approx(const.G)
    assert units["velocity_m_s"] == pytest.approx(units["length_m"] / units["time_s"])


def test_gravitational_unit_system_rejects_nonpositive_inputs():
    with pytest.raises(ZeroDivisionError):
        const.gravitational_unit_system(length_m=1.0, mass_kg=0.0)
