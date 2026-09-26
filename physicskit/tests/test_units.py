"""Tests for physicskit.units."""

import json
import sys

import numpy as np
import pytest
import scipy.constants as sc

import physicskit.constants as const
from physicskit import units as u

SOLAR = const.SOLAR_MASS_KG
BOHR = sc.physical_constants["Bohr radius"][0]
AMU = sc.physical_constants["atomic mass constant"][0]


# ---------------------------------------------------------------------------
# Known conversions (closed forms)
# ---------------------------------------------------------------------------


def test_geometrized_solar_mass_length_and_time():
    sun = u.geometrized_units(mass_kg=SOLAR)
    # G M / c^2 and G M / c^3 (MTW Box 1.8)
    assert sun.scale("length") == pytest.approx(const.G * SOLAR / const.C**2, rel=1e-12)
    assert sun.scale("time") == pytest.approx(const.G * SOLAR / const.C**3, rel=1e-12)
    # The textbook value 1476.6 m (to the precision of M_sun)
    assert sun.scale("length") == pytest.approx(1476.6, rel=1e-4)
    assert sun.scale("mass") == pytest.approx(SOLAR, rel=1e-12)


def test_geometrized_from_length_matches_from_mass():
    by_mass = u.geometrized_units(mass_kg=SOLAR)
    by_length = u.geometrized_units(length_m=by_mass.scale("length"))
    for dim in ["mass", "time", "energy", "momentum"]:
        assert by_length.scale(dim) == pytest.approx(by_mass.scale(dim), rel=1e-12)


def test_geometrized_sets_G_and_c_to_one():
    sun = u.geometrized_units(mass_kg=SOLAR)
    for name in ["G", "c"]:
        dim, value = u.NATURAL_CONSTANTS[name]
        assert sun.from_si(value, dim) == pytest.approx(1.0, rel=1e-12)


def test_astro_matches_constants_gravitational_unit_system():
    ref = const.gravitational_unit_system(length_m=const.PARSEC_M, mass_kg=SOLAR)
    sys_ = u.astro_units(length_m=const.PARSEC_M, mass_kg=SOLAR)
    assert sys_.scale("time") == pytest.approx(ref["time_s"], rel=1e-12)
    assert sys_.scale("velocity") == pytest.approx(ref["velocity_m_s"], rel=1e-12)


def test_astro_earth_orbit_period_is_one_year():
    # Kepler's third law with G=1, a=1 AU, M=1 Msun: P = 2 pi sqrt(a^3/(G M))
    sys_ = u.astro_units(length_m=const.ASTRONOMICAL_UNIT_M, mass_kg=SOLAR)
    period_s = sys_.to_si(2 * np.pi, "time")
    assert period_s / (365.25 * 86400) == pytest.approx(1.0, rel=2e-4)


def test_quantum_units_reproduce_hartree_atomic_units():
    au = u.quantum_units(mass_kg=const.ELECTRON_MASS, length_m=BOHR)
    # E_h = hbar^2 / (m_e a_0^2); t_au = hbar / E_h
    assert au.scale("energy") == pytest.approx(sc.physical_constants["Hartree energy"][0], rel=1e-9)
    assert au.scale("time") == pytest.approx(sc.physical_constants["atomic unit of time"][0], rel=1e-9)
    assert au.scale("velocity") == pytest.approx(sc.physical_constants["atomic unit of velocity"][0], rel=1e-9)


def test_quantum_units_from_energy_matches_from_length():
    by_length = u.quantum_units(mass_kg=const.ELECTRON_MASS, length_m=BOHR)
    by_energy = u.quantum_units(mass_kg=const.ELECTRON_MASS, energy_j=by_length.scale("energy"))
    assert by_energy.scale("length") == pytest.approx(BOHR, rel=1e-12)


def test_statphys_temperature_scale_is_energy_over_kB():
    sys_ = u.statphys_units(temperature_k=300.0)
    assert sys_.scale("energy") == pytest.approx(const.K_B * 300.0, rel=1e-12)
    same = u.statphys_units(energy_j=const.K_B * 300.0)
    assert same.scale("temperature") == pytest.approx(300.0, rel=1e-12)
    # With k_B = 1, entropy is dimensionless in natural units: one unit = k_B.
    assert sys_.scale("entropy") == pytest.approx(const.K_B, rel=1e-12)


def test_statphys_lennard_jones_time_unit():
    sigma, eps_over_kB, m = 3.405e-10, 119.8, 39.948 * AMU
    sys_ = u.statphys_units(temperature_k=eps_over_kB, length_m=sigma, mass_kg=m)
    tau = sigma * np.sqrt(m / (const.K_B * eps_over_kB))  # Allen & Tildesley App. B
    assert sys_.scale("time") == pytest.approx(tau, rel=1e-12)
    assert sys_.scale("time") == pytest.approx(2.156e-12, rel=1e-3)


def test_hbar_c_one_gev_length_is_hbar_c_over_gev():
    gev = u.natural_units("hbar", "c", energy_j=1e9 * const.ELECTRONVOLT)
    hbar_c_mev_fm = sc.physical_constants["reduced Planck constant times c in MeV fm"][0]
    assert gev.scale("length") * 1e15 == pytest.approx(hbar_c_mev_fm / 1000.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------

SYSTEMS = {
    "astro": u.astro_units(length_m=const.PARSEC_M, mass_kg=SOLAR),
    "relativity": u.geometrized_units(mass_kg=10 * SOLAR),
    "quantum": u.quantum_units(mass_kg=const.ELECTRON_MASS, length_m=BOHR),
    "statphys": u.statphys_units(temperature_k=119.8, length_m=3.405e-10, mass_kg=39.948 * AMU),
}


@pytest.mark.parametrize("name", sorted(SYSTEMS))
@pytest.mark.parametrize("dim", ["length", "mass", "time", "velocity", "energy", "pressure", "action"])
def test_round_trip_scalar_and_array(name, dim):
    sys_ = SYSTEMS[name]
    x = np.array([0.0, 1.5, -3.25, 1e6])
    assert sys_.from_si(sys_.to_si(x, dim), dim) == pytest.approx(x, rel=1e-12)
    assert u.from_si(u.to_si(2.5, dim, sys_), dim, sys_) == pytest.approx(2.5, rel=1e-12)
    assert isinstance(sys_.to_si(2.5, dim), float)


def test_to_si_accepts_lists_and_dimension_objects():
    sys_ = SYSTEMS["relativity"]
    out = sys_.to_si([1.0, 2.0], u.Dimension(length=1))
    assert isinstance(out, np.ndarray)
    assert out[1] == pytest.approx(2 * sys_.scale("length"))


def test_to_dict_from_dict_round_trip_through_json():
    for sys_ in SYSTEMS.values():
        again = u.UnitSystem.from_dict(json.loads(json.dumps(sys_.to_dict())))
        assert again == sys_


# ---------------------------------------------------------------------------
# Dimension algebra and validation
# ---------------------------------------------------------------------------


def test_dimension_algebra():
    E = u.DIMENSIONS["energy"]
    assert E / u.DIMENSIONS["time"] == u.DIMENSIONS["power"]
    assert u.DIMENSIONS["length"] ** 3 == u.DIMENSIONS["volume"]
    assert u.DIMENSIONS["action"] * u.DIMENSIONS["frequency"] == E
    assert str(E) == "L^2 M^1 T^-2"
    assert str(u.DIMENSIONS["dimensionless"]) == "1"


def test_dimensionless_is_always_one():
    sys_ = u.statphys_units(energy_j=1.0)
    assert sys_.scale("dimensionless") == 1.0
    assert u.UnitSystem("empty", ()).scale("dimensionless") == 1.0


def test_undetermined_dimension_raises():
    ising = u.statphys_units(energy_j=1e-21)
    assert not ising.is_determined("length")
    with pytest.raises(ValueError, match="does not fix"):
        ising.to_si(1.0, "time")
    assert not u.UnitSystem("empty", ()).is_determined("length")


def test_relativity_temperature_is_undetermined():
    assert not u.geometrized_units(mass_kg=SOLAR).is_determined("temperature")


def test_overdetermined_inconsistent_raises():
    with pytest.raises(ValueError, match="overdetermined"):
        u.natural_units("c", length_m=1.0, time_s=1.0)


def test_redundant_consistent_anchors_are_accepted():
    sys_ = u.natural_units("c", length_m=const.C, time_s=1.0)
    assert sys_.scale("velocity") == pytest.approx(const.C)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_nonpositive_anchor_raises(bad):
    with pytest.raises(ValueError, match="positive finite"):
        u.natural_units("G", length_m=bad, mass_kg=1.0)


def test_unknown_names_raise():
    with pytest.raises(ValueError, match="unknown constant"):
        u.natural_units("e")
    with pytest.raises(ValueError, match="unknown dimension"):
        SYSTEMS["astro"].scale("luminosity")


@pytest.mark.parametrize(
    "build",
    [
        lambda: u.geometrized_units(),
        lambda: u.geometrized_units(mass_kg=1.0, length_m=1.0),
        lambda: u.quantum_units(mass_kg=1.0),
        lambda: u.statphys_units(energy_j=1.0, temperature_k=1.0),
    ],
)
def test_presets_require_exactly_one_free_scale(build):
    with pytest.raises(ValueError, match="exactly one"):
        build()


def test_default_name_lists_constants():
    assert u.natural_units("hbar", "c", energy_j=1.0).name == "hbar=c=1"


@pytest.mark.parametrize("subpackage", sorted(u.SUBPACKAGE_CONVENTIONS))
def test_subpackage_conventions_match_presets(subpackage):
    conv = u.SUBPACKAGE_CONVENTIONS[subpackage]
    sys_ = SYSTEMS[subpackage]
    assert hasattr(u, conv["preset"])
    for name in conv["constants"]:
        dim, value = u.NATURAL_CONSTANTS[name]
        assert sys_.from_si(value, dim) == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# pint interop (optional)
# ---------------------------------------------------------------------------


def test_pint_missing_raises_helpful_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pint", None)
    with pytest.raises(ImportError, match=r"physicskit\[units\]"):
        u.to_pint(1.0, "length", SYSTEMS["relativity"])


def test_pint_round_trip():
    pint = pytest.importorskip("pint")
    ureg = pint.UnitRegistry()
    sun = u.geometrized_units(mass_kg=SOLAR)
    q = u.to_pint(1.0, "length", sun, registry=ureg)
    assert q.to("km").magnitude == pytest.approx(const.G * SOLAR / const.C**2 / 1e3, rel=1e-12)
    assert u.from_pint(q, sun) == pytest.approx(1.0, rel=1e-12)
    # Dimension is read from the quantity: 1 ms in units of G M_sun / c^3
    assert u.from_pint(1.0 * ureg.millisecond, sun) == pytest.approx(1e-3 / sun.scale("time"), rel=1e-12)
    # Composite and non-SI input units
    energy = u.from_pint(1.0 * ureg.hartree, SYSTEMS["quantum"])
    assert energy == pytest.approx(1.0, rel=1e-8)
    dimless = u.to_pint(3.0, "dimensionless", sun, registry=ureg)
    assert dimless.magnitude == pytest.approx(3.0)


def test_pint_rejects_unsupported_dimension():
    pint = pytest.importorskip("pint")
    ureg = pint.UnitRegistry()
    with pytest.raises(ValueError, match="current"):
        u.from_pint(1.0 * ureg.ampere, SYSTEMS["quantum"])
