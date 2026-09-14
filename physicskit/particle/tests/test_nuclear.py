import pytest

from physicskit.particle.nuclear import binding_energy_per_nucleon, q_value, semf_binding_energy


def test_iron_56_binding_energy_per_nucleon_near_known_maximum():
    bpn = binding_energy_per_nucleon(26, 56)
    assert 8.0 < bpn < 9.0


def test_binding_energy_per_nucleon_lower_for_heavy_nucleus():
    # Heavy nuclei (large Coulomb repulsion) are less tightly bound per
    # nucleon than the iron-group peak.
    bpn_fe = binding_energy_per_nucleon(26, 56)
    bpn_u = binding_energy_per_nucleon(92, 238)
    assert bpn_u < bpn_fe


def test_binding_energy_per_nucleon_lower_for_very_light_nucleus():
    bpn_fe = binding_energy_per_nucleon(26, 56)
    bpn_he = binding_energy_per_nucleon(2, 4)
    assert bpn_he < bpn_fe


def test_pairing_term_sign():
    # Even-even vs. odd-odd nuclei at the same A=14 differ only by the
    # pairing term: Z=6 (N=8, even-even) should be more bound than
    # Z=7 (N=7, odd-odd).
    even_even = semf_binding_energy(6, 14)
    odd_odd = semf_binding_energy(7, 14)
    assert even_even > odd_odd


def test_q_value_positive_for_dt_fusion():
    Q = q_value([2.014102, 3.016049], [4.002602, 1.008665])
    assert Q == pytest.approx(0.018884, abs=1e-6)
    assert Q > 0
