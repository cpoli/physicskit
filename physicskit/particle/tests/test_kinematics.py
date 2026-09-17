import numpy as np
import pytest

from physicskit.particle.kinematics import FourVector, boost, boost_generic, boost_to_com, invariant_mass, rapidity


def test_mass_of_rest_particle():
    p = FourVector(1.5, 0.0, 0.0, 0.0)
    assert p.mass == pytest.approx(1.5)
    assert p.beta == pytest.approx(0.0)


def test_boost_of_rest_frame_reproduces_boosted_energy_momentum():
    m = 1.0
    beta = 0.6
    gamma = 1.0 / np.sqrt(1 - beta**2)
    p_rest = FourVector(m, 0.0, 0.0, 0.0)
    p_boosted = boost(p_rest, beta, axis="z")
    assert p_boosted.E == pytest.approx(gamma * m)
    assert p_boosted.pz == pytest.approx(gamma * m * beta)
    assert p_boosted.mass == pytest.approx(m)


def test_boost_is_invertible():
    p = FourVector(3.0, 0.5, -0.2, 1.1)
    beta = 0.4
    for axis in ("x", "y", "z"):
        p_back = boost(boost(p, beta, axis), -beta, axis)
        assert p_back.E == pytest.approx(p.E)
        assert p_back.px == pytest.approx(p.px)
        assert p_back.py == pytest.approx(p.py)
        assert p_back.pz == pytest.approx(p.pz)


def test_invariant_mass_of_two_back_to_back_photons_like_particles():
    p1 = FourVector(1.0, 0.0, 0.0, 0.6)
    p2 = FourVector(1.0, 0.0, 0.0, -0.6)
    assert invariant_mass([p1, p2]) == pytest.approx(2.0)


def test_rapidity_is_additive_under_boost_along_same_axis():
    p = FourVector(2.0, 0.1, 0.2, 1.0)
    beta = 0.5
    y0 = rapidity(p, axis="z")
    y1 = rapidity(boost(p, beta, axis="z"), axis="z")
    assert y1 == pytest.approx(y0 + np.arctanh(beta))


def test_boost_to_com_of_symmetric_pair_is_zero():
    p1 = FourVector(2.0, 0.0, 0.0, 1.0)
    p2 = FourVector(2.0, 0.0, 0.0, -1.0)
    beta_com = boost_to_com([p1, p2])
    assert beta_com == pytest.approx([0.0, 0.0, 0.0])


def test_boost_to_com_of_asymmetric_pair():
    p1 = FourVector(2.0, 0.0, 0.0, 1.0)
    p2 = FourVector(1.0, 0.0, 0.0, 0.0)
    beta_com = boost_to_com([p1, p2])
    assert beta_com == pytest.approx([0.0, 0.0, 1.0 / 3.0])


def test_gamma_raises_for_massless_vector():
    p = FourVector(1.0, 0.0, 0.0, 1.0)
    assert p.mass == pytest.approx(0.0)
    with pytest.raises(ZeroDivisionError):
        _ = p.gamma


def test_gamma_equals_energy_over_mass_for_massive_vector():
    p = FourVector(2.0, 0.0, 0.0, 1.0)
    assert p.gamma == pytest.approx(p.E / p.mass)


def test_four_vector_repr_shows_components():
    p = FourVector(1.0, 2.0, 3.0, 4.0)
    assert repr(p) == "FourVector(E=1, px=2, py=3, pz=4)"


def test_boost_rejects_invalid_beta_and_axis():
    p = FourVector(1.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="beta"):
        boost(p, 1.5)
    with pytest.raises(ValueError, match="axis"):
        boost(p, 0.5, axis="w")


def test_boost_generic_matches_axis_boost_when_aligned():
    p = FourVector(3.0, 0.5, -0.2, 1.1)
    beta = 0.45
    for axis, vec in [("x", [beta, 0, 0]), ("y", [0, beta, 0]), ("z", [0, 0, beta])]:
        pb_axis = boost(p, beta, axis=axis)
        pb_generic = boost_generic(p, vec)
        assert pb_generic.E == pytest.approx(pb_axis.E)
        assert pb_generic.px == pytest.approx(pb_axis.px)
        assert pb_generic.py == pytest.approx(pb_axis.py)
        assert pb_generic.pz == pytest.approx(pb_axis.pz)


def test_boost_generic_of_rest_particle_along_arbitrary_direction():
    m = 2.0
    beta_vec = np.array([0.3, -0.4, 0.2])
    beta = np.linalg.norm(beta_vec)
    gamma = 1.0 / np.sqrt(1 - beta**2)
    p_rest = FourVector(m, 0.0, 0.0, 0.0)
    pb = boost_generic(p_rest, beta_vec)
    assert pb.E == pytest.approx(gamma * m)
    assert pb.p_vec == pytest.approx(gamma * m * beta_vec)
    assert pb.mass == pytest.approx(m)


def test_boost_generic_is_invertible():
    p = FourVector(3.0, 0.5, -0.2, 1.1)
    beta_vec = np.array([0.1, 0.2, -0.3])
    p_back = boost_generic(boost_generic(p, beta_vec), -beta_vec)
    assert p_back.E == pytest.approx(p.E)
    assert p_back.p_vec == pytest.approx(p.p_vec)


def test_boost_generic_zero_velocity_is_identity():
    p = FourVector(3.0, 0.5, -0.2, 1.1)
    pb = boost_generic(p, [0.0, 0.0, 0.0])
    assert pb.E == pytest.approx(p.E)
    assert pb.p_vec == pytest.approx(p.p_vec)


def test_boost_generic_rejects_superluminal_velocity():
    with pytest.raises(ValueError):
        boost_generic(FourVector(1.0, 0, 0, 0), [0.9, 0.9, 0.0])


def test_add_and_sub():
    p1 = FourVector(1.0, 0.1, 0.2, 0.3)
    p2 = FourVector(2.0, -0.1, 0.0, 0.1)
    s = p1 + p2
    assert (s.E, s.px, s.py, s.pz) == pytest.approx((3.0, 0.0, 0.2, 0.4))
    d = p1 - p2
    assert (d.E, d.px, d.py, d.pz) == pytest.approx((-1.0, 0.2, 0.2, 0.2))
