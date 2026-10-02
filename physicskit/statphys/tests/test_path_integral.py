import pytest

from physicskit.statphys.chapters.path_integral import (
    PathIntegralParticle,
    harmonic_x2_exact,
    harmonic_x2_primitive,
)


def test_exact_x2_limits():
    assert harmonic_x2_exact(1e3, omega=2.0) == pytest.approx(0.25)
    # classical limit: <x^2> -> 1/(beta omega^2)
    assert harmonic_x2_exact(1e-3) == pytest.approx(1e3, rel=1e-6)


def test_primitive_action_converges_to_exact_as_tau_squared():
    beta = 4.0
    exact = harmonic_x2_exact(beta)
    err = [abs(harmonic_x2_primitive(beta, P) - exact) for P in (16, 32, 64)]
    assert err[0] / err[1] == pytest.approx(4.0, rel=0.05)
    assert err[1] / err[2] == pytest.approx(4.0, rel=0.05)


def test_single_bead_is_classical():
    assert harmonic_x2_primitive(2.0, 1) == pytest.approx(0.5)


@pytest.mark.parametrize("beta", [0.5, 5.0])
def test_pimc_matches_discrete_ring_polymer(beta):
    out = PathIntegralParticle(beta, n_beads=16, seed=2).run(n_equil=2000, n_measure=60000)
    target = harmonic_x2_primitive(beta, 16)
    assert out["x2"] == pytest.approx(target, rel=0.05)
    # virial estimator for a harmonic well: E = omega^2 <x^2>
    assert out["energy"] == pytest.approx(out["x2"])
    assert 0.2 < out["acceptance"] < 0.9


def test_anharmonic_ground_state_energy():
    # E0 of x^2/2 + x^4 is 0.803771 (Hioe and Montroll 1975)
    out = PathIntegralParticle(10.0, n_beads=128, lam=1.0, seed=4).run(n_equil=2000, n_measure=60000)
    assert out["energy"] == pytest.approx(0.803771, rel=0.03)


def test_negative_quartic_rejected():
    with pytest.raises(ValueError):
        PathIntegralParticle(1.0, lam=-1.0)
