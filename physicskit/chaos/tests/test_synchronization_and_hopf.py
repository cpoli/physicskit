import numpy as np
import pytest

from physicskit.chaos.exceptions import InvalidParameterError
from physicskit.chaos.systems.bifurcations import Brusselator, HopfNormalForm
from physicskit.chaos.systems.synchronization import Kuramoto, kuramoto_order_parameter_lorentzian


def test_lorentzian_order_parameter_closed_form():
    assert kuramoto_order_parameter_lorentzian(0.9, gamma=0.5) == 0.0
    assert kuramoto_order_parameter_lorentzian(2.0, gamma=0.5) == pytest.approx(np.sqrt(0.5))


def test_critical_couplings():
    assert Kuramoto(10, gamma=0.5).critical_coupling == pytest.approx(1.0)
    assert Kuramoto(10, gamma=1.0, distribution="gaussian").critical_coupling == pytest.approx(2.0 / (np.pi / np.sqrt(2 * np.pi)))


@pytest.mark.parametrize("K", [1.5, 3.0])
def test_kuramoto_steady_state_matches_theory(K):
    r = Kuramoto(1000, K=K, gamma=0.5).order_parameter_series(t_max=120.0, seed=1)
    assert r[-400:].mean() == pytest.approx(kuramoto_order_parameter_lorentzian(K, 0.5), abs=0.02)


def test_kuramoto_incoherent_below_onset():
    r = Kuramoto(1000, K=0.5, gamma=0.5).order_parameter_series(t_max=120.0, seed=1)
    assert r[-400:].mean() < 0.1


def test_kuramoto_rhs_matches_pairwise_sum():
    model = Kuramoto(7, K=1.3, gamma=0.5)
    theta = model.initial_state(seed=4)
    pairwise = model.omega + model.K / 7 * np.sin(theta[None, :] - theta[:, None]).sum(axis=1)
    np.testing.assert_allclose(model.rhs(theta, 0.0), pairwise)


def test_kuramoto_trajectory_and_order_parameter_agree():
    model = Kuramoto(50, K=2.0, gamma=0.5)
    _, states = model.trajectory(dt=0.05, n_steps=100)
    r = model.order_parameter_series(t_max=5.0, dt=0.05)
    np.testing.assert_allclose(model.order_parameter(states), r, atol=1e-10)


def test_kuramoto_rejects_bad_distribution():
    with pytest.raises(InvalidParameterError):
        Kuramoto(10, distribution="uniform")


@pytest.mark.parametrize("mu", [0.04, 0.16])
def test_supercritical_cycle_radius_is_sqrt_mu(mu):
    system = HopfNormalForm(mu=mu)
    _, states = system.trajectory(n_steps=100000)
    assert np.hypot(*states[-1]) == pytest.approx(np.sqrt(mu), rel=1e-6)
    assert system.limit_cycle_radii() == [(pytest.approx(np.sqrt(mu)), "stable")]


def test_below_onset_spirals_into_fixed_point():
    _, states = HopfNormalForm(mu=-0.1).trajectory(state0=np.array([0.5, 0.0]), n_steps=20000)
    assert np.hypot(*states[-1]) < 1e-6


def test_subcritical_bistability():
    system = HopfNormalForm(mu=-0.1, a=1.0, c=-1.0)
    (r_u, s_u), (r_s, s_s) = system.limit_cycle_radii()
    assert (s_u, s_s) == ("unstable", "stable")
    _, inside = system.trajectory(state0=np.array([0.9 * r_u, 0.0]), n_steps=50000)
    _, outside = system.trajectory(state0=np.array([1.1 * r_u, 0.0]), n_steps=50000)
    assert np.hypot(*inside[-1]) < 1e-3
    assert np.hypot(*outside[-1]) == pytest.approx(r_s, rel=1e-6)


def test_subcritical_needs_quintic():
    with pytest.raises(InvalidParameterError):
        HopfNormalForm(a=1.0, c=0.0)


def test_brusselator_eigenvalues_cross_imaginary_axis_at_hopf_point():
    a = 1.5
    b_c = Brusselator(a=a).hopf_point
    assert b_c == pytest.approx(1 + a**2)
    lam = Brusselator(a=a, b=b_c).eigenvalues()
    np.testing.assert_allclose(lam.real, 0.0, atol=1e-12)
    np.testing.assert_allclose(sorted(lam.imag), [-a, a])
    assert np.all(Brusselator(a=a, b=b_c - 0.1).eigenvalues().real < 0)
    assert np.all(Brusselator(a=a, b=b_c + 0.1).eigenvalues().real > 0)


def test_brusselator_fixed_point_is_stationary():
    system = Brusselator(a=1.2, b=3.0)
    np.testing.assert_allclose(system.rhs(system.fixed_point, 0.0), 0.0, atol=1e-14)
