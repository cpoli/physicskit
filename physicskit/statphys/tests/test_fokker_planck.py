"""1D Fokker-Planck solver against closed forms and against Langevin ensembles."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from physicskit.statphys import BrownianMotion, fokker_planck_1d, fokker_planck_operator, fokker_planck_stationary


def _moments(x, p):
    dx = x[1] - x[0]
    mean = np.sum(x * p) * dx
    return mean, np.sum((x - mean) ** 2 * p) * dx


def test_free_diffusion_variance_grows_as_2Dt():
    x = np.linspace(-15, 15, 601)
    D = 0.7
    t, p = fokker_planck_1d(np.exp(-(x**2) / 0.5), x, 0.0, D, t_max=3.0, dt=0.005, n_frames=6)
    var = np.array([_moments(x, pk)[1] for pk in p])
    assert var == pytest.approx(0.25 + 2 * D * t, rel=2e-3)


def test_probability_is_conserved_and_stays_positive():
    x = np.linspace(-3, 3, 121)
    p0 = np.where(np.abs(x - 1) < 0.3, 1.0, 0.0)
    _, p = fokker_planck_1d(p0, x, lambda y: -2 * y**3, 0.3, t_max=2.0, dt=0.01, n_frames=10, theta=1.0)
    dx = x[1] - x[0]
    assert p.sum(axis=1) * dx == pytest.approx(np.ones(11), abs=1e-12)
    assert p.min() > -1e-10


def test_ornstein_uhlenbeck_relaxes_to_stationary_gaussian():
    """A = theta (mu - x), D = sigma^2/2: mean relaxes as e^{-theta t}, variance to sigma^2/(2 theta)."""
    theta, mu, sigma = 1.2, 0.5, 0.8
    x = np.linspace(-5, 5, 401)
    p0 = np.exp(-((x - 3.0) ** 2) / (2 * 0.01))
    t, p = fokker_planck_1d(p0, x, lambda y: theta * (mu - y), sigma**2 / 2, t_max=6.0, dt=0.005, n_frames=12)
    means = np.array([_moments(x, pk)[0] for pk in p])
    assert means[1:] == pytest.approx(mu + (3.0 - mu) * np.exp(-theta * t[1:]), abs=2e-3)
    assert _moments(x, p[-1])[1] == pytest.approx(sigma**2 / (2 * theta), rel=5e-3)


def test_double_well_relaxes_to_boltzmann():
    """Overdamped particle in U = (x^2 - 1)^2: p_st = e^{-U/kT}/Z, from both the
    closed form and long-time evolution."""
    gamma, kT = 1.0, 0.4
    x = np.linspace(-2.2, 2.2, 441)
    U = (x**2 - 1) ** 2
    boltz = np.exp(-U / kT)
    boltz /= trapezoid(boltz, x)
    drift = lambda y: -4 * y * (y**2 - 1) / gamma
    st = fokker_planck_stationary(x, drift, kT / gamma)
    assert st == pytest.approx(boltz, rel=1e-3, abs=1e-6)  # trapezoid-rule error in the exponent
    _, p = fokker_planck_1d(np.exp(-((x - 1.0) ** 2) / 0.02), x, drift, kT / gamma, t_max=60.0, dt=0.02, n_frames=3)
    assert np.max(np.abs(p[-1] - boltz)) < 5e-3 * boltz.max()


def test_matches_brownian_ensemble_in_a_trap():
    """The density of a Langevin ensemble is the Fokker-Planck solution (mean and variance)."""
    k, gamma, kT = 2.0, 1.0, 0.5
    bm = BrownianMotion(n_particles=40000, gamma=gamma, kT=kT, stiffness=k, seed=4)
    _, xs = bm.run(t_max=0.5, dt=0.001, n_frames=1, x0=np.full(40000, 1.0))
    x = np.linspace(-3, 4, 701)
    p0 = np.exp(-((x - 1.0) ** 2) / (2 * 0.02**2))
    _, p = fokker_planck_1d(p0, x, lambda y: -k * y / gamma, kT / gamma, t_max=0.5, dt=0.001, n_frames=1)
    mean, var = _moments(x, p[-1])
    # exact OU transient from x0 = 1 (initial width 0.02 in the FP run)
    decay = np.exp(-k * 0.5 / gamma)
    assert mean == pytest.approx(decay, rel=1e-3)
    assert var == pytest.approx(kT / k * (1 - decay**2) + 0.02**2 * decay**2, rel=5e-3)
    assert xs[-1].mean() == pytest.approx(mean, abs=0.01)
    assert xs[-1].var() == pytest.approx(var, rel=0.03)


def test_operator_conserves_mass_and_rejects_bad_grids():
    L = fokker_planck_operator(np.linspace(0, 1, 20), np.sin(np.linspace(0, 1, 20)), 0.1 + np.linspace(0, 1, 20))
    assert np.asarray(L.sum(axis=0)).ravel() == pytest.approx(np.zeros(20), abs=1e-10)
    with pytest.raises(ValueError):
        fokker_planck_operator(np.array([0.0, 1.0, 3.0]), 0.0, 1.0)
    with pytest.raises(ValueError):
        fokker_planck_1d(np.ones(5), np.linspace(0, 1, 5), 0.0, 1.0, 1.0, 0.1, theta=0.2)
