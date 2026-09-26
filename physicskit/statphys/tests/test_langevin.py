"""Langevin dynamics against closed forms: MSD = 2Dt, the Einstein relation, OU statistics."""

import numpy as np
import pytest

from physicskit.statphys import (
    BrownianMotion,
    LangevinDynamics,
    OrnsteinUhlenbeck,
    einstein_diffusion_coefficient,
    stokes_einstein_diffusion_coefficient,
)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_free_brownian_msd_is_2dDt(dim):
    """<|x(t) - x(0)|^2> = 2 d D t with D = kT/gamma (Einstein 1905)."""
    bm = BrownianMotion(n_particles=6000, dim=dim, gamma=2.0, kT=1.5, seed=dim)
    t, _ = bm.run(t_max=4.0, dt=0.01, n_frames=40)
    _, msd = bm.mean_squared_displacement()
    slope = np.sum(msd * t) / np.sum(t * t)
    assert slope == pytest.approx(2 * dim * 1.5 / 2.0, rel=0.03)


def test_einstein_relation_diffusion_equals_mobility_times_kT():
    """Measure D from the spread and mu from the drift under a constant force:
    D = mu kT independently of the force (Einstein 1905)."""
    gamma, kT = 3.0, 0.8
    bm = BrownianMotion(n_particles=8000, dim=2, gamma=gamma, kT=kT, force=[1.2, -0.5], seed=11)
    bm.run(t_max=5.0, dt=0.01, n_frames=25)
    D, mu = bm.measured_diffusion_coefficient(), bm.measured_mobility()
    assert mu == pytest.approx(1 / gamma, rel=0.03)
    assert D == pytest.approx(kT / gamma, rel=0.03)
    assert D / (mu * kT) == pytest.approx(1.0, rel=0.05)


def test_trapped_brownian_particle_obeys_equipartition():
    """In a harmonic trap <x^2> -> kT/k; Euler-Maruyama shifts it by the factor 1/(1 - k dt / 2 gamma)."""
    k, gamma, kT, dt = 4.0, 1.0, 0.5, 0.005
    bm = BrownianMotion(n_particles=20000, gamma=gamma, kT=kT, stiffness=k, seed=5)
    _, x = bm.run(t_max=3.0, dt=dt, n_frames=6)
    assert np.var(x[-1]) == pytest.approx(kT / k / (1 - k * dt / (2 * gamma)), rel=0.03)


def test_underdamped_langevin_thermalizes_and_diffuses():
    """Velocities reach <v^2> = kT/m; at long times MSD grows as 2Dt with D = kT/gamma,
    after the ballistic-to-diffusive crossover
    <x^2> = 2D[t - tau(1 - e^{-t/tau})] for particles started with thermal velocities."""
    m, gamma, kT, dt = 1.0, 2.0, 0.5, 0.002
    n = 8000
    rng = np.random.default_rng(0)
    v0 = rng.normal(scale=np.sqrt(kT / m), size=(n, 1))
    ld = LangevinDynamics(n_particles=n, mass=m, gamma=gamma, kT=kT, seed=3)
    t, x, v = ld.run(t_max=10.0, dt=dt, n_frames=50, v0=v0)
    assert ld.kinetic_temperature() == pytest.approx(kT, rel=0.04)
    tau = m / gamma
    D = kT / gamma
    _, msd = ld.mean_squared_displacement()
    exact = 2 * D * (t - tau * (1 - np.exp(-t / tau)))
    assert msd[5:] == pytest.approx(exact[5:], rel=0.05)
    assert ld.diffusion_coefficient == D
    assert ld.velocity_relaxation_time == tau


def test_weakly_damped_trap_stays_at_bath_temperature_with_baoab():
    """Equipartition m<v^2> = k<x^2> = kT for gamma = 0.1; Euler-Maruyama instead heats
    the oscillator by about gamma / (gamma - m omega^2 dt)."""
    m, k, gamma, kT, dt = 1.0, 4.0, 0.1, 0.5, 0.01
    runs = {}
    for method in ("baoab", "euler_maruyama"):
        ld = LangevinDynamics(n_particles=4000, mass=m, gamma=gamma, kT=kT, stiffness=k, seed=8)
        _, x, v = ld.run(t_max=100.0, dt=dt, n_frames=20, method=method)
        runs[method] = (k * np.mean(x[-4:] ** 2), m * np.mean(v[-4:] ** 2))
    assert runs["baoab"] == pytest.approx((kT, kT), rel=0.04)
    heating = gamma / (gamma - k / m * dt)
    assert runs["euler_maruyama"][1] == pytest.approx(kT * heating, rel=0.06)
    with pytest.raises(ValueError):
        LangevinDynamics(n_particles=2).run(1.0, 0.1, n_frames=1, method="verlet")


@pytest.mark.parametrize("method", ["exact", "euler_maruyama"])
def test_ou_stationary_variance(method):
    theta, sigma, dt = 1.5, 0.9, 0.005
    ou = OrnsteinUhlenbeck(theta=theta, mu=2.0, sigma=sigma, n_paths=30000, seed=1)
    _, X = ou.run(t_max=6.0, dt=dt, n_frames=12, x0=-1.0, method=method)
    expected = sigma**2 / (2 * theta)
    if method == "euler_maruyama":
        expected = sigma**2 / (theta * (2 - theta * dt))
    assert X[-1].var() == pytest.approx(expected, rel=0.03)
    assert X[-1].mean() == pytest.approx(2.0, abs=0.02)


def test_ou_transient_mean_variance_and_autocovariance():
    ou = OrnsteinUhlenbeck(theta=0.8, mu=1.0, sigma=0.6, n_paths=40000, seed=2)
    t, X = ou.run(t_max=4.0, dt=0.01, n_frames=40, x0=5.0)
    sel = [5, 10, 20, 40]
    assert X[sel].mean(axis=1) == pytest.approx(ou.mean(t[sel], 5.0), abs=0.01)
    assert X[sel].var(axis=1) == pytest.approx(ou.variance(t[sel]), rel=0.04)
    # stationary start: autocovariance e^{-theta tau}
    t, X = ou.run(t_max=3.0, dt=0.01, n_frames=30)
    lag = 10
    cov = np.mean((X[0] - 1.0) * (X[lag] - 1.0))
    assert cov == pytest.approx(ou.autocovariance(t[lag]), rel=0.05)


def test_einstein_helpers_and_errors():
    assert einstein_diffusion_coefficient(1.0, 4.0) == 0.25
    assert stokes_einstein_diffusion_coefficient(1.0, 1.0, 1.0) == pytest.approx(1 / (6 * np.pi))
    with pytest.raises(ValueError):
        BrownianMotion(gamma=0.0)
    with pytest.raises(RuntimeError):
        BrownianMotion().mean_squared_displacement()
    with pytest.raises(ValueError):
        OrnsteinUhlenbeck(theta=-1.0)
    with pytest.raises(ValueError):
        OrnsteinUhlenbeck().run(1.0, 0.1, n_frames=2, method="heun")
    bm = BrownianMotion(n_particles=10, seed=0)
    bm.run(1.0, 0.1, n_frames=2)
    assert np.isnan(bm.measured_mobility())
