"""Tests for physicskit.optics.lasers: the lasing threshold and inversion
clamping, rate-equation steady states and relaxation oscillations, and the
Maxwell-Bloch thresholds and Lorenz isomorphism."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import find_peaks

from physicskit.integrators import rk4_integrate
from physicskit.optics.lasers import LaserRateEquations, MaxwellBloch, _maxwell_bloch_rhs


def test_threshold_values():
    laser = LaserRateEquations(pump=1.0, tau=2.0, tau_c=0.05, B=4.0)
    assert laser.threshold_inversion == pytest.approx(1 / (4.0 * 0.05))
    assert laser.threshold_pump == pytest.approx(1 / (4.0 * 0.05 * 2.0))


@pytest.mark.parametrize("r", [0.3, 0.9, 1.5, 4.0])
def test_steady_state_below_and_above_threshold(r):
    base = LaserRateEquations(tau=1.0, tau_c=0.01, B=1.0)
    laser = LaserRateEquations(pump=r * base.threshold_pump, tau=1.0, tau_c=0.01, B=1.0)
    N, q = laser.steady_state()
    if r < 1:
        assert N == pytest.approx(laser.pump * laser.tau)
        assert q == pytest.approx(0.0, abs=1e-9)
    else:
        assert N == pytest.approx(laser.threshold_inversion)  # gain clamping
        assert q == pytest.approx(laser.tau_c * (laser.pump - laser.threshold_pump))


def test_integration_converges_to_steady_state():
    laser = LaserRateEquations(pump=250.0, tau=1.0, tau_c=0.01, B=1.0)
    _, N, q = laser.integrate(t_max=30.0, dt=5e-4, q0=1e-3)
    N_s, q_s = laser.steady_state()
    assert N[-1] == pytest.approx(N_s, rel=1e-6)
    assert q[-1] == pytest.approx(q_s, rel=1e-6)


def test_output_is_linear_in_pump_above_threshold():
    pumps = np.array([150.0, 200.0, 300.0, 400.0])
    q_out = [LaserRateEquations(pump=p, tau=1.0, tau_c=0.01).steady_state()[1] for p in pumps]
    slope = np.diff(q_out) / np.diff(pumps)
    np.testing.assert_allclose(slope, 0.01)  # slope efficiency tau_c


def test_spontaneous_emission_smooths_threshold():
    kw = dict(tau=1.0, tau_c=0.01, B=1.0, beta=1e-3)
    below = LaserRateEquations(pump=50.0, **kw).steady_state()
    above = LaserRateEquations(pump=300.0, **kw).steady_state()
    assert 0 < below[1] < 0.01 * above[1]
    # beta > 0 steady state really is stationary
    laser = LaserRateEquations(pump=120.0, **kw)
    rhs = laser._deriv_njit(np.array(laser.steady_state()), 0.0, laser.params)
    np.testing.assert_allclose(rhs, 0.0, atol=1e-9)


def test_relaxation_oscillation_frequency():
    laser = LaserRateEquations(pump=3000.0, tau=1.0, tau_c=1e-3, B=1.0)  # r = 3
    Omega, Gamma = laser.relaxation_oscillation()
    # tiny inversion kick (delta q / q_s ~ B N_s delta N / (Omega q_s) ~ 1%): the linear regime
    N_s, q_s = laser.steady_state()
    t, _, q = laser.integrate(t_max=2.0, dt=1e-5, N0=(1 + 1e-6) * N_s, q0=q_s)
    peaks, _ = find_peaks(q)
    assert np.mean(np.diff(t[peaks])) == pytest.approx(2 * np.pi / Omega, rel=1e-3)
    # peak heights above the steady state decay at rate Gamma
    rate = -np.polyfit(t[peaks], np.log(q[peaks] - q_s), 1)[0]
    assert rate == pytest.approx(Gamma, rel=0.02)


def test_relaxation_oscillation_requires_lasing():
    with pytest.raises(ValueError):
        LaserRateEquations(pump=50.0, tau=1.0, tau_c=0.01).relaxation_oscillation()


@pytest.mark.parametrize("r", [0.6, 1.8, 5.0])
def test_maxwell_bloch_threshold(r):
    mb = MaxwellBloch(kappa=1.0, gamma_perp=1.0, gamma_par=0.5, r=r)  # good cavity: no second threshold
    assert mb.second_threshold == np.inf
    _, y = mb.integrate([0.01, 0.0, r], t_max=200.0, dt=0.01)
    np.testing.assert_allclose(y[-1], mb.steady_state(), atol=1e-6)
    if r > 1:
        assert y[-1, 0] ** 2 == pytest.approx(r - 1, rel=1e-6)


def test_second_threshold_is_where_lasing_state_goes_unstable():
    mb = MaxwellBloch(kappa=10.0, gamma_perp=1.0, gamma_par=8 / 3)
    rH = mb.second_threshold
    assert rH == pytest.approx(10 * (10 + 8 / 3 + 3) / (10 - 8 / 3 - 1))
    for r, stable in ((0.98 * rH, True), (1.02 * rH, False)):
        mb.r = r
        max_re = np.linalg.eigvals(mb.jacobian(mb.steady_state())).real.max()
        assert (max_re < 0) == stable


def test_chaotic_output_above_second_threshold():
    mb = MaxwellBloch(kappa=10.0, gamma_perp=1.0, gamma_par=8 / 3, r=28.0)
    _, y = mb.integrate([1.0, 1.0, 1.0], t_max=60.0, dt=0.002)
    late = y[len(y) // 2 :, 0]
    assert late.std() > 1.0  # never settles on +-sqrt(r-1)
    assert (late > 0).any() and (late < 0).any()  # switches between the two lobes


def test_maxwell_bloch_is_lorenz():
    kappa, gp, gl, r = 10.0, 2.0, 16 / 3, 28.0  # sigma = 5, b = 8/3 in units gamma_perp t
    mb = MaxwellBloch(kappa, gp, gl, r)
    s0 = np.array([0.3, -0.2, 20.0])
    dt, n = 1e-4, 5000
    _, y = mb.integrate(s0, t_max=n * dt, dt=dt)
    sigma, b = kappa / gp, gl / gp
    xyz = np.array([np.sqrt(b) * y[:, 0], np.sqrt(b) * y[:, 1], r - y[:, 2]]).T

    # Lorenz equations integrated independently in rescaled time tau = gamma_perp t
    lorenz = np.array([sigma, 1.0, b, r])
    # (E, P, D)-form rhs with kappa->sigma, gamma_perp->1, gamma_par->b on the mapped state
    _, y_ref = rk4_integrate(_maxwell_bloch_rhs, s0.copy(), 0.0, dt * gp, n, lorenz)
    np.testing.assert_allclose(y, y_ref, atol=1e-8)  # time-rescaling invariance
    x, yy, z = xyz.T
    dxyz = np.gradient(xyz, dt * gp, axis=0)[10:-10]
    rhs = np.array([sigma * (yy - x), x * (r - z) - yy, x * yy - b * z]).T[10:-10]
    np.testing.assert_allclose(dxyz, rhs, rtol=1e-3, atol=1e-3)
