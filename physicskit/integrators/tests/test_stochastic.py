"""Euler-Maruyama, Milstein and BAOAB against closed forms; the SDE schemes against the exact solution of geometric Brownian motion.

dX = mu X dt + sigma X dW has the pathwise solution
X_t = X_0 exp((mu - sigma^2/2) t + sigma W_t) (Kloeden and Platen 1992, eq. 4.4.6),
so strong errors can be measured on the same Brownian path at several step sizes.
"""

import numpy as np
import pytest
from numba import njit

from physicskit.integrators import (
    baoab_integrate,
    baoab_step,
    euler_maruyama_integrate,
    euler_maruyama_step,
    milstein_integrate,
    milstein_step,
    velocity_verlet_step,
    wiener_increments,
)

MU, SIGMA = 0.5, 0.8


@njit
def _gbm_drift(x, t, p):
    return p[0] * x


@njit
def _gbm_diffusion(x, t, p):
    return p[1] * x


@njit
def _gbm_diffusion_derivative(x, t, p):
    return p[1] * np.ones_like(x)


@njit
def _ou_drift(x, t, p):
    return -p[0] * x


@njit
def _const_diffusion(x, t, p):
    return p[1] * np.ones_like(x)


@njit
def _zero(x, t, p):
    return np.zeros_like(x)


def _strong_errors(integrate, *callbacks, n_paths=400, T=1.0):
    """Mean |X_T - exact| over independent paths, at dt = T/2^k for k = 4..8."""
    params = np.array([MU, SIGMA])
    n_fine = 2**10
    dW_fine = wiener_increments(n_fine, n_paths, T / n_fine, rng=42)
    x0 = np.ones(n_paths)
    exact = np.exp((MU - 0.5 * SIGMA**2) * T + SIGMA * dW_fine.sum(axis=0))
    dts, errs = [], []
    for k in range(4, 9):
        n = 2**k
        dW = dW_fine.reshape(n, n_fine // n, n_paths).sum(axis=1)
        _, X = integrate(*callbacks, x0, 0.0, T / n, dW, params)
        dts.append(T / n)
        errs.append(np.mean(np.abs(X[-1] - exact)))
    return np.array(dts), np.array(errs)


def test_euler_maruyama_strong_order_one_half():
    dts, errs = _strong_errors(euler_maruyama_integrate, _gbm_drift, _gbm_diffusion)
    order = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    assert order == pytest.approx(0.5, abs=0.1)


def test_milstein_strong_order_one():
    dts, errs = _strong_errors(milstein_integrate, _gbm_drift, _gbm_diffusion, _gbm_diffusion_derivative)
    order = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    assert order == pytest.approx(1.0, abs=0.1)
    _, em_errs = _strong_errors(euler_maruyama_integrate, _gbm_drift, _gbm_diffusion)
    assert np.all(errs < em_errs)


def test_euler_maruyama_weak_mean_of_gbm():
    """E[X_T] = X_0 exp(mu T), with weak error O(dt)."""
    n_paths, n, T = 40_000, 200, 1.0
    dW = wiener_increments(n, n_paths, T / n, rng=7)
    _, X = euler_maruyama_integrate(_gbm_drift, _gbm_diffusion, np.ones(n_paths), 0.0, T / n, dW, np.array([MU, SIGMA]), n)
    assert X.shape == (2, n_paths)
    stderr = X[-1].std() / np.sqrt(n_paths)
    assert X[-1].mean() == pytest.approx(np.exp(MU * T), abs=4 * stderr + 0.01)


def test_ornstein_uhlenbeck_stationary_variance():
    """dX = -theta X dt + s dW has stationary variance s^2/(2 theta); Euler-Maruyama
    shifts it to s^2/(theta (2 - theta dt)) exactly."""
    theta, s, dt, n_paths = 2.0, 0.7, 0.01, 20_000
    dW = wiener_increments(2000, n_paths, dt, rng=3)
    _, X = euler_maruyama_integrate(_ou_drift, _const_diffusion, np.zeros(n_paths), 0.0, dt, dW, np.array([theta, s]), 2000)
    assert X[-1].var() == pytest.approx(s**2 / (theta * (2 - theta * dt)), rel=0.03)


def test_milstein_reduces_to_euler_maruyama_for_additive_noise():
    dW = wiener_increments(300, 5, 0.01, rng=0)
    params = np.array([1.5, 0.3])
    x0 = np.linspace(-1, 1, 5)
    _, em = euler_maruyama_integrate(_ou_drift, _const_diffusion, x0, 0.0, 0.01, dW, params)
    _, mil = milstein_integrate(_ou_drift, _const_diffusion, _zero, x0, 0.0, 0.01, dW, params)
    assert np.array_equal(em, mil)


def test_single_steps_and_save_every():
    params = np.array([MU, SIGMA])
    x = np.array([2.0])
    dW = np.array([0.1])
    em = euler_maruyama_step(_gbm_drift, _gbm_diffusion, x, 0.0, 0.01, dW, params)
    assert em[0] == pytest.approx(2.0 + MU * 2.0 * 0.01 + SIGMA * 2.0 * 0.1)
    mil = milstein_step(_gbm_drift, _gbm_diffusion, _gbm_diffusion_derivative, x, 0.0, 0.01, dW, params)
    assert mil[0] - em[0] == pytest.approx(0.5 * SIGMA * 2.0 * SIGMA * (0.1**2 - 0.01))
    t, X = euler_maruyama_integrate(_gbm_drift, _gbm_diffusion, x, 0.0, 0.01, wiener_increments(10, 1, 0.01, rng=1), params, 5)
    assert t == pytest.approx([0.0, 0.05, 0.1])
    assert X.shape == (3, 1)


def test_wiener_increments_are_reproducible():
    assert np.array_equal(wiener_increments(5, 3, 0.1, rng=9), wiener_increments(5, 3, 0.1, rng=9))
    assert wiener_increments(5, 3, 0.1).shape == (5, 3)


@njit
def _spring_force(x, t, p):
    return -p[0] * x


def test_baoab_without_friction_is_velocity_verlet():
    params = np.array([3.0])
    x, v = np.array([0.4, -1.0]), np.array([0.2, 0.5])
    xb, vb = baoab_step(_spring_force, x, v, 0.0, 0.05, 1.0, 0.0, 1.0, np.array([0.7, -0.3]), params)
    # velocity_verlet_step takes an acceleration; mass is 1 here
    xv, vv = velocity_verlet_step(_spring_force, x, v, 0.0, 0.05, params)
    assert xb == pytest.approx(xv, rel=1e-14)
    assert vb == pytest.approx(vv, rel=1e-14)


def test_baoab_harmonic_equipartition_at_large_step():
    """BAOAB samples k<x^2> = kT exactly in a harmonic well even at omega dt = 0.5
    (Leimkuhler and Matthews 2013); m<v^2> carries an O(dt^2) error."""
    k, m, gamma, kT, dt, n = 4.0, 1.0, 1.0, 0.7, 0.25, 20000
    eta = np.random.default_rng(5).standard_normal((400, n))
    _, x, v = baoab_integrate(_spring_force, np.zeros(n), np.zeros(n), 0.0, dt, m, gamma, kT, eta, np.array([k]), 400)
    assert k * np.mean(x[-1] ** 2) == pytest.approx(kT, rel=0.03)
    assert m * np.mean(v[-1] ** 2) == pytest.approx(kT * (1 - k * dt**2 / (4 * m)), rel=0.03)
