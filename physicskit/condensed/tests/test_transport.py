"""Tests for physicskit.condensed.transport: Drude conductivity, Hall
effect and sum rule, and the Boltzmann relaxation-time solver against
Drude, Wiedemann-Franz, and Mott."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from physicskit.condensed.transport import (
    boltzmann_transport,
    drude_ac_conductivity,
    drude_conductivity,
    drude_conductivity_tensor,
    fermi_window,
    hall_coefficient,
)


def test_drude_dc_and_ac_limits():
    n, tau, q, m = 3.0, 0.4, -1.0, 2.0
    sigma0 = drude_conductivity(n, tau, q, m)
    assert sigma0 == pytest.approx(n * q**2 * tau / m)
    assert drude_ac_conductivity(0.0, n, tau, q, m) == pytest.approx(sigma0)
    s = drude_ac_conductivity(1 / tau, n, tau, q, m)
    assert s.real == pytest.approx(sigma0 / 2) and s.imag == pytest.approx(sigma0 / 2)


def test_drude_f_sum_rule():
    n, tau, q, m = 1.7, 0.3, -1.0, 1.4
    integral = quad(lambda w: drude_ac_conductivity(w, n, tau, q, m).real, 0, np.inf)[0]
    assert integral == pytest.approx(np.pi * n * q**2 / (2 * m), rel=1e-8)


@pytest.mark.parametrize("q", [-1.0, 1.0])
def test_drude_hall_resistivity_and_no_magnetoresistance(q):
    n, tau, m = 0.8, 2.0, 1.5
    for B in (0.0, 0.5, 3.0):
        rho = np.linalg.inv(drude_conductivity_tensor(B, n, tau, q, m))
        assert rho[0, 0] == pytest.approx(1 / drude_conductivity(n, tau, q, m))
        assert rho[1, 0] == pytest.approx(B * hall_coefficient(n, q))
        assert rho[0, 1] == pytest.approx(-B * hall_coefficient(n, q))


def test_fermi_window_normalized():
    assert quad(lambda e: fermi_window(e, 0.3, 0.05), -5, 5)[0] == pytest.approx(1.0)


@pytest.fixture(scope="module")
def parabolic_2d():
    # 2D parabolic band eps = k^2 / 2m, fine grid resolving the thermal window
    m, mu, T = 0.8, 1.0, 0.02
    k = np.linspace(-2.2, 2.2, 1401)
    KX, KY = np.meshgrid(k, k)
    eps = (KX**2 + KY**2) / (2 * m)
    v = np.stack([KX.ravel(), KY.ravel()], axis=1) / m
    dk = k[1] - k[0]
    return m, mu, T, boltzmann_transport(eps.ravel(), v, dk**2, mu=mu, T=T, tau=0.5, q=-1.0)


def test_boltzmann_reproduces_drude(parabolic_2d):
    m, mu, T, res = parabolic_2d
    # 2D, g_s = 2: n = m mu / pi
    assert res.density == pytest.approx(m * mu / np.pi, rel=2e-3)
    assert res.sigma[0, 0] == pytest.approx(drude_conductivity(res.density, 0.5, -1.0, m), rel=2e-3)
    assert res.sigma[0, 0] == pytest.approx(res.sigma[1, 1], rel=1e-10)
    assert abs(res.sigma[0, 1]) < 1e-10 * res.sigma[0, 0]


def test_wiedemann_franz_law(parabolic_2d):
    _, _, _, res = parabolic_2d
    assert res.lorenz_number == pytest.approx(np.pi**2 / 3, rel=2e-3)


def test_mott_seebeck_formula(parabolic_2d):
    # 2D parabolic band: sigma(eps) ∝ eps, so S = (pi^2 T / 3q) / mu
    _, mu, T, res = parabolic_2d
    assert res.seebeck[0, 0] == pytest.approx(np.pi**2 * T / (3 * -1.0 * mu), rel=0.01)


def test_tight_binding_chain_conductivity():
    # 1D band eps = -2t cos k at T -> 0: sigma = q^2 tau g_s |v_F| / pi, v_F = 2t sin k_F
    t, tau = 1.0, 2.0
    k = np.linspace(-np.pi, np.pi, 200001)[:-1]
    eps = -2 * t * np.cos(k)
    v = (2 * t * np.sin(k))[:, None]
    mu = -1.0
    res = boltzmann_transport(eps, v, k[1] - k[0], mu=mu, T=0.005, tau=tau)
    kF = np.arccos(-mu / (2 * t))
    assert res.sigma[0, 0] == pytest.approx(tau * 2 * 2 * t * np.sin(kF) / np.pi, rel=1e-3)
    assert res.density == pytest.approx(2 * kF / np.pi, rel=1e-3)
