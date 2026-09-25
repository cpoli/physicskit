"""Physics-correctness tests for physicskit.semiclassical.core.propagators."""

from __future__ import annotations

import numpy as np
import pytest
from numba import njit

from physicskit.semiclassical.core.propagators import (
    coherent_state_overlap,
    frozen_gaussian_1d,
    herman_kluk_propagate_wavepacket,
    propagate_trajectory_monodromy_action,
    van_vleck_propagator_1d,
)

_ZERO = njit(lambda q, params: 0.0, cache=False)


def _harmonic_potential(m=1.0, omega=1.0):
    params = np.array([m * omega**2])
    dVdx = njit(lambda q, params: params[0] * q, cache=False)
    d2Vdx2 = njit(lambda q, params: params[0], cache=False)
    V = njit(lambda q, params: 0.5 * params[0] * q**2, cache=False)
    return dVdx, d2Vdx2, V, params


def test_monodromy_matches_harmonic_oscillator_and_preserves_liouville():
    dVdx, d2Vdx2, V, params = _harmonic_potential()
    t = 2.1
    _q, _p, M, _S, _hist = propagate_trajectory_monodromy_action(1.0, 0.4, dVdx, d2Vdx2, V, m=1.0, dt=t / 3000, steps=3000, params=params)
    M_exact = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    assert np.max(np.abs(M - M_exact)) < 1e-5
    assert abs(np.linalg.det(M) - 1.0) < 1e-8


def test_van_vleck_propagator_matches_free_particle_exact():
    q_t, K = van_vleck_propagator_1d(q0=0.0, p0=1.0, dVdx=_ZERO, d2Vdx2=_ZERO, V=_ZERO, m=1.0, dt=1.0 / 2000, steps=2000)
    K_exact = np.sqrt(1.0 / (2j * np.pi)) * np.exp(1j * q_t**2 / 2.0)
    assert abs(K - K_exact) < 1e-6


def test_van_vleck_propagator_matches_harmonic_oscillator_exact():
    dVdx, d2Vdx2, V, params = _harmonic_potential()
    q0, p0, t = 1.0, 0.3, 1.3
    q_t, K = van_vleck_propagator_1d(q0, p0, dVdx, d2Vdx2, V, m=1.0, dt=t / 4000, steps=4000, hbar=1.0, params=params)
    K_exact = np.sqrt(1.0 / (2j * np.pi * np.sin(t))) * np.exp(1j / (2 * np.sin(t)) * ((q_t**2 + q0**2) * np.cos(t) - 2 * q_t * q0))
    assert abs(K - K_exact) < 1e-5


def test_coherent_state_overlap_matches_numerical_integration():
    x = np.linspace(-40, 40, 20000)
    gamma = 1.0
    psi1 = frozen_gaussian_1d(x, qc=0.3, pc=0.7, gamma=gamma)
    psi2 = frozen_gaussian_1d(x, qc=-0.2, pc=1.1, gamma=gamma)
    numeric = np.trapezoid(np.conj(psi1) * psi2, x)
    closed_form = coherent_state_overlap(0.3, 0.7, -0.2, 1.1, gamma)
    assert abs(numeric - closed_form) < 1e-6


def test_herman_kluk_reconstructs_initial_state_at_short_time():
    dVdx, d2Vdx2, V, params = _harmonic_potential()
    x = np.linspace(-6, 6, 400)
    psi = herman_kluk_propagate_wavepacket(
        1.0, 0.0, gamma=1.0, dVdx=dVdx, d2Vdx2=d2Vdx2, V=V, m=1.0, dt=1e-6, steps=1, x_eval=x, n_grid=61, n_sigma=7.0, params=params
    )
    psi0 = frozen_gaussian_1d(x, qc=1.0, pc=0.0, gamma=1.0)
    fidelity = abs(np.trapezoid(np.conj(psi0) * psi, x)) ** 2
    assert fidelity > 0.999


def test_herman_kluk_defaults_params_to_an_empty_array_when_unused():
    # dVdx/d2Vdx2/V here never index into params, so params=None (->
    # np.empty(0) internally) is fine and should behave like the
    # explicit-params harmonic oscillator above (m=omega=1).
    dVdx = njit(lambda q, params: q, cache=False)
    d2Vdx2 = njit(lambda q, params: 1.0, cache=False)
    V = njit(lambda q, params: 0.5 * q**2, cache=False)
    x = np.linspace(-6, 6, 400)
    psi = herman_kluk_propagate_wavepacket(1.0, 0.0, gamma=1.0, dVdx=dVdx, d2Vdx2=d2Vdx2, V=V, m=1.0, dt=1e-6, steps=1, x_eval=x, n_grid=61, n_sigma=7.0)
    psi0 = frozen_gaussian_1d(x, qc=1.0, pc=0.0, gamma=1.0)
    fidelity = abs(np.trapezoid(np.conj(psi0) * psi, x)) ** 2
    assert fidelity > 0.999


@pytest.mark.parametrize("gamma", [0.25, 0.5, 1.0])
def test_herman_kluk_is_exact_for_the_harmonic_oscillator(gamma):
    # HK is exact for quadratic potentials, so the propagated packet must stay normalized and
    # follow the classical centre at a finite time (where M is far from the identity).
    # Previously the prefactor used gamma instead of 2*gamma for e^{-gamma x^2} Gaussians,
    # giving norms of 1.16 / 0.86 at t = 0.9.
    import numpy as np
    from numba import njit

    from physicskit.semiclassical.core.propagators import herman_kluk_prefactor, herman_kluk_propagate_wavepacket

    t = 0.9
    M = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    assert abs(herman_kluk_prefactor(M, gamma=0.5)) == pytest.approx(1.0, abs=1e-12)  # width-matched coherent state

    dVdx = njit(lambda q, params: q)
    d2Vdx2 = njit(lambda q, params: 1.0)
    V = njit(lambda q, params: 0.5 * q * q)
    x = np.linspace(-7.0, 7.0, 281)
    q0, p0, steps = 1.0, 0.5, 90
    psi = herman_kluk_propagate_wavepacket(q0, p0, gamma, dVdx, d2Vdx2, V, 1.0, t / steps, steps, x, n_grid=41)
    density = np.abs(psi) ** 2
    assert np.trapezoid(density, x) == pytest.approx(1.0, abs=2e-3)
    assert np.trapezoid(x * density, x) == pytest.approx(q0 * np.cos(t) + p0 * np.sin(t), abs=2e-3)
