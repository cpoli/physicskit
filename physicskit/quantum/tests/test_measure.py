"""Tests for physicskit.quantum.utils.measure, using a normalized Gaussian
wavepacket ``psi(x) = (2*pi*sigma^2)^(-1/4) * exp(-(x-x0)^2/(4*sigma^2))
* exp(i*p0*x/hbar)`` as a fixture with exactly known moments:
``<x> = x0``, ``Var(x) = sigma^2``, ``<p> = p0``,
``Var(p) = hbar^2/(4*sigma^2)`` -- the standard minimum-uncertainty
state, so ``dx*dp`` should equal ``hbar/2`` exactly (to numerical
precision), not merely satisfy the Heisenberg inequality.

No test in this package previously covered this module at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.quantum.utils.measure import (
    ExpectationMonitor,
    expectation_value,
    momentum_density,
    momentum_expectation,
    momentum_variance,
    position_expectation,
    position_variance,
    simulate_position_measurement,
    uncertainty,
)

HBAR = 1.0
X0, P0, SIGMA = 2.0, 3.0, 1.0
X = np.linspace(-20.0, 20.0, 4000)


def _gaussian_wavepacket(x0=X0, p0=P0, sigma=SIGMA):
    return (2.0 * np.pi * sigma**2) ** -0.25 * np.exp(-((X - x0) ** 2) / (4.0 * sigma**2)) * np.exp(1j * p0 * X / HBAR)


def test_expectation_value_of_position_operator_matches_position_expectation():
    psi = _gaussian_wavepacket()
    ev = expectation_value(lambda x, p: x * p, X, psi)
    assert ev.real == pytest.approx(X0, abs=1e-9)
    assert ev.imag == pytest.approx(0.0, abs=1e-9)
    assert ev.real == pytest.approx(position_expectation(X, psi))


def test_position_expectation_and_variance_match_gaussian_moments():
    psi = _gaussian_wavepacket()
    assert position_expectation(X, psi) == pytest.approx(X0, abs=1e-9)
    assert position_variance(X, psi) == pytest.approx(SIGMA**2, abs=1e-9)


def test_momentum_density_is_normalized():
    psi = _gaussian_wavepacket()
    p, psi_p = momentum_density(X, psi, HBAR)
    assert np.all(np.diff(p) > 0)  # returned ascending, per docstring
    assert np.trapezoid(np.abs(psi_p) ** 2, p) == pytest.approx(1.0, rel=1e-6)


def test_momentum_expectation_and_variance_match_gaussian_moments():
    psi = _gaussian_wavepacket()
    assert momentum_expectation(X, psi, HBAR) == pytest.approx(P0, rel=1e-6)
    assert momentum_variance(X, psi, HBAR) == pytest.approx(HBAR**2 / (4.0 * SIGMA**2), rel=1e-4)


def test_uncertainty_saturates_heisenberg_bound_for_minimum_uncertainty_gaussian():
    psi = _gaussian_wavepacket()
    dx, dp, product = uncertainty(X, psi, HBAR)
    assert dx == pytest.approx(SIGMA, abs=1e-6)
    assert dp == pytest.approx(HBAR / (2.0 * SIGMA), rel=1e-4)
    assert product == pytest.approx(HBAR / 2.0, rel=1e-4)
    assert product >= HBAR / 2.0 - 1e-9


def test_expectation_monitor_record_all_matches_direct_calls_per_frame():
    times = np.array([0.0, 1.0, 2.0])
    frames = np.array([_gaussian_wavepacket(x0=X0 + dt) for dt in times])

    monitor = ExpectationMonitor(x=X, hbar=HBAR)
    monitor.record_all(times, frames)
    arrays = monitor.as_arrays()

    np.testing.assert_allclose(arrays["t"], times)
    np.testing.assert_allclose(arrays["x"], X0 + times, atol=1e-9)
    np.testing.assert_allclose(arrays["norm"], 1.0, atol=1e-6)
    np.testing.assert_allclose(arrays["dx"], SIGMA, atol=1e-6)


def test_expectation_monitor_record_single_snapshot_matches_uncertainty_directly():
    monitor = ExpectationMonitor(x=X, hbar=HBAR)
    psi = _gaussian_wavepacket()
    monitor.record(0.5, psi)
    dx, dp, _ = uncertainty(X, psi, HBAR)
    assert monitor.history["t"] == [0.5]
    assert monitor.history["dx"][0] == pytest.approx(dx)
    assert monitor.history["dp"][0] == pytest.approx(dp)


def test_simulate_position_measurement_samples_match_born_rule_moments():
    psi = _gaussian_wavepacket()
    samples = simulate_position_measurement(X, psi, n_samples=200_000, rng=np.random.default_rng(0))
    assert samples.shape == (200_000,)
    assert samples.mean() == pytest.approx(X0, abs=0.05)
    assert samples.var() == pytest.approx(SIGMA**2, rel=0.05)


def test_simulate_position_measurement_defaults_to_a_single_sample():
    psi = _gaussian_wavepacket()
    sample = simulate_position_measurement(X, psi)
    assert sample.shape == (1,)
    assert X.min() <= sample[0] <= X.max()
