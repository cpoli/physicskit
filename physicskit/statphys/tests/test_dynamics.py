import numpy as np
import pytest

from physicskit.statphys.utils.dynamics import autocorrelation_function, integrated_autocorrelation_time


def test_autocorrelation_function_zero_lag_is_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    rho = autocorrelation_function(x)
    assert rho[0] == pytest.approx(1.0)


def test_autocorrelation_decays_for_white_noise():
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    rho = autocorrelation_function(x, max_lag=50)
    assert abs(rho[20]) < 0.2


def test_autocorrelation_function_high_for_slow_series():
    rng = np.random.default_rng(2)
    # a slowly varying (highly autocorrelated) random walk
    x = np.cumsum(rng.normal(size=2000))
    rho = autocorrelation_function(x, max_lag=10)
    assert rho[5] > 0.5


def test_integrated_autocorrelation_time_small_for_independent_samples():
    rng = np.random.default_rng(3)
    x = rng.normal(size=5000)
    tau = integrated_autocorrelation_time(x)
    assert 0.0 < tau < 3.0


def test_integrated_autocorrelation_time_larger_for_correlated_series():
    rng = np.random.default_rng(4)
    n = 5000
    # AR(1) process with strong positive autocorrelation
    x = np.empty(n)
    x[0] = rng.normal()
    phi = 0.9
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()

    tau_correlated = integrated_autocorrelation_time(x)
    tau_independent = integrated_autocorrelation_time(rng.normal(size=n))
    assert tau_correlated > tau_independent


def test_integrated_autocorrelation_time_falls_back_when_window_never_closes():
    # A strongly autocorrelated AR(1) series (tau grows large), restricted
    # to a small max_lag: the automatic-windowing condition c*tau <= M is
    # never satisfied within that short lag range, so the function should
    # fall back to the full-sum estimate instead of raising or looping
    # forever.
    rng = np.random.default_rng(4)
    n = 5000
    x = np.empty(n)
    x[0] = rng.normal()
    phi = 0.95
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()

    tau = integrated_autocorrelation_time(x, max_lag=3)
    assert tau > 0.0
