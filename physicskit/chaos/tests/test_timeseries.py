import numpy as np
import pytest

from physicskit.chaos.systems.continuous import Lorenz
from physicskit.chaos.utils.timeseries import (
    average_log_divergence,
    delay_embed,
    iaaft_surrogate,
    rosenstein_lyapunov,
    surrogate_test,
)


def test_delay_embed_shape_and_values():
    series = np.arange(10.0)
    embedded = delay_embed(series, dim=3, tau=2)
    assert embedded.shape == (6, 3)
    np.testing.assert_allclose(embedded[0], [0.0, 2.0, 4.0])
    np.testing.assert_allclose(embedded[-1], [5.0, 7.0, 9.0])


def test_delay_embed_rejects_series_too_short():
    with pytest.raises(ValueError):
        delay_embed(np.arange(5.0), dim=10, tau=1)


def test_average_log_divergence_returns_matching_shapes():
    rng = np.random.default_rng(0)
    series = rng.normal(size=500)
    steps, log_div = average_log_divergence(series, dim=3, tau=1, max_iter=20)
    assert steps.shape == log_div.shape
    assert steps.size > 0


def test_rosenstein_lyapunov_matches_known_lorenz_exponent():
    """The Lorenz system's largest Lyapunov exponent is well known to be
    approximately 0.905; estimating it from a raw x(t) time series (no
    equations used) with Rosenstein's algorithm should get reasonably close."""
    system = Lorenz()
    dt = 0.01
    _, states = system.trajectory(n_steps=20000, dt=dt)
    x = states[1000:, 0]

    lam = rosenstein_lyapunov(x, dim=5, tau=10, dt=dt, fit_fraction=0.3)
    assert 0.5 < lam < 1.3


def test_rosenstein_lyapunov_rejects_too_short_series():
    with pytest.raises(ValueError):
        rosenstein_lyapunov(np.arange(5.0), dim=3, tau=1)


def test_iaaft_surrogate_preserves_mean_and_std():
    rng = np.random.default_rng(0)
    series = rng.normal(loc=3.0, scale=2.0, size=1000)
    surrogate = iaaft_surrogate(series, n_iter=50, seed=1)
    assert surrogate.mean() == pytest.approx(series.mean(), abs=1e-9)
    assert surrogate.std() == pytest.approx(series.std(), abs=1e-9)


def test_iaaft_surrogate_preserves_power_spectrum_closely():
    system = Lorenz()
    _, states = system.trajectory(n_steps=4000, dt=0.01)
    x = states[500:, 0]

    surrogate = iaaft_surrogate(x, n_iter=100, seed=0)
    original_spectrum = np.abs(np.fft.rfft(x))
    surrogate_spectrum = np.abs(np.fft.rfft(surrogate))
    relative_error = np.max(np.abs(original_spectrum - surrogate_spectrum)) / np.max(original_spectrum)
    assert relative_error < 0.05


def test_surrogate_test_detects_nonlinear_structure_in_lorenz():
    """Lorenz x(t) has genuine deterministic/nonlinear structure that IAAFT
    surrogates (linear stochastic realizations with the same spectrum) lack,
    so the observed statistic should be many surrogate-std-devs away from the
    surrogate distribution."""
    system = Lorenz()
    dt = 0.01
    _, states = system.trajectory(n_steps=6000, dt=dt)
    x = states[1000:, 0]

    def statistic(series):
        return rosenstein_lyapunov(series, dim=5, tau=10, dt=dt, fit_fraction=0.3)

    _observed, surrogate_values, significance = surrogate_test(x, statistic, n_surrogates=9, n_iter=30, seed=0)
    assert surrogate_values.shape == (9,)
    assert significance > 2.0


def test_surrogate_test_finds_no_structure_in_pure_noise():
    """White noise has no structure beyond its (flat) spectrum and
    distribution, so a nonlinear statistic (here, the same Rosenstein
    exponent used to detect Lorenz's structure) should come out statistically
    indistinguishable between the noise and its own surrogates (low
    significance) -- unlike a trivially IAAFT-invariant statistic such as the
    standard deviation, which every surrogate reproduces exactly by
    construction and would give a meaningless zero-variance comparison."""
    rng = np.random.default_rng(0)
    noise = rng.normal(size=2000)

    def statistic(series):
        return rosenstein_lyapunov(series, dim=5, tau=1, fit_fraction=0.3)

    _observed, _surrogate_values, significance = surrogate_test(noise, statistic, n_surrogates=9, n_iter=20, seed=1)
    assert significance < 2.0
