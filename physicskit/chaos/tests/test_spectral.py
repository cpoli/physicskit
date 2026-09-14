import numpy as np
import pytest

from physicskit.chaos.utils.spectral import autocorrelation, power_spectrum


def test_power_spectrum_of_pure_sine_peaks_at_its_frequency():
    dt = 0.01
    t = np.arange(0, 10, dt)
    freq = 3.0
    series = np.sin(2.0 * np.pi * freq * t)

    frequencies, power = power_spectrum(series, dt=dt)
    peak_freq = frequencies[np.argmax(power)]
    assert peak_freq == pytest.approx(freq, abs=0.05)


def test_power_spectrum_shapes_and_nonnegative():
    series = np.random.default_rng(0).normal(size=200)
    frequencies, power = power_spectrum(series, dt=1.0)
    assert frequencies.shape == power.shape == (101,)
    assert np.all(power >= 0.0)


def test_autocorrelation_of_pure_sine_is_periodic():
    dt = 0.01
    t = np.arange(0, 20, dt)
    freq = 2.0
    series = np.sin(2.0 * np.pi * freq * t)

    acf = autocorrelation(series, max_lag=len(series) - 1)
    assert acf[0] == pytest.approx(1.0)
    period_samples = round(1.0 / freq / dt)
    assert acf[period_samples] == pytest.approx(1.0, abs=0.05)


def test_autocorrelation_of_white_noise_decays_immediately():
    rng = np.random.default_rng(0)
    series = rng.normal(size=5000)
    acf = autocorrelation(series, max_lag=50)
    assert acf[0] == pytest.approx(1.0)
    assert abs(acf[1]) < 0.1


def test_autocorrelation_default_max_lag():
    series = np.random.default_rng(0).normal(size=50)
    acf = autocorrelation(series)
    assert acf.shape == (50,)
