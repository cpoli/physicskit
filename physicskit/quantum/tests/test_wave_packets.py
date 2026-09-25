"""Tests for physicskit.quantum.chapters.wave_packets: the parts of
GaussianDispersion (center, density) and TwinSlit not already exercised
by test_physics_checks.py/test_animations.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.quantum._compat import trapz
from physicskit.quantum.chapters.wave_packets import GaussianDispersion, TwinSlit


def test_gaussian_dispersion_center_moves_ballistically():
    gd = GaussianDispersion(x0=1.0, k0=3.0, hbar=1.0, m=2.0)
    t = np.array([0.0, 1.0, 2.5])
    np.testing.assert_allclose(gd.center(t), gd.x0 + (gd.hbar * gd.k0 / gd.m) * t)


def test_gaussian_dispersion_density_matches_abs_psi_squared():
    gd = GaussianDispersion()
    x = np.linspace(-10, 10, 200)
    np.testing.assert_allclose(gd.density(x, t=0.7), np.abs(gd.psi(x, 0.7)) ** 2)


def test_gaussian_dispersion_density_stays_normalized_over_time():
    gd = GaussianDispersion(sigma0=1.0, k0=0.0)  # k0=0 keeps the packet centered at x0
    x = np.linspace(-40, 40, 4000)
    for t in (0.0, 5.0, 20.0):
        assert trapz(gd.density(x, t), x) == pytest.approx(1.0, abs=1e-3)


def test_twin_slit_intensity_is_normalized():
    ts = TwinSlit(slit_separation=4.0, slit_width=0.4, k0=10.0)
    x = np.linspace(-20, 20, 4000)
    I = ts.intensity(x, screen_distance=50.0)
    assert trapz(I, x) == pytest.approx(1.0, abs=1e-6)


def test_twin_slit_intensity_is_symmetric_about_the_axis():
    ts = TwinSlit(slit_separation=4.0, slit_width=0.4, k0=10.0)
    x = np.linspace(-20, 20, 501)  # ascending and symmetric, so trapz normalization matches
    I = ts.intensity(x, screen_distance=50.0)
    np.testing.assert_allclose(I, I[::-1], rtol=1e-10)


def test_twin_slit_amplitude_is_complex_and_finite():
    ts = TwinSlit()
    x = np.linspace(-10, 10, 50)
    amp = ts.amplitude(x, screen_distance=30.0)
    assert np.iscomplexobj(amp)
    assert np.all(np.isfinite(amp))


def test_twin_slit_single_slit_envelope_narrows_as_the_slit_widens():
    # Far field of a Gaussian aperture exp(-y^2/w^2): envelope exp(-(k w sin(theta))^2 / 4).
    L, k = 50.0, 10.0
    x = np.array([0.0, 20.0])
    for w in (0.2, 0.6):
        ts = TwinSlit(slit_separation=0.0, slit_width=w, k0=k)  # coincident slits: envelope only
        amp = np.abs(ts.amplitude(x, screen_distance=L))
        r = np.sqrt(L**2 + x**2)
        expected_ratio = np.exp(-((k * w * x[1] / r[1]) ** 2) / 4.0) * np.sqrt(r[0] / r[1])
        assert amp[1] / amp[0] == pytest.approx(expected_ratio, rel=1e-12)
