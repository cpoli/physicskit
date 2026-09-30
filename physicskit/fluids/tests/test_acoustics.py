"""Linear acoustics: pipe and room eigenmodes from time-domain simulations."""

import numpy as np
import pytest

from physicskit.fluids.exceptions import InvalidParameterError
from physicskit.fluids.systems.acoustics import (
    acoustic_wave_1d,
    acoustic_wave_2d,
    ideal_gas_sound_speed,
    pipe_mode_frequencies,
    pipe_mode_shape,
    rectangular_room_mode_frequencies,
)


def _spectral_peaks(trace, dt, n_peaks):
    """Frequencies of the lowest significant local maxima of the windowed power spectrum, refined parabolically."""
    sig = (trace - trace.mean()) * np.hanning(len(trace))
    P = np.abs(np.fft.rfft(sig, 8 * len(sig))) ** 2
    f = np.fft.rfftfreq(8 * len(sig), dt)
    is_peak = (P[1:-1] > P[:-2]) & (P[1:-1] >= P[2:]) & (P[1:-1] > 1e-3 * P.max())
    peaks = (np.flatnonzero(is_peak) + 1)[:n_peaks]
    out = []
    for i in peaks:
        a, b, c = np.log(P[i - 1]), np.log(P[i]), np.log(P[i + 1])
        out.append(f[i] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0]))
    return np.array(out)


@pytest.mark.parametrize("ends", [("open", "open"), ("closed", "closed"), ("closed", "open"), ("open", "closed")])
def test_pipe_eigenmodes_from_impulse_response(ends):
    """An off-center pressure pulse rings at exactly the pipe's modal frequencies."""
    L, c, n = 1.0, 340.0, 400
    x = (np.arange(n) + 0.5) * L / n
    p0 = np.exp(-(((x - 0.27 * L) / 0.01) ** 2))  # 0.27 L is not near a node of any of the first four modes
    _, _, trace, dt = acoustic_wave_1d(p0, L, c, t_max=0.4, ends=ends, n_frames=1, probe=int(0.13 * n))
    expected = pipe_mode_frequencies(L, c, 4, ends)
    assert _spectral_peaks(trace, dt, 4) == pytest.approx(expected, rel=2e-3)


@pytest.mark.parametrize("ends,mode", [(("closed", "closed"), 2), (("open", "open"), 3), (("closed", "open"), 2)])
def test_standing_mode_oscillates_at_its_frequency(ends, mode):
    """Seeded with a mode shape, the pressure is p(x) cos(2 pi f t): after a half period it is inverted."""
    L, c, n = 2.0, 1.0, 800
    x = (np.arange(n) + 0.5) * L / n
    shape = pipe_mode_shape(x, L, mode, ends)
    f = pipe_mode_frequencies(L, c, mode, ends)[-1]
    dt = 0.25 * L / n / c
    half = 0.5 / f
    _, P, _, _ = acoustic_wave_1d(shape, L, c, t_max=round(half / dt) * dt, dt=dt, ends=ends, n_frames=1)
    assert P[-1] == pytest.approx(-shape, abs=5e-3)


def test_closed_pipe_conserves_mean_pressure_and_energy():
    L, c, n, rho0 = 1.0, 2.0, 200, 1.3
    x = (np.arange(n) + 0.5) * L / n
    p0 = np.exp(-(((x - 0.4) / 0.05) ** 2))
    _, P, _, dt = acoustic_wave_1d(p0, L, c, t_max=3.0, rho0=rho0, n_frames=30)
    assert P.mean(axis=1) == pytest.approx(np.full(len(P), p0.mean()), abs=1e-12)
    assert np.max(np.abs(P)) <= 1.0 + 1e-9


def test_room_modes_2d():
    """A rigid rectangular room rings at f_mn = (c/2) sqrt((m/Lx)^2 + (n/Ly)^2)."""
    Lx, Ly, c = 4.0, 3.0, 340.0
    nx, ny = 80, 60
    xc = (np.arange(nx) + 0.5) * Lx / nx
    yc = (np.arange(ny) + 0.5) * Ly / ny
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    p0 = np.exp(-((X - 1.1) ** 2 + (Y - 0.7) ** 2) / 0.05**2)
    _, _, trace, dt = acoustic_wave_2d(p0, (Lx, Ly), c, t_max=0.5, n_frames=1, probe=(7, 11))
    expected = rectangular_room_mode_frequencies(Lx, Ly, c, n_max=2)[:3, 2]
    assert _spectral_peaks(trace, dt, 3) == pytest.approx(expected, rel=5e-3)


def test_sound_speed_and_validation():
    assert ideal_gas_sound_speed(1.4, 1e5, 1.0) == pytest.approx(np.sqrt(1.4e5))
    assert pipe_mode_frequencies(0.5, 340.0, 2).tolist() == [340.0, 680.0]
    with pytest.raises(InvalidParameterError):
        acoustic_wave_1d(np.zeros(10), 1.0, 1.0, 1.0, dt=0.2)
    with pytest.raises(InvalidParameterError):
        pipe_mode_frequencies(1.0, 340.0, ends=("open", "flanged"))
    with pytest.raises(InvalidParameterError):
        acoustic_wave_2d(np.zeros((4, 4)), (1.0, 1.0), 1.0, 1.0, dt=1.0)
