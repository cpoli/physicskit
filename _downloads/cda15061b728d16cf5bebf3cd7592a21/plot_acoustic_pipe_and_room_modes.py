r"""
Sound in pipes and rooms: standing modes from a single pulse
===============================================================

Small pressure disturbances in air travel at :math:`c = \sqrt{\gamma p/\rho}`
and obey the wave equation. In a pipe, reflections at the two ends allow
only a discrete set of frequencies to persist: its standing modes. An end
closed by a rigid cap reflects a pressure maximum. An open end, to a good
approximation, fixes the pressure at ambient. As a result a pipe open at
both ends (a flute) sounds all harmonics :math:`f_n = nc/2L`, while a pipe
closed at one end (a clarinet, roughly) sounds only the odd ones,
:math:`f_n = (2n-1)c/4L`, an octave lower.

This example strikes each pipe once with a short pressure pulse, using
:func:`~physicskit.fluids.acoustic_wave_1d`, listens at one point, and
reads the modes off the spectrum. It then does the same for a rectangular
room with :func:`~physicskit.fluids.acoustic_wave_2d`.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from physicskit.fluids import (
    acoustic_wave_1d,
    acoustic_wave_2d,
    ideal_gas_sound_speed,
    pipe_mode_frequencies,
    rectangular_room_mode_frequencies,
)

c = ideal_gas_sound_speed(1.4, 101325.0, 1.204)
print(f"speed of sound in air at 20 C: {c:.1f} m/s")


def spectrum(trace, dt):
    sig = (trace - trace.mean()) * np.hanning(len(trace))
    P = np.abs(np.fft.rfft(sig, 4 * len(sig))) ** 2
    return np.fft.rfftfreq(4 * len(sig), dt), P / P.max()


# %%
# Open and half-closed pipes
# --------------------------

L, n = 0.6, 300
x = (np.arange(n) + 0.5) * L / n
pulse = np.exp(-(((x - 0.27 * L) / 0.01) ** 2))
results = {}
for ends in [("open", "open"), ("closed", "open")]:
    t, P, trace, dt = acoustic_wave_1d(pulse, L, c, t_max=0.3, ends=ends, n_frames=200, probe=int(0.11 * n))
    results[ends] = (t, P, *spectrum(trace, dt))
    print(f"{ends[0]}-{ends[1]} pipe, predicted modes (Hz): {np.round(pipe_mode_frequencies(L, c, 5, ends), 1)}")

# %%
# A rectangular room
# ------------------
# A 4 m x 3 m room with rigid walls rings at
# :math:`f_{mn} = \tfrac{c}{2}\sqrt{(m/L_x)^2 + (n/L_y)^2}`.

Lx, Ly, nx, ny = 4.0, 3.0, 80, 60
X, Y = np.meshgrid((np.arange(nx) + 0.5) * Lx / nx, (np.arange(ny) + 0.5) * Ly / ny, indexing="ij")
p0 = np.exp(-((X - 1.1) ** 2 + (Y - 0.7) ** 2) / 0.05**2)
t2, P2, trace2, dt2 = acoustic_wave_2d(p0, (Lx, Ly), c, t_max=0.5, n_frames=50, probe=(7, 11))
room = rectangular_room_mode_frequencies(Lx, Ly, c, n_max=2)
print("lowest room modes (m, n, Hz):", [(int(m), int(k), round(float(f), 1)) for m, k, f in room[:4]])

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
t, P, f, S = results[("open", "open")]
axes[0, 0].imshow(P.T, aspect="auto", origin="lower", extent=(0, t[-1] * 1e3, 0, L), cmap="RdBu_r", vmin=-0.3, vmax=0.3)
axes[0, 0].set_xlabel("t (ms)")
axes[0, 0].set_ylabel("x (m)")
axes[0, 0].set_title("open-open pipe: the pulse bounces, inverting at each open end")
axes[0, 0].set_xlim(0, 12)

for ends, color in [(("open", "open"), "tab:blue"), (("closed", "open"), "tab:orange")]:
    _, _, f, S = results[ends]
    axes[0, 1].semilogy(f, S, color=color, label=f"{ends[0]}-{ends[1]}")
    for fm in pipe_mode_frequencies(L, c, 6, ends):
        axes[0, 1].axvline(fm, color=color, ls=":", lw=0.8)
axes[0, 1].set_xlim(0, 1800)
axes[0, 1].set_ylim(1e-6, 2)
axes[0, 1].set_xlabel("frequency (Hz)")
axes[0, 1].set_title("spectra (dotted: nc/2L and (2n-1)c/4L)")
axes[0, 1].legend()

f2, S2 = spectrum(trace2, dt2)
axes[1, 0].semilogy(f2, S2)
for m, k, fm in room[:8]:
    axes[1, 0].axvline(fm, color="gray", ls=":", lw=0.8)
    axes[1, 0].text(fm, 1.5, f"{int(m)}{int(k)}", fontsize=7, ha="center")
axes[1, 0].set_xlim(0, 180)
axes[1, 0].set_ylim(1e-5, 3)
axes[1, 0].set_xlabel("frequency (Hz)")
axes[1, 0].set_title("room spectrum, labeled by mode (m, n)")

im = axes[1, 1].imshow(P2[8].T, origin="lower", extent=(0, Lx, 0, Ly), cmap="RdBu_r")
axes[1, 1].set_title(f"room pressure at t = {t2[8] * 1e3:.1f} ms")
fig.colorbar(im, ax=axes[1, 1])
fig.tight_layout()
plt.show()

# %%
# Check
# -----
# c = sqrt(gamma p / rho); pipe resonances n c / 2L (open-open) and
# (2n - 1) c / 4L (closed-open); room modes (c/2) sqrt((m/Lx)^2 + (n/Ly)^2).
assert abs(c - np.sqrt(1.4 * 101325.0 / 1.204)) < 1e-9
for ends, n_modes in ((("open", "open"), np.arange(1, 5)), (("closed", "open"), 2 * np.arange(1, 5) - 1)):
    _, _, f, S = results[ends]
    expected = n_modes * c / (2 * L if ends[0] == "open" else 4 * L)
    np.testing.assert_allclose(pipe_mode_frequencies(L, c, 4, ends), expected, rtol=1e-12)
    peaks = f[find_peaks(S, prominence=0.05)[0]][:4]
    np.testing.assert_allclose(peaks, expected, atol=3.0)
room_peaks = f2[find_peaks(S2, prominence=0.05)[0]][:3]
np.testing.assert_allclose(room_peaks, [r[2] for r in room[:3]], atol=1.5)
