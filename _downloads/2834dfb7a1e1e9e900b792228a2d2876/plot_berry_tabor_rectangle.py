r"""
The Berry-Tabor formula: an integrable billiard's spectrum
=============================================================

Gutzwiller's trace formula assumes each periodic orbit is isolated. In
an integrable system that fails. Periodic orbits come in continuous
families filling whole resonant tori, and the Gutzwiller stability
weight :math:`1/\sqrt{|\det(M-I)|}` diverges. Berry and Tabor (1976)
replaced the sum over isolated orbits with a sum over these families,
labelled by integer winding numbers :math:`(M, N)`. Each family adds an
oscillation to the density of states with amplitude proportional to the
area it sweeps and falling off only as :math:`L^{-1/2}` with its length
:math:`L`.

The rectangular billiard (sides :math:`a, b`) shows this with exact
quantum levels :math:`k_{mn}^2 = \pi^2(m^2/a^2 + n^2/b^2)` and periodic-orbit
families of length :math:`L_{MN} = 2\sqrt{(Ma)^2 + (Nb)^2}`. The *length
spectrum*, the Fourier transform of the exact levels in wavenumber, has
a peak at each of those lengths. For contrast, this example also
evaluates
:func:`~physicskit.semiclassical.core.gutzwiller.gutzwiller_amplitude_from_monodromy`
on a family orbit's monodromy matrix, which has :math:`\operatorname{tr}M = 2`,
to show where the isolated-orbit formula breaks.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.semiclassical.core.gutzwiller import gutzwiller_amplitude_from_monodromy

a, b = 1.0, (1 + np.sqrt(5)) / 2  # irrational aspect ratio: no accidental degeneracies

# %%
# Exact levels
# ----------------
m, n = np.meshgrid(np.arange(1, 400), np.arange(1, 400), indexing="ij")
k_all = np.pi * np.sqrt((m / a) ** 2 + (n / b) ** 2)
K_max = 250.0
k = np.sort(k_all[k_all < K_max])
weyl = a * b * k**2 / (4 * np.pi) - 2 * (a + b) * k / (4 * np.pi)
print(f"{k.size} levels below k = {K_max}; Weyl estimate {a * b * K_max**2 / (4 * np.pi) - 2 * (a + b) * K_max / (4 * np.pi):.0f}")

# %%
# The length spectrum
# -----------------------
# :math:`|\sum_n w(k_n)\,e^{ik_nL}|`, with a smooth window :math:`w` to
# suppress ringing from the cutoff. Peaks sit at the family lengths
# :math:`L_{MN}`; the Weyl term only contributes near :math:`L=0`.
L = np.linspace(0.3, 9.0, 3000)
window = np.cos(0.5 * np.pi * k / K_max) ** 2
length_spectrum = np.abs(np.exp(1j * np.outer(L, k)) @ window)
families = {(M, N): 2 * np.hypot(M * a, N * b) for M in range(0, 5) for N in range(0, 4) if (M, N) != (0, 0)}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
ax1.plot(L, length_spectrum, color="steelblue", lw=1)
for (M, N), LMN in sorted(families.items(), key=lambda kv: kv[1]):
    if LMN < L[-1]:
        ax1.axvline(LMN, color="firebrick", ls=":", lw=0.8)
        ax1.text(LMN, length_spectrum.max() * 1.02, f"({M},{N})", rotation=90, fontsize=7, ha="center", va="bottom")
ax1.set_xlabel("orbit length L")
ax1.set_ylabel("length spectrum")
ax1.set_title("Fourier transform of the exact levels: one peak per orbit family", pad=28)

found = []
for (M, N), LMN in sorted(families.items(), key=lambda kv: kv[1])[:6]:
    window_mask = np.abs(L - LMN) < 0.1
    found.append((M, N, LMN, L[window_mask][np.argmax(length_spectrum[window_mask])]))
print("\nfamily (M,N): predicted length, peak in the length spectrum")
for M, N, LMN, Lpk in found:
    print(f"  ({M},{N}): {LMN:.3f}  {Lpk:.3f}")

# %%
# Why Gutzwiller's weight fails here
# --------------------------------------
# A bouncing-ball orbit between the two sides of length :math:`a`
# (family (0,1)) is neutrally stable. Displacing it sideways gives
# another periodic orbit, so its monodromy matrix is a shear,
# :math:`\begin{pmatrix}1&L\\0&1\end{pmatrix}`, with trace 2. The isolated-orbit
# amplitude :math:`1/\sqrt{|2-\operatorname{tr}M|}` is infinite, and a
# whole family must be summed instead, which is what Berry and Tabor did.
L01 = families[(0, 1)]
for eps in (1e-1, 1e-3, 1e-6):
    M_near = np.array([[1.0 + eps, L01], [0.0, 1.0 / (1.0 + eps)]])  # slightly unstable perturbation
    print(f"trace M = {np.trace(M_near):.8f}: Gutzwiller amplitude {gutzwiller_amplitude_from_monodromy(M_near):.3e}")

# %%
# Fluctuations about Weyl's law
# ---------------------------------
# The counting function :math:`N(k)` minus Weyl's smooth estimate is the
# oscillatory part that the Berry-Tabor sum describes. It fluctuates about
# zero with no trend, and its size grows slowly with :math:`k`, as the
# :math:`L^{-1/2}` family amplitudes predict.
ax2.plot(k, np.arange(1, k.size + 1) - weyl, color="darkorange", lw=0.6)
ax2.axhline(0, color="k", lw=0.6)
ax2.set_xlabel("wavenumber k")
ax2.set_ylabel("N(k) - Weyl")
ax2.set_title("Spectral fluctuations")
fig.tight_layout()

plt.show()
