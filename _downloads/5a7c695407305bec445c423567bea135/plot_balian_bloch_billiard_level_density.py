r"""
Balian and Bloch's level-density expansion for billiards
===========================================================

Weyl's law :math:`N(k)\simeq Ak^2/4\pi` is only the first term. Balian
and Bloch (1970-1972) built the level density of a 2D billiard from a
multiple-reflection expansion of its Green function and obtained a full
asymptotic series. The smooth part is

.. math::

    \bar N(k) = \frac{A k^2}{4\pi} \mp \frac{L k}{4\pi}
    + \sum_{\rm corners}\frac{\pi^2-\theta_i^2}{24\pi\theta_i}
    + \frac{1}{12\pi}\oint\kappa\,ds,

with the perimeter :math:`L` entering with a minus sign for Dirichlet
walls. What remains, :math:`N-\bar N`, oscillates around zero, and its
Fourier transform has a peak at the length of each closed classical
orbit: the same periodic orbits as in Gutzwiller's formula, here
appearing in a billiard. This example checks both statements with
:func:`~physicskit.semiclassical.core.gutzwiller.balian_bloch_counting_function`
on the unit disk (exact levels are Bessel zeros) and a 2:1 rectangle.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jn_zeros

from physicskit.semiclassical.core.gutzwiller import balian_bloch_counting_function

k_max = 70.0
disk = []
for m in range(int(k_max) + 1):
    z = jn_zeros(m, int(k_max))
    z = z[z < k_max]
    disk.extend(z if m == 0 else np.repeat(z, 2))
disk = np.sort(disk)

a, b = 2.0, 1.0
n = np.arange(1, 100)
rect = np.sort((np.pi * np.hypot(n[:, None] / a, n[None, :] / b)).ravel())
rect = rect[rect < k_max]

shapes = {
    "unit disk": dict(levels=disk, area=np.pi, perimeter=2 * np.pi, corner_angles=(), total_curvature=2 * np.pi),
    "2:1 rectangle": dict(levels=rect, area=a * b, perimeter=2 * (a + b), corner_angles=[np.pi / 2] * 4, total_curvature=0.0),
}

# %%
# Term by term
# ----------------
# Subtract the Weyl area term from the exact staircase: what is left falls
# linearly, at the rate :math:`-L/4\pi` set by the perimeter. Subtract the
# perimeter term as well and the remainder fluctuates around the constant
# from corners and curvature, :math:`1/4` for the rectangle and
# :math:`1/6` for the disk.
k = np.linspace(1.0, k_max - 0.5, 6000)
fig1, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for ax, (name, sh) in zip(axes, shapes.items()):
    N = np.searchsorted(sh["levels"], k)
    weyl = sh["area"] * k**2 / (4 * np.pi)
    full = balian_bloch_counting_function(k, sh["area"], sh["perimeter"], sh["corner_angles"], sh["total_curvature"])
    constant = balian_bloch_counting_function(0.0, sh["area"], sh["perimeter"], sh["corner_angles"], sh["total_curvature"])
    ax.plot(k, N - weyl, color="0.6", lw=0.5, label=r"$N - Ak^2/4\pi$")
    ax.plot(k, full - weyl, color="firebrick", lw=1.5, label=r"$-Lk/4\pi + C$")
    ax.set_xlabel("wavenumber k")
    ax.set_title(f"{name}: {len(sh['levels'])} levels")
    ax.legend(fontsize=8)
    resid = N - full
    print(f"{name:14s}: C = {constant:.4f}, mean of N - Nbar for k > 10: {resid[k > 10].mean():+.4f} (std {resid[k > 10].std():.2f})")
axes[0].set_ylabel("levels")
fig1.tight_layout()

# %%
# What the smooth terms cannot see
# ------------------------------------
# The oscillating remainder is a sum of waves :math:`\cos(k\ell)`, one for
# every closed orbit of length :math:`\ell`. A windowed Fourier transform
# of the level sequence, :math:`\sum_n w(k_n)e^{ik_n\ell}`, picks them out.
# In the disk the closed orbits are inscribed polygons: the diameter
# bounced twice (:math:`4R`), the triangle (:math:`3\sqrt3R`), the square
# (:math:`4\sqrt2R`), the pentagon, and so on towards the circumference.
# In the rectangle they are the bouncing orbits :math:`2b` and
# :math:`2a=4b` and the diagonal families
# :math:`2\sqrt{(n_xa)^2+(n_yb)^2}`.
ell = np.linspace(0.5, 7.5, 3000)
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 3.8))
for ax, (name, sh) in zip(axes2, shapes.items()):
    kn = sh["levels"][sh["levels"] > 5]
    w = np.sin(np.pi * (kn - 5) / (k_max - 5)) ** 2
    spectrum = np.abs(np.exp(1j * np.outer(ell, kn)) @ w) / w.sum()
    ax.plot(ell, spectrum, color="steelblue", lw=1)
    ax.set_xlabel(r"orbit length $\ell$")
    ax.set_title(f"{name}: length spectrum of the levels")
    if name == "unit disk":
        orbits = {"diameter": 4.0, "triangle": 3 * np.sqrt(3), "square": 4 * np.sqrt(2), "pentagon": 10 * np.sin(np.pi / 5)}
    else:
        orbits = {"2b": 2 * b, "2a": 2 * a}
        orbits.update({rf"$2\sqrt{{a^2+{n**2}b^2}}$": 2 * np.hypot(a, n * b) for n in (1, 2, 3)})
    for label, L in orbits.items():
        ax.axvline(L, color="firebrick", ls=":", lw=1)
        ax.text(L, ax.get_ylim()[1] * 0.98, label, rotation=90, fontsize=8, ha="left", va="top", color="firebrick")
axes2[0].set_ylabel("|amplitude|")
fig2.tight_layout()

plt.show()

# %%
# Check
# -----
# Weyl + boundary + curvature/corner terms: the constant is 1/6 for any
# smooth billiard (the disk) and sum over corners (pi^2 - a^2) / 24 pi a = 1/4
# for the rectangle; the counting function fluctuates about N-bar with zero mean.
expected_C = {"unit disk": 1 / 6, "2:1 rectangle": 1 / 4}
for name, sh in shapes.items():
    C = balian_bloch_counting_function(0.0, sh["area"], sh["perimeter"], sh["corner_angles"], sh["total_curvature"])
    assert abs(C - expected_C[name]) < 1e-12
    resid = np.searchsorted(sh["levels"], k) - balian_bloch_counting_function(k, sh["area"], sh["perimeter"], sh["corner_angles"], sh["total_curvature"])
    assert abs(resid[k > 10].mean()) < 0.05
