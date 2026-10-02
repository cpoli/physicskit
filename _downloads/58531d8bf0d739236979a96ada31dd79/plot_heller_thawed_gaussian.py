r"""
Heller's thawed Gaussian wavepacket dynamics
===============================================

Heller (1975) proposed the simplest semiclassical wavepacket method: keep
the packet a Gaussian,

.. math::

    \psi(x,t)=\exp\!\left\{\tfrac{i}{\hbar}\left[\alpha_t(x-q_t)^2
    +p_t(x-q_t)+s_t\right]\right\},

and expand the potential to second order about its centre. The centre
then follows Hamilton's equations, the complex width :math:`\alpha_t`
obeys a Riccati equation driven by the local curvature :math:`V''(q_t)`,
and :math:`s_t` collects the action and normalization. Unlike a frozen
Gaussian of fixed width, the width is free ("thawed") to breathe, squeeze
and shear.

This example uses
:func:`~physicskit.semiclassical.core.propagators.thawed_gaussian_propagate`
and compares it with exact split-operator evolution
(:class:`~physicskit.quantum.core.solvers.SplitOperatorSolver1D`) and with
a rigid Gaussian that only follows the classical centre. Units: :math:`m=1`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from physicskit.quantum.core.solvers import SplitOperatorSolver1D
from physicskit.semiclassical.core.propagators import frozen_gaussian_1d, thawed_gaussian_propagate, thawed_gaussian_wavefunction


def fidelity(a, b, dx):
    return abs(np.sum(a.conj() * b) * dx) ** 2


# %%
# A squeezed state in a harmonic well
# ---------------------------------------
# In a quadratic potential the thawed Gaussian is exact. A packet narrower
# than the ground state breathes at twice the oscillator frequency: it
# spreads, refocuses and spreads again. The frozen Gaussian cannot, and its
# overlap with the true state drops every half period.
V_ho = njit(lambda q, params: 0.5 * q**2, cache=False)
dV_ho = njit(lambda q, params: q, cache=False)
d2V_ho = njit(lambda q, params: 1.0, cache=False)

x = np.linspace(-12, 12, 1024)
dx = x[1] - x[0]
gamma_sq, dt = 3.0, 0.005  # ground state has gamma = 1/2
steps = int(round(2 * 2 * np.pi / dt))
t, q, p, alpha, s = thawed_gaussian_propagate(2.0, 0.0, gamma_sq, dV_ho, d2V_ho, V_ho, 1.0, dt, steps)
save = 20
frames, times = SplitOperatorSolver1D(x, lambda xx: 0.5 * xx**2, dt=dt).propagate(frozen_gaussian_1d(x, 2.0, 0.0, gamma_sq), steps, save_every=save)
ks = np.arange(0, steps + 1, save)
f_tg = [fidelity(f, thawed_gaussian_wavefunction(x, q[k], p[k], alpha[k], s[k]), dx) for f, k in zip(frames, ks)]
f_fr = [fidelity(f, frozen_gaussian_1d(x, q[k], p[k], gamma_sq), dx) for f, k in zip(frames, ks)]

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(t / (2 * np.pi), np.sqrt(1 / (4 * alpha.imag)), color="firebrick", label=r"thawed width $\sqrt{\hbar/4\,\mathrm{Im}\,\alpha_t}$")
ax1.axhline(np.sqrt(1 / (4 * gamma_sq)), color="steelblue", ls="--", label="frozen width")
ax1.axhline(np.sqrt(0.5), color="0.5", ls=":", label="ground-state width")
ax1.set_xlabel("t / period")
ax1.set_ylabel("position spread")
ax1.set_title("Squeezed packet: the width breathes")
ax1.legend(fontsize=8)
ax2.plot(times / (2 * np.pi), f_tg, color="firebrick", label="thawed Gaussian")
ax2.plot(times / (2 * np.pi), f_fr, color="steelblue", label="frozen Gaussian on the same centre")
ax2.set_xlabel("t / period")
ax2.set_ylabel("fidelity with exact state")
ax2.set_ylim(0, 1.05)
ax2.set_title("Harmonic well: the thawed Gaussian is exact")
ax2.legend(fontsize=8)
fig1.tight_layout()
print(f"harmonic well: min fidelity thawed {min(f_tg):.10f}, frozen {min(f_fr):.4f}")
f_tg_harmonic, f_fr_harmonic = f_tg, f_fr

# %%
# A Morse well: the semiclassical limit
# -----------------------------------------
# In an anharmonic well the method is an approximation. The packet's
# energy spread makes its parts oscillate with different periods, so it
# shears in phase space; the thawed width captures the linear part of that
# shear, but the curvature :math:`V''` changes across the packet. The
# error is controlled by the packet's size, which for a minimum-uncertainty
# packet scales as :math:`\sqrt\hbar`. Shrinking :math:`\hbar` with the
# classical motion fixed (:math:`\gamma\propto1/\hbar`) therefore makes the
# thawed Gaussian converge to the exact state. The frozen Gaussian does not
# converge, because the shape change it misses is a classical effect.
D_e, a_m = 20.0, 0.3
params = np.array([D_e, a_m])


@njit
def V(q, params):
    return params[0] * (1 - np.exp(-params[1] * q)) ** 2


@njit
def dVdx(q, params):
    e = np.exp(-params[1] * q)
    return 2 * params[0] * params[1] * e * (1 - e)


@njit
def d2Vdx2(q, params):
    e = np.exp(-params[1] * q)
    return 2 * params[0] * params[1] ** 2 * e * (2 * e - 1)


x = np.linspace(-8, 30, 4096)
dx = x[1] - x[0]
q0, p0, t_final, dt = 0.0, 3.0, 4.0, 0.002
steps, save = int(round(t_final / dt)), 25
hbars = [1.0, 0.25, 0.0625]
colors = ["0.6", "darkorange", "firebrick"]

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))
fidelity_at_2 = {}
for hbar, color in zip(hbars, colors):
    gamma = 1.0 / hbar
    t, q, p, alpha, s = thawed_gaussian_propagate(q0, p0, gamma, dVdx, d2Vdx2, V, 1.0, dt, steps, hbar=hbar, params=params)
    solver = SplitOperatorSolver1D(x, lambda xx: D_e * (1 - np.exp(-a_m * xx)) ** 2, dt=dt, hbar=hbar)
    frames, times = solver.propagate(frozen_gaussian_1d(x, q0, p0, gamma, hbar), steps, save_every=save)
    ks = np.arange(0, steps + 1, save)
    f_tg = [fidelity(f, thawed_gaussian_wavefunction(x, q[k], p[k], alpha[k], s[k], hbar), dx) for f, k in zip(frames, ks)]
    f_fr = [fidelity(f, frozen_gaussian_1d(x, q[k], p[k], gamma, hbar), dx) for f, k in zip(frames, ks)]
    ax3.plot(times, f_tg, color=color, label=rf"thawed, $\hbar={hbar:g}$")
    ax3.plot(times, f_fr, color=color, ls="--", lw=1, label=rf"frozen, $\hbar={hbar:g}$")
    print(f"Morse, hbar = {hbar:6.4f}: fidelity at t = 2: thawed {f_tg[len(ks) // 2]:.3f}, frozen {f_fr[len(ks) // 2]:.3f}")
    fidelity_at_2[hbar] = (f_tg[len(ks) // 2], f_fr[len(ks) // 2])
    if hbar == hbars[-1]:
        k = steps // 2
        ax4.plot(x, np.abs(frames[len(ks) // 2]) ** 2, color="k", lw=3, alpha=0.35, label="exact")
        ax4.plot(x, np.abs(thawed_gaussian_wavefunction(x, q[k], p[k], alpha[k], s[k], hbar)) ** 2, color="firebrick", label="thawed")
        ax4.plot(x, np.abs(frozen_gaussian_1d(x, q[k], p[k], gamma, hbar)) ** 2, color="steelblue", ls="--", label="frozen")
        ax4.set_xlim(q[k] - 2.5, q[k] + 2.5)
        ax4.set_title(rf"Packet at t = 2, $\hbar = {hbar:g}$")
ax3.set_xlabel("time t")
ax3.set_ylabel("fidelity with exact state")
ax3.set_ylim(0, 1.05)
ax3.set_title("Morse well: thawed converges as $\\hbar\\to0$")
ax3.legend(fontsize=7, ncol=2)
ax4.set_xlabel("x")
ax4.set_ylabel(r"$|\psi|^2$")
ax4.legend(fontsize=8)
fig2.tight_layout()

plt.show()

# %%
# Check
# -----
# In a harmonic well the thawed Gaussian is exact (and the frozen one is not);
# in the Morse well it converges to the exact state as hbar -> 0.
assert min(f_tg_harmonic) > 1 - 1e-6 and min(f_fr_harmonic) < 0.5
thawed = [fidelity_at_2[h][0] for h in hbars]
assert np.all(np.diff(thawed) > 0) and thawed[-1] > 0.9
assert all(fidelity_at_2[h][0] > fidelity_at_2[h][1] for h in hbars)
