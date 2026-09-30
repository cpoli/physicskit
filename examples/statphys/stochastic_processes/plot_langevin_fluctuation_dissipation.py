r"""
The Langevin equation: friction and noise must balance
=========================================================

Langevin's 1908 equation for a Brownian particle,

.. math::

    m\,dv = (F(x) - \gamma v)\,dt + \sqrt{2\gamma k_BT}\,dW,

contains a drag and a random kick with the same :math:`\gamma` in both.
This is not a modeling convenience. It is the fluctuation-dissipation
balance: the drag drains energy at a rate proportional to :math:`\gamma`,
the noise pumps it in at a rate proportional to its variance, and they
cancel only at :math:`\langle v^2\rangle = k_BT/m`.

This example integrates the equation for particles in a harmonic trap
with :class:`~physicskit.statphys.LangevinDynamics`, which uses the BAOAB
splitting (:func:`~physicskit.integrators.baoab_integrate`) so that even
weak friction does not let the time step heat the particles. The same temperature
comes out for frictions spanning two decades, and every quadratic degree
of freedom carries :math:`k_BT/2` (equipartition). It then breaks the
balance by integrating the equation with noise that is too weak, using the
shared :func:`~physicskit.integrators.euler_maruyama_integrate` directly,
and the particles settle at the wrong temperature.
"""

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from physicskit.integrators import euler_maruyama_integrate, wiener_increments
from physicskit.statphys import LangevinDynamics

m, k, kT = 1.0, 4.0, 0.5

# %%
# One temperature, any friction
# -----------------------------
# Particles start at rest at the trap's edge. Light friction lets them ring
# for many periods and heavy friction makes them creep, yet all settle at
# :math:`m\langle v^2\rangle = k\langle x^2\rangle = k_BT`. We average
# over :math:`t > 60`, many relaxation times :math:`m/\gamma` even for the
# lightest friction.

results = {}
for gamma in (0.1, 1.0, 10.0):
    ld = LangevinDynamics(n_particles=4000, mass=m, gamma=gamma, kT=kT, stiffness=k, seed=int(10 * gamma))
    t, x, v = ld.run(t_max=80.0, dt=0.002, n_frames=400, x0=np.full(4000, 1.0))
    results[gamma] = (t, x[:, :, 0], v[:, :, 0])
    late = t > 60
    print(f"gamma = {gamma:5.1f}: m<v^2> = {m * np.mean(v[late] ** 2):.3f}, k<x^2> = {k * np.mean(x[late] ** 2):.3f}   (kT = {kT})")

# %%
# Breaking the balance
# --------------------
# Keep the friction at :math:`\gamma = 1` but give the kicks only half the
# fluctuation-dissipation variance, :math:`\gamma k_BT` instead of
# :math:`2\gamma k_BT`. The particles cool to half the bath temperature.


@njit
def trap_drift(s, t, p):
    n = s.shape[0] // 2
    out = np.empty_like(s)
    out[:n] = s[n:]
    out[n:] = (-p[0] * s[:n] - p[1] * s[n:]) / p[2]
    return out


@njit
def weak_noise(s, t, p):
    n = s.shape[0] // 2
    out = np.zeros_like(s)
    out[n:] = np.sqrt(p[3]) / p[2]
    return out


n, dt, chunk = 4000, 0.002, 100
gamma = 1.0
weak_params = np.array([k, gamma, m, gamma * kT])
rng = np.random.default_rng(1908)
state = np.concatenate([np.full(n, 1.0), np.zeros(n)])
t_w, s_w = [0.0], [state]
for i in range(400):  # 400 chunks of 100 steps: t = 0 .. 80
    dW = wiener_increments(chunk, 2 * n, dt, rng=rng)
    _, traj = euler_maruyama_integrate(trap_drift, weak_noise, state, i * chunk * dt, dt, dW, weak_params, chunk)
    state = traj[-1]
    t_w.append((i + 1) * chunk * dt)
    s_w.append(state)
t_w, s_w = np.array(t_w), np.array(s_w)
T_weak = m * np.mean(s_w[t_w > 60, n:] ** 2)
print(f"half-strength noise: m<v^2> = {T_weak:.3f}, not kT = {kT}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
colors = {0.1: "tab:blue", 1.0: "tab:orange", 10.0: "tab:green"}
for gamma, (t, x, v) in results.items():
    axes[0].plot(t, x[:, 0], lw=0.7, color=colors[gamma], label=rf"$\gamma = {gamma}$")
    axes[1].plot(t, m * np.mean(v**2, axis=1), color=colors[gamma], label=rf"$\gamma = {gamma}$")
axes[1].plot(t_w, m * np.mean(s_w[:, n:] ** 2, axis=1), color="gray", label="half-strength noise")
axes[1].axhline(kT, color="k", ls="--", lw=1)
axes[1].axhline(kT / 2, color="gray", ls=":", lw=1)
axes[0].set_xlabel("t")
axes[0].set_ylabel("x")
axes[0].set_title("one particle per friction")
axes[0].legend(fontsize=8)
axes[1].set_xlabel("t")
axes[1].set_ylabel(r"$m\langle v^2\rangle$")
axes[1].set_title("kinetic temperature")
axes[1].legend(fontsize=8)

t, x, v = results[1.0]
late = t > 60
xs = np.linspace(-1.5, 1.5, 200)
axes[2].hist(x[late].ravel(), bins=80, density=True, alpha=0.6, label="positions, t > 60")
axes[2].plot(xs, np.sqrt(k / (2 * np.pi * kT)) * np.exp(-k * xs**2 / (2 * kT)), "k--", label=r"Boltzmann $e^{-kx^2/2k_BT}$")
axes[2].set_xlabel("x")
axes[2].set_title(r"equilibrium in the trap ($\gamma = 1$)")
axes[2].legend(fontsize=8)
fig.tight_layout()
plt.show()
