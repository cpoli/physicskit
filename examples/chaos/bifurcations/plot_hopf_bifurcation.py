r"""
The Hopf bifurcation: a limit cycle born from a fixed point
============================================================

When a pair of complex eigenvalues :math:`\mu \pm i\omega` of a fixed point
crosses the imaginary axis, the fixed point turns from a stable into an
unstable spiral and a limit cycle appears (Andronov 1929, Hopf 1942). This
is how oscillations start in systems as different as chemical reactions,
lasers, heart cells and flow past a cylinder, and it is a different route
from the period-doubling cascade: there is no sequence of bifurcations, just
one cycle whose amplitude grows from zero. Near onset every such system
reduces to the normal form

.. math::

    \dot r = \mu r + a r^3 + c r^5, \qquad \dot\phi = \omega.

With :math:`a < 0` (supercritical) the cycle radius is
:math:`\sqrt{-\mu/a}`; with :math:`a > 0, c < 0` (subcritical) the
oscillation appears suddenly at finite amplitude and persists below onset,
with hysteresis. The last panel finds the same :math:`\sqrt{\mu}` law in the
Brusselator, whose Hopf point is :math:`b_c = 1 + a^2`.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.chaos.systems.bifurcations import Brusselator, HopfNormalForm

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# %%
# Phase portraits below and above onset
# -------------------------------------
for mu, color in [(-0.1, "steelblue"), (0.1, "crimson")]:
    system = HopfNormalForm(mu=mu)
    for x0 in (0.02, 0.6):
        _, s = system.trajectory(state0=np.array([x0, 0.0]), n_steps=6000)
        axes[0].plot(s[:, 0], s[:, 1], color=color, lw=0.8, label=rf"$\mu = {mu}$" if x0 == 0.6 else None)
axes[0].set_aspect("equal")
axes[0].set_title("Normal form, supercritical")
axes[0].legend(loc="upper right", fontsize=8)

# %%
# Supercritical: amplitude grows as the square root of mu
# -------------------------------------------------------
mus = np.linspace(-0.2, 0.3, 26)
r_super = []
for mu in mus:
    _, s = HopfNormalForm(mu=mu).trajectory(state0=np.array([0.5, 0.0]), n_steps=60000)
    r_super.append(np.hypot(*s[-1]))
r_super = np.array(r_super)
mu_f = np.linspace(0.0, 0.3, 100)
axes[1].plot(mus, r_super, "o", color="crimson", label="simulated")
axes[1].plot(mu_f, np.sqrt(mu_f), "k-", label=r"$\sqrt{\mu}$")
axes[1].set_xlabel(r"$\mu$")
axes[1].set_ylabel("cycle radius")
axes[1].set_title("Supercritical ($a = -1$)")
axes[1].legend()

# %%
# Subcritical: a jump and hysteresis
# ----------------------------------
# Sweep mu up and then down, starting each run from where the last one ended.
mu_sweep = np.linspace(-0.4, 0.15, 23)
state = np.array([1e-3, 0.0])
r_up, r_down = [], []
for mu in mu_sweep:
    _, s = HopfNormalForm(mu=mu, a=1.0, c=-1.0).trajectory(state0=state, n_steps=80000)
    state = s[-1] if np.hypot(*s[-1]) > 1e-3 else np.array([1e-3, 0.0])
    r_up.append(np.hypot(*s[-1]))
for mu in mu_sweep[::-1]:
    _, s = HopfNormalForm(mu=mu, a=1.0, c=-1.0).trajectory(state0=state, n_steps=80000)
    state = s[-1] if np.hypot(*s[-1]) > 1e-3 else np.array([1e-3, 0.0])
    r_down.append(np.hypot(*s[-1]))
r_down = np.array(r_down[::-1])
r_up = np.array(r_up)
mu_b = np.linspace(-0.25, 0.15, 300)
rho_stable = (1 + np.sqrt(1 + 4 * mu_b)) / 2
rho_unstable = (1 - np.sqrt(1 + 4 * mu_b)) / 2
axes[2].plot(mu_b, np.sqrt(rho_stable), "k-", lw=1, label="stable cycle")
axes[2].plot(mu_b[mu_b < 0], np.sqrt(rho_unstable[mu_b < 0]), "k--", lw=1, label="unstable cycle")
axes[2].plot(mu_sweep, r_up, ">", color="crimson", label=r"$\mu$ increasing")
axes[2].plot(mu_sweep, r_down, "<", color="steelblue", label=r"$\mu$ decreasing")
axes[2].set_xlabel(r"$\mu$")
axes[2].set_title("Subcritical ($a = 1, c = -1$)")
axes[2].legend(fontsize=7)

# %%
# The Brusselator: a chemical Hopf bifurcation
# --------------------------------------------
# Linear fit of the squared amplitude against b gives the onset, compared
# with b_c = 1 + a^2 from the Jacobian.
a = 1.0
b_values = np.linspace(2.02, 2.2, 10)
amp = []
for b in b_values:
    _, s = Brusselator(a=a, b=b).trajectory(dt=0.01, n_steps=150000)
    x = s[-30000:, 0]
    amp.append((x.max() - x.min()) / 2)
amp = np.array(amp)
slope, intercept = np.polyfit(b_values, amp**2, 1)
b_c_fit = -intercept / slope
b_c = Brusselator(a=a).hopf_point
axes[3].plot(b_values, amp**2, "o", color="crimson", label="simulated")
axes[3].plot([b_c_fit, b_values[-1]], slope * np.array([b_c_fit, b_values[-1]]) + intercept, "k-", lw=1)
axes[3].axvline(b_c, color="gray", ls="--", label=rf"$b_c = 1 + a^2 = {b_c}$")
axes[3].set_xlabel("b")
axes[3].set_ylabel("amplitude$^2$ of x")
axes[3].set_title(f"Brusselator, fitted onset b = {b_c_fit:.3f}")
axes[3].legend(fontsize=8)
plt.tight_layout()
plt.show()
print(f"Brusselator: fitted b_c = {b_c_fit:.4f}, exact {b_c}")

# %%
# Check
# -----
np.testing.assert_allclose(r_super[mus > 0.02], np.sqrt(mus[mus > 0.02]), rtol=1e-3)
assert np.all(r_super[mus < -0.02] < 1e-3)
hysteresis = (mu_sweep > -0.2) & (mu_sweep < -0.02)
assert np.all(r_up[hysteresis] < 1e-2) and np.all(r_down[hysteresis] > 0.8)
assert np.all(r_up[mu_sweep > 0.0] > 0.9)  # the jump onto the large cycle
assert abs(b_c_fit - b_c) < 0.02
