r"""
Einstein's Brownian motion: diffusion equals mobility times temperature
=========================================================================

Einstein's 1905 paper made two separate claims about a suspended particle.
First, its random wandering spreads as :math:`\langle\Delta x^2\rangle = 2Dt`.
Second, the diffusion coefficient is not a free parameter: it equals the
particle's mobility :math:`\mu`, the drift speed per unit applied force,
times the temperature,

.. math::

    D = \mu\,k_BT.

Einstein got this by balancing the diffusion current against the drift
caused by a force, such as gravity acting on sedimenting particles. The
relation ties a fluctuation (the spread) to a response (the drift). Perrin
used it with Stokes' :math:`\mu = 1/6\pi\eta a` to count molecules.

This example simulates overdamped Brownian particles with
:class:`~physicskit.statphys.BrownianMotion`. It measures :math:`D` from
the spread and :math:`\mu` from the drift under a pull, for several
frictions and temperatures, and checks that :math:`D/(\mu k_BT) = 1` in
every case. Units have :math:`k_B = 1`.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.statphys import BrownianMotion, stokes_einstein_diffusion_coefficient

# %%
# Spread and drift of one driven ensemble
# ---------------------------------------
# A constant force :math:`F` moves the cloud's center at :math:`\mu F`. The
# cloud's width still grows as :math:`2Dt`, exactly as without the force.

gamma, kT, F = 2.0, 1.0, 1.5
bm = BrownianMotion(n_particles=4000, dim=1, gamma=gamma, kT=kT, force=F, seed=1905)
t, x = bm.run(t_max=10.0, dt=0.01, n_frames=100)
center = x[:, :, 0].mean(axis=1)
spread = x[:, :, 0].var(axis=1)
print(f"mobility: measured {bm.measured_mobility():.4f}, 1/gamma = {1 / gamma:.4f}")
print(f"diffusion: measured {bm.measured_diffusion_coefficient():.4f}, kT/gamma = {kT / gamma:.4f}")

# %%
# The Einstein relation across frictions and temperatures
# -------------------------------------------------------
# Changing the friction changes :math:`D` and :math:`\mu` together. Changing
# the temperature changes only :math:`D`. Their ratio is always
# :math:`k_BT`.

rows = []
for g in (0.5, 1.0, 2.0, 4.0):
    for T in (0.5, 1.0, 2.0):
        run = BrownianMotion(n_particles=3000, gamma=g, kT=T, force=1.0, seed=int(100 * g + 10 * T))
        run.run(t_max=5.0, dt=0.01, n_frames=25)
        rows.append((g, T, run.measured_diffusion_coefficient(), run.measured_mobility()))
rows = np.array(rows)
ratio = rows[:, 2] / (rows[:, 3] * rows[:, 1])
print(f"D / (mu kT) over {len(rows)} runs: mean {ratio.mean():.3f}, spread {ratio.std():.3f}")

# %%
# Perrin's numbers
# ----------------
# For a 0.5 micron bead in water at 20 C, the Stokes-Einstein coefficient is
# a fraction of a square micron per second, so it takes seconds to wander a
# micron: exactly the scale Perrin followed under his microscope.

D_bead = stokes_einstein_diffusion_coefficient(1.380649e-23 * 293.15, 1.0e-3, 0.5e-6)
print(f"Stokes-Einstein D for a 0.5 um bead in water: {D_bead * 1e12:.3f} um^2/s")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
for i in range(12):
    axes[0].plot(t, x[:, i, 0], lw=0.7)
axes[0].plot(t, F / gamma * t, "k--", lw=1.5, label=r"drift $\mu F t$")
axes[0].set_xlabel("t")
axes[0].set_ylabel("x")
axes[0].set_title("driven Brownian particles")
axes[0].legend()

axes[1].plot(t, spread, color="steelblue", label=r"Var$[x(t)]$")
axes[1].plot(t, 2 * kT / gamma * t, "k--", lw=1, label=r"$2Dt$, $D = k_BT/\gamma$")
axes[1].plot(t, center - F / gamma * t, color="firebrick", label=r"$\langle x\rangle - \mu F t$")
axes[1].set_xlabel("t")
axes[1].set_title("spread grows as 2Dt; the drift is separate")
axes[1].legend()

sc = axes[2].scatter(rows[:, 3] * rows[:, 1], rows[:, 2], c=rows[:, 1], cmap="viridis", s=50, edgecolors="k")
line = np.linspace(0, (rows[:, 3] * rows[:, 1]).max() * 1.05, 10)
axes[2].plot(line, line, "k--", lw=1, label=r"$D = \mu k_BT$")
axes[2].set_xlabel(r"$\mu\,k_BT$ (mobility from drift)")
axes[2].set_ylabel(r"$D$ (from spread)")
axes[2].set_title("Einstein relation, 12 runs")
fig.colorbar(sc, ax=axes[2], label=r"$k_BT$")
axes[2].legend()
fig.tight_layout()
plt.show()

# %%
# Check
# -----
# mu = 1/gamma, D = kT/gamma, so D = mu kT for every friction and
# temperature (Einstein 1905); Stokes-Einstein for a 0.5 um bead in water.
assert abs(bm.measured_mobility() * gamma - 1) < 0.03
assert abs(bm.measured_diffusion_coefficient() / (kT / gamma) - 1) < 0.03
assert abs(ratio.mean() - 1) < 0.03 and ratio.std() < 0.05
assert abs(D_bead - 1.380649e-23 * 293.15 / (6 * np.pi * 1.0e-3 * 0.5e-6)) < 1e-18
