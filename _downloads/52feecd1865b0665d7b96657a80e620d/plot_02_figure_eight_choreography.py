r"""
The figure-eight three-body choreography
============================================

For two centuries after Newton, every known periodic three-body solution
demanded either a special mass ratio (Lagrange's equilateral triangle)
or a rigidly symmetric configuration (Euler's collinear solutions).
Moore found something qualitatively different in 1993: three *equal*
masses chasing one another endlessly around a single figure-eight
curve, a genuine *choreography* with no distinguished body -- proven to
exist by Chenciner and Montgomery in 2000.
:func:`~physicskit.astro.nbody.figure_eight_initial_conditions` returns
exactly this initial condition; this example integrates it with
:class:`~physicskit.astro.nbody.NBodySystem`, checks that the
configuration is genuinely periodic by finding the time at which body 1
first returns closest to its own starting point, and checks that energy
and angular momentum stay conserved over one full period.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.astro.nbody import NBodySystem, figure_eight_initial_conditions
from physicskit.astro.visualizers import animate_nbody_trajectories, plot_nbody_trajectories

# %%
# Integrate one and a bit periods
# -------------------------------------
positions0, velocities0, masses = figure_eight_initial_conditions()

dt, n_steps = 0.002, 3500  # generously more than one period, found below
system = NBodySystem(positions0, velocities0, masses)
E0 = system.total_energy()
L0 = system.total_angular_momentum()
history = system.simulate(dt, n_steps)
E1 = system.total_energy()
L1 = system.total_angular_momentum()

print(f"energy:            E0={E0:.8f}, E1={E1:.8f}, relative drift={abs((E1 - E0) / E0):.2e}")
print(f"angular momentum:  L0_z={L0[2]:.8f}, L1_z={L1[2]:.8f}")

# %%
# Finding the period numerically
# ------------------------------------
# Rather than trust a hardcoded literature value, find the period
# directly: track body 1's distance back to its own initial position
# after a short initial exclusion window, and take the time of its
# first sharp local minimum.
t = np.arange(n_steps + 1) * dt
dist_to_start = np.linalg.norm(history[:, 0, :] - positions0[0], axis=1)
search = t > 1.0  # skip the initial departure from the starting point
i_period = np.argmin(dist_to_start[search]) + np.argmax(search)
T_period = t[i_period]
print(f"\nnumerically found period: T = {T_period:.6f} (return distance = {dist_to_start[i_period]:.2e})")

fig0, ax0 = plt.subplots(figsize=(6, 3.5))
ax0.plot(t, dist_to_start)
ax0.axvline(T_period, color="firebrick", ls="--", label=f"period T={T_period:.4f}")
ax0.set_xlabel("t")
ax0.set_ylabel("|body 1 - its own start|")
ax0.set_title("Finding the choreography's period")
ax0.legend()
fig0.tight_layout()

# %%
# The choreography itself
# -----------------------------
# All three equal masses trace the *same* figure-eight curve, each
# lagging the next by exactly a third of the period -- no body plays a
# distinguished role.
n_steps_one_period = int(round(T_period / dt))
fig1, ax1 = plot_nbody_trajectories(history[: n_steps_one_period + 1])
ax1.set_title("The figure-eight choreography, one full period")
fig1.tight_layout()

anim = animate_nbody_trajectories(history[: n_steps_one_period + 1], trail=n_steps_one_period, title="Figure-eight choreography")

plt.show()
