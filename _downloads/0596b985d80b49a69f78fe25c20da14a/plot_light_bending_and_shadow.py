"""
Light bending, the photon sphere, and the black hole shadow
==================================================================

Starlight grazing the Sun is deflected by :math:`\\delta\\phi \\approx 4M/b`
-- twice the (wrong) Newtonian prediction -- confirmed by Eddington's 1919
solar eclipse expedition and the observation that made Einstein a household
name overnight. Push the impact parameter down toward the critical value
:math:`b_c = 3\\sqrt{3}M` and deflection diverges: photons can orbit
(unstably) forever at the photon sphere :math:`r=3M`, and anything with
:math:`b < b_c` is captured. This example traces individual light rays at a
range of impact parameters, then renders the full 2D shadow silhouette by
ray-tracing an entire camera image -- the calculation behind the 2019 Event
Horizon Telescope image of M87*.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.relativity.chapters.schwarzschild import SchwarzschildBlackHole
from physicskit.relativity.visualizers.shadow_render import plot_black_hole_shadow, render_black_hole_image

# %%
# Individual light rays at a range of impact parameters
# ------------------------------------------------------------
bh = SchwarzschildBlackHole(M=1.0)
fig, ax = plt.subplots(figsize=(6, 6))
for b in [4.0, 5.0, bh.critical_impact_parameter, 6.0, 10.0, 20.0]:
    y0 = bh.null_geodesic_initial_state(r0=200.0, impact_parameter=b, ingoing=True)
    traj = bh.integrate_geodesic(y0, dtau=0.05, n_steps=20000)
    x = traj["r"] * np.cos(traj["phi"])
    y = traj["r"] * np.sin(traj["phi"])
    label = f"b={b:.2f}M" + (" (critical)" if np.isclose(b, bh.critical_impact_parameter) else "")
    ax.plot(x, y, linewidth=1, label=label)
circle = plt.Circle((0, 0), bh.horizon_radius, color="black", zorder=5)
ax.add_patch(circle)
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.set_aspect("equal")
ax.set_xlabel("x [M]")
ax.set_ylabel("y [M]")
ax.set_title("Photon trajectories near a Schwarzschild black hole")
ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()

# %%
# Light deflection versus the weak-field 4M/b formula
# ------------------------------------------------------------
impact_params = np.linspace(10.0, 100.0, 12)
measured = []
for b in impact_params:
    y0 = bh.null_geodesic_initial_state(r0=2.0e5, impact_parameter=b, ingoing=True)
    traj = bh.integrate_geodesic(y0, dtau=2.0, n_steps=300000)
    measured.append((traj["phi"][-1] - traj["phi"][0]) - np.pi)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(impact_params, measured, "o", label="exact (geodesic integration)")
ax.plot(impact_params, bh.light_deflection_angle(impact_params), "--", label="weak field, $4M/b$")
ax.set_xlabel("impact parameter b [M]")
ax.set_ylabel("deflection angle [rad]")
ax.set_title("Gravitational light deflection")
ax.legend()
plt.tight_layout()

# %%
# The black hole shadow: ray-tracing a full camera image
# ------------------------------------------------------------
result = render_black_hole_image(M=1.0, ny=250, nx=250, inclination=1.3)
fig, ax = plt.subplots(figsize=(7, 7))
plot_black_hole_shadow(result, ax=ax)
plt.tight_layout()
plt.show()
