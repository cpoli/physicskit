r"""
Graded-index fiber guiding (Kao's low-loss optical fiber)
===============================================================

Charles Kao and George Hockham's insight that glass fiber could in
principle be purified to carry light over kilometers launched the
fiber-optic communications industry. The graded-index (GRIN) fiber profile
that followed guides light by continuously bending rays back toward the
core axis, rather than by a single sharp core-cladding interface: for the
standard quadratic radial index profile :math:`n(r) = n_0(1 - n_2 r^2/2)`,
paraxial rays oscillate sinusoidally about the axis instead of walking off
it. :func:`~physicskit.optics.ray.grin_medium` builds the ABCD matrix of
exactly this profile; chaining it through
:class:`~physicskit.optics.ray.OpticalSystem` and calling
:meth:`~physicskit.optics.ray.OpticalSystem.trace_ray` reproduces the
self-confining ray trajectory below, contrasted against the same ray
walking away from the axis in a homogeneous medium.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.optics.ray import OpticalElement, OpticalSystem, free_space, grin_medium

# %%
# A ray launched off-axis, traced through many thin GRIN slices
# -------------------------------------------------------------------

n0, n2, length = 1.46, 1.0, 20.0  # fiber core index, gradient coefficient, length (mm)
n_segments = 200
dz = length / n_segments

fiber = OpticalSystem([OpticalElement(grin_medium(n0, n2, dz), length=dz) for _ in range(n_segments)])
homogeneous = OpticalSystem([OpticalElement(free_space(dz), length=dz) for _ in range(n_segments)])

y0, theta0 = 0.05, 0.02
trace_fiber = fiber.trace_ray(y0=y0, theta0=theta0)
y_fiber = trace_fiber[:, 0]
y_free = homogeneous.trace_ray(y0=y0, theta0=theta0)[:, 0]
z = np.linspace(0, length, n_segments + 1)

# %%
# The GRIN fiber's ray oscillates sinusoidally about the axis with period
# :math:`2\pi/\sqrt{n_2}` and never escapes; the same ray in a homogeneous
# medium of the same length just walks linearly off axis.

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(z, y_fiber, label="GRIN fiber: ray confined, oscillates about axis")
ax.plot(z, y_free, "--", label="homogeneous medium: ray walks off axis")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("z (mm)")
ax.set_ylabel("ray height y (mm)")
ax.legend()
fig.tight_layout()

period = 2 * np.pi / np.sqrt(n2)
print(f"fiber length: {length} mm, predicted oscillation period: {period:.4f} mm")
print(f"max |y| in GRIN fiber over {length} mm:   {np.max(np.abs(y_fiber)):.4f} mm")
print(f"max |y| in homogeneous medium over {length} mm: {np.max(np.abs(y_free)):.4f} mm")
print("the fiber ray stays bounded near its starting height; the free-space")
print("ray diverges linearly, exactly as GRIN guiding predicts.")

# %%
# The (y, theta) phase-space portrait: closed orbits, the SHM analogue
# --------------------------------------------------------------------------
# :meth:`~physicskit.optics.ray.OpticalSystem.trace_ray` already returns
# both the ray height *and* its angle at every plane; the angle column was
# unused above. Since the GRIN profile drives simple-harmonic-oscillator-
# like ray motion, plotting :math:`\theta` against :math:`y` at every plane
# -- the ray's phase-space trajectory -- traces a closed ellipse for each
# launch condition inside the fiber (an invariant of the SHM-like guiding),
# while the same construction for a homogeneous medium just produces an
# open, unbounded line, since there ``theta`` never changes at all while
# ``y`` runs away.

fig2, ax2 = plt.subplots(figsize=(5, 4.5))
for y0_i, theta0_i in [(0.05, 0.02), (0.03, -0.03), (0.02, 0.04), (-0.04, -0.01)]:
    trace_i = fiber.trace_ray(y0=y0_i, theta0=theta0_i)
    ax2.plot(trace_i[:, 0], trace_i[:, 1], lw=1.2, label=f"y0={y0_i}, theta0={theta0_i}")
ax2.axhline(0, color="k", lw=0.4)
ax2.axvline(0, color="k", lw=0.4)
ax2.set_xlabel("ray height y (mm)")
ax2.set_ylabel("ray angle theta (rad)")
ax2.legend(fontsize=7)
ax2.set_title("GRIN fiber phase-space portrait: closed elliptical orbits")
ax2.set_aspect("auto")
fig2.tight_layout()
