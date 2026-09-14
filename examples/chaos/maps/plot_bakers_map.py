r"""
Baker's Map: Stretch, Cut, and Stack
=======================================

The (generalized) baker's map cuts the unit square :math:`[0,1) \times
[0,1)` at :math:`x=\alpha`, stretches each piece horizontally back to unit
width (contracting it vertically to match), and stacks the two pieces:

.. math::

    T(x, y) = \begin{cases}
        (x / \alpha,\ \alpha y) & 0 \le x < \alpha \\
        ((x - \alpha) / (1 - \alpha),\ \alpha + (1 - \alpha) y) & \alpha \le x < 1
    \end{cases}

It is the textbook conceptual bridge of deterministic chaos: it makes this
"stretch, cut, and stack" mechanism -- the geometric operation behind
exponential sensitivity and topological mixing -- completely explicit and
exactly solvable. Unlike the Henon map, it is *area-preserving* (its
Jacobian determinant is exactly 1 everywhere), yet it is still uniformly
hyperbolic, ergodic, and mixing. The example below uses the classic
symmetric cut, :math:`\alpha=0.5`.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.chaos.systems.maps import BakersMap
from physicskit.chaos.visualizers import theme
from physicskit.chaos.visualizers.dynamic_plots import animate_bakers_map

system = BakersMap(alpha=0.5)

# %%
# Animation: stretch, cut, and stack, in motion
# ------------------------------------------------
# A dense, regular grid of square markers starts colored in just two bands --
# blue on the left half, red on the right -- about as simple an initial
# state as there is, and one that (for this map's default ``alpha = 0.5``)
# lines up exactly with the map's own cut point. Rather than blending
# straight to each iteration's end state, every iteration is animated as
# three distinct phases: the square visibly **stretches** out to twice its
# width (squashed vertically, stretched horizontally, spilling past the
# unit square's outline), pauses on the dashed **cut** line at ``x = 1``,
# then **stacks** as the right-hand piece slides back and up onto the
# left-hand piece -- landing the two original colors as clean top and
# bottom halves after the very first iteration. Because every square keeps
# its original color, the two halves are then seen getting sliced and
# interleaved into progressively thinner, more numerous fragments. After 20
# iterations the two colors are, to the eye, uniformly salt-and-peppered
# across the whole square: mixing, made literal.
anim = animate_bakers_map(system, n_points=10000, n_iterations=20, frames_per_iteration=15)

plt.show()

# %%
# To save the animation to a file instead of (or in addition to) displaying
# it interactively, use e.g.::
#
#     anim.save("bakers_map_animation.gif", writer="pillow", fps=20)

# %%
# The stripes get thinner, iteration by iteration
# ---------------------------------------------------
# The static counterpart of the animation above: render the unit square as a
# fine-grained raster of colored horizontal stripes, then trace each output
# pixel *backward* through the map (the baker's map's inverse is just as
# explicit as its forward form) to find which stripe it originally belonged
# to. This is far crisper than a scatter plot and makes the exponential
# refinement of the partition -- the hallmark of mixing -- directly visible:
# by 5 iterations, the coarse stripes have already become an interleaved weave
# too fine to resolve by eye.
alpha = system.alpha
resolution = 500
n_stripes = 8


def _inverse_step(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One backward step of the baker's map: undo stretch-cut-stack."""
    bottom = y < alpha
    x_new = np.where(bottom, alpha * x, alpha + (1.0 - alpha) * x)
    y_new = np.where(bottom, y / alpha, (y - alpha) / (1.0 - alpha))
    return x_new, y_new


centers = (np.arange(resolution) + 0.5) / resolution
grid_x, grid_y = np.meshgrid(centers, centers)

cmap = plt.get_cmap(theme.SEQUENTIAL_CMAP, n_stripes)
iterations_shown = [0, 1, 2, 3, 5]
fig, axes = plt.subplots(1, len(iterations_shown), figsize=(15, 3.4))

x, y = grid_x, grid_y
prev_n = 0
for ax, n in zip(axes, iterations_shown):
    for _ in range(n - prev_n):
        x, y = _inverse_step(x, y)
    prev_n = n
    stripe_idx = np.minimum((y * n_stripes).astype(int), n_stripes - 1)
    ax.imshow(stripe_idx, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=0, vmax=n_stripes - 1)
    ax.set_title(f"n = {n}")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("The initial horizontal stripes, refined by n iterations of the map")
fig.tight_layout()

# %%
# Mixing: many orbits fill the square uniformly
# -----------------------------------------------------------
# Because the map is area-preserving and ergodic, the ergodic theorem
# promises that a *single* orbit, run long enough, visits every region of the
# unit square with equal density -- there is no fractal attractor to see
# here (unlike the dissipative Henon map): the "attractor" is the whole
# square.
#
# In double-precision floating point, though, a single orbit of the classic
# ``alpha=0.5`` map is a poor way to actually demonstrate that: because each
# step is an *exact* bit-shift of the state's binary expansion (see the
# caveat in :class:`~physicskit.chaos.systems.maps.BakersMap`'s docstring), a
# double-precision orbit runs out of fresh bits and its ``x`` collapses to
# the exact fixed point ``0`` within roughly 50 iterations -- nowhere near
# enough to fill the square. A small ensemble of independent short orbits
# sidesteps this entirely while showing exactly the same mixing: even after
# just a few dozen iterations each, their combined points already spread
# densely and (up to small-sample noise) uniformly across the square.
rng = np.random.default_rng(3)
n_orbits, orbit_len = 80, 20
starts = rng.uniform(size=(n_orbits, 2))
orbits = [system.trajectory(s0, n_iter=orbit_len) for s0 in starts]
traj = np.concatenate(orbits, axis=0)
iter_index = np.tile(np.arange(orbit_len + 1), n_orbits)

fig2, ax2 = plt.subplots(figsize=(6.5, 6))
sc = ax2.scatter(traj[:, 0], traj[:, 1], c=iter_index, s=6, cmap=theme.SEQUENTIAL_CMAP, alpha=0.7)
ax2.set_aspect("equal")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title(f"{n_orbits} independent {orbit_len}-step orbits already fill the square")
fig2.colorbar(sc, ax=ax2, label="iteration n (within each orbit)")

# %%
# Exponential sensitivity, exactly
# -----------------------------------
# Because each branch of the map is exactly linear, the "stretch" step
# doubles the horizontal separation between two nearby points at *every
# single iteration* (for the symmetric :math:`\alpha=0.5` map) -- no
# averaging or fitting needed, unlike for smooth chaotic flows. This gives
# the baker's map its role as the ground-truth validation case for
# physicskit.chaos's numerical Lyapunov estimators:
# :meth:`~physicskit.chaos.systems.maps.BakersMap.lyapunov_exponents` returns
# the pair of *exact* analytic exponents :math:`\pm h`, with
#
# .. math::
#
#     h = -\alpha \ln\alpha - (1 - \alpha)\ln(1 - \alpha),
#
# the Shannon entropy of the two-symbol Bernoulli process that describes
# which branch a typical orbit lands in at each step (:math:`h = \ln 2` for
# the symmetric map).
delta0 = 1e-10
state_a = np.array([0.1, 0.2])
state_b = np.array([0.1 + delta0, 0.2])
separations = [delta0]
for _ in range(20):
    state_a = system.step(state_a)
    state_b = system.step(state_b)
    separations.append(abs(state_b[0] - state_a[0]))
separations = np.array(separations)

lam_plus, lam_minus = system.lyapunov_exponents()

fig3, ax3 = plt.subplots(figsize=(7, 5))
n = np.arange(len(separations))
ax3.semilogy(n, separations, "o-", color=theme.ACCENT, label="measured separation")
ax3.semilogy(n, delta0 * np.exp(lam_plus * n), "--", color=theme.MUTED, label=r"$\delta_0 e^{\lambda n}$")
ax3.set_xlabel("iteration n")
ax3.set_ylabel(r"$|x_a - x_b|$")
ax3.set_title(rf"Exact doubling every step: $\lambda = \ln 2 \approx {lam_plus:.4f}$")
ax3.legend()

plt.show()
