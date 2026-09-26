r"""
The KAM theorem: which invariant tori survive a perturbation
================================================================

Kolmogorov (1954), Arnold (1963) and Moser (1962) proved that when an
integrable Hamiltonian system is perturbed slightly, *most* of its
invariant tori survive, merely deformed. The ones that break first are
the resonant tori, whose frequency ratio is rational; the last to go are
those with the "most irrational" ratio, poorly approximated by
fractions, the golden mean above all.

The cleanest place to watch this is the Chirikov-Taylor standard map,
:class:`~physicskit.chaos.systems.maps.StandardMap`, the surface of
section of a periodically kicked rotor:

.. math::

    p' = p + K\sin\theta, \qquad \theta' = \theta + p' .

At :math:`K=0` every line :math:`p=\text{const}` is an invariant torus
with rotation number :math:`\nu=p/2\pi`. This example turns :math:`K` on
and follows which tori remain.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.chaos.systems.maps import StandardMap

GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0  # 0.618..., the most irrational number

# %%
# Tori deform, resonances break up
# ------------------------------------
# Orbits launched on a grid of momenta, for increasing kick strength. At
# small :math:`K` almost every orbit still traces a curve that wraps all
# the way around in :math:`\theta`: a surviving KAM torus. The rational
# tori at :math:`\nu=0,\ 1/2,\ 1/3,\ 2/3` are the first to go, replaced by
# chains of islands with thin chaotic layers. By :math:`K=1.2` the last
# wrapping curves are gone.
Ks = [0.3, 0.7, 0.97, 1.2]
fig1, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
for ax, K in zip(axes, Ks):
    smap = StandardMap(k=K)
    for p0 in np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False):
        for th0 in (0.0, np.pi):
            traj = smap.trajectory(np.array([th0, p0]), n_iter=600)
            ax.plot(traj[:, 0], traj[:, 1], ",", color=plt.cm.twilight(p0 / (2 * np.pi)))
    ax.axhline(2 * np.pi * GOLDEN, color="k", lw=0.6, ls="--")
    ax.set_title(f"K = {K}")
    ax.set_xlabel(r"$\theta$")
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 2 * np.pi)
axes[0].set_ylabel("p")
fig1.suptitle(r"Standard map: KAM tori (dashed: golden-mean torus at $K=0$)")
fig1.tight_layout()

# %%
# The golden torus is the last barrier
# ----------------------------------------
# A rotational KAM torus is a wall: no orbit can cross it. Start an orbit
# in the chaotic layer at the :math:`\nu=0` resonance and record the
# largest distance it ever wanders from :math:`p=0`. As long as golden-mean
# tori survive, that distance stays below them; past Greene's critical
# value :math:`K_c\approx0.9716` the last wall is gone and the orbit
# reaches the :math:`\nu=1/2` resonance at :math:`p=\pi`.
K_values = np.linspace(0.5, 1.3, 33)
reach = []
for K in K_values:
    traj = StandardMap(k=K).trajectory(np.array([1e-3, 0.0]), n_iter=200_000)
    p_centred = (traj[:, 1] + np.pi) % (2 * np.pi) - np.pi
    reach.append(np.max(np.abs(p_centred)))
reach = np.array(reach)

for K, r in zip(K_values[::4], reach[::4]):
    print(f"K = {K:.3f}: max |p| reached = {r:.3f}   ({'trapped' if r < 0.9 * np.pi else 'escaped to p = pi'})")

fig2, ax2 = plt.subplots(figsize=(6, 3.8))
ax2.plot(K_values, reach, "o-", color="steelblue", ms=4)
ax2.axvline(0.9716, color="firebrick", ls="--", label=r"$K_c\approx0.9716$ (golden torus breaks)")
ax2.axhline(np.pi, color="0.6", lw=0.8)
ax2.set_xlabel("kick strength K")
ax2.set_ylabel(r"max $|p|$ reached from $p\approx0$")
ax2.set_title("Crossing the last KAM torus")
ax2.legend(fontsize=8)
fig2.tight_layout()

plt.show()
