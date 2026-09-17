r"""
Bhabha and Heitler's cascade theory of cosmic-ray showers
================================================================

Bhabha, Heitler, Carlson, and Oppenheimer (1937) explained the
cosmic-ray "transition curve" -- the number of shower particles first
growing, then shrinking, with depth -- as a branching cascade: a single
high-energy electron, positron, or photon alternately radiates a photon
or materializes one into an electron-positron pair, each product
repeating the process until individual energies fall low enough for
ordinary ionization losses to take over. This example builds exactly
this kind of branching cascade with
:func:`~physicskit.particle.collider.simple_shower`, checks that its
total four-momentum is conserved from the single primary particle down
to every final-state leaf, and shows the exponential growth in particle
count with generation that gives the transition curve its characteristic
rise.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.particle.collider import flatten_shower, simple_shower
from physicskit.particle.kinematics import invariant_mass
from physicskit.particle.visualizers import animate_particle_cascade

# %%
# A branching cascade from one 100 GeV primary
# ---------------------------------------------------
rng = np.random.default_rng(3)
root = simple_shower(100.0, E_threshold=2.0, rng=rng)
nodes = flatten_shower(root)
leaves = [n for n in nodes if n.is_leaf]

print(f"primary particle energy: {root.four_vector.E:.3f} GeV")
print(f"total nodes in the cascade tree: {len(nodes)}")
print(f"final-state (leaf) particles: {len(leaves)}")
print(f"max generation reached: {max(n.generation for n in nodes)}")

# %%
# Four-momentum conservation, from primary down to every leaf
# ------------------------------------------------------------------
# Every branching is an exact two-body decay reusing already-tested
# kinematics building blocks, so the invariant mass of the *entire*
# final state should reproduce the primary particle's own mass exactly,
# no matter how many generations deep the cascade went.
reconstructed_mass = invariant_mass([leaf.four_vector for leaf in leaves])
print(f"\nprimary particle's own (virtuality) mass: {root.four_vector.mass:.8f} GeV")
print(f"invariant mass reconstructed from ALL {len(leaves)} final-state leaves: {reconstructed_mass:.8f} GeV")
print(f"difference: {abs(reconstructed_mass - root.four_vector.mass):.2e} GeV (machine precision)")

# %%
# Particle count growing, then the cascade dying out
# --------------------------------------------------------
# Counting nodes by generation shows the multiplication Bhabha, Heitler,
# Carlson, and Oppenheimer derived: roughly doubling generation over
# generation while energy remains available, then stopping abruptly
# once every branch has fallen below the energy threshold.
max_gen = max(n.generation for n in nodes)
counts_by_generation = [sum(1 for n in nodes if n.generation == g) for g in range(max_gen + 1)]
print("\nparticles per generation (roughly doubling, then the cascade terminates):")
for g, c in enumerate(counts_by_generation):
    print(f"  generation {g}: {c} particles")

anim = animate_particle_cascade(root)

plt.show()
