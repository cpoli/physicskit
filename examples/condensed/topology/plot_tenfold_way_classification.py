r"""
The Tenfold Way: Three Topological Invariants, One Classification Scheme
================================================================================

Schnyder, Ryu, Furusaki, and Ludwig, and independently Kitaev, showed that
every noninteracting Bloch or Bogoliubov-de Gennes Hamiltonian falls into
one of just ten symmetry classes, fixed by whether time-reversal and
particle-hole symmetry are present, the sign each squares to, and whether
their product enforces a chiral symmetry -- and that the allowed
topological invariant (:math:`\mathbb{Z}`, :math:`\mathbb{Z}_2`, or none)
in a given spatial dimension depends *only* on which of the ten classes a
Hamiltonian belongs to. Three of this package's own topological models,
already built for their own separate breakthroughs, are three different
rows of that table:

- :func:`~physicskit.condensed.models.haldane_model` has neither
  time-reversal nor particle-hole symmetry (class A) -- its invariant is
  the integer Chern number computed by
  :func:`~physicskit.condensed.topology.compute_chern_number`.
- :func:`~physicskit.condensed.models.kane_mele_hamiltonian` has
  time-reversal symmetry squaring to :math:`-1` (class AII) -- its
  invariant is the :math:`\mathbb{Z}_2` index computed by
  :func:`~physicskit.condensed.topology.z2_invariant`.
- :func:`~physicskit.condensed.models.kitaev_chain_bdg_real_space` has only
  particle-hole symmetry, built into the BdG formalism itself (class D) --
  its invariant is a :math:`\mathbb{Z}_2` Majorana number, diagnosed here
  by whether a near-zero-energy end mode appears.

Three different symmetry classes, three different invariant types, and
three completely different-looking pieces of code -- unified as three
entries in one ten-row table.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.condensed.models import haldane_model, kane_mele_hamiltonian, kitaev_chain_bdg_real_space
from physicskit.condensed.topology import compute_chern_number, z2_invariant

# %%
# Class A: no symmetry constraint -- a Z-valued Chern number
# -----------------------------------------------------------------
# Haldane's model breaks both time-reversal (via the complex second-neighbor
# hopping) and any particle-hole-like constraint; nothing protects the
# invariant from taking any integer value, and it does: +-1 in the
# topological regime, 0 once the sublattice mass term dominates.

M_values = np.linspace(-2.0, 2.0, 9)
chern = [compute_chern_number(lambda k1, k2, M=M: haldane_model(k1, k2, t=1.0, t2=0.2, phi=np.pi / 2, M=M), grid_size=20)[0] for M in M_values]

# %%
# Class AII: time-reversal with T^2 = -1 -- a Z2 invariant
# -----------------------------------------------------------------
# Kane-Mele's model is exactly Haldane's model doubled into two
# time-reversed spin copies; the extra symmetry collapses the integer
# Chern number of each spin sector down to a two-valued (trivial/
# topological) invariant -- any nonzero intrinsic spin-orbit coupling
# ``lambda_so`` already opens the topological gap.

lambda_so_values = [0.0, 0.06]
z2 = [z2_invariant(lambda k1, k2, lam=lam: kane_mele_hamiltonian(k1, k2, lambda_so=lam), grid_size=20) for lam in lambda_so_values]

# %%
# Class D: particle-hole symmetry alone -- a Z2 Majorana number
# ---------------------------------------------------------------------
# The Kitaev chain's Bogoliubov-de Gennes Hamiltonian has an *intrinsic*
# particle-hole symmetry (not an extra assumption but a redundancy built
# into the BdG doubling itself); with no other symmetry protecting it, the
# 1D invariant this class allows is again only Z2 -- trivial or topological,
# diagnosed here by whether an end-localized state pins to exactly zero
# energy.

mu_values = np.linspace(-4.0, 4.0, 17)
min_gap = [np.min(np.abs(np.linalg.eigvalsh(kitaev_chain_bdg_real_space(n_sites=60, mu=mu, t=1.0, delta=1.0)))) for mu in mu_values]

# %%
# Three rows of one table
# -----------------------------
print(f"{'class':<6}{'symmetry':<28}{'invariant':<12}{'model'}")
print(f"{'A':<6}{'none':<28}{'Z (Chern)':<12}Haldane model")
print(f"{'AII':<6}{'T^2 = -1':<28}{'Z2':<12}Kane-Mele model")
print(f"{'D':<6}{'particle-hole only':<28}{'Z2':<12}Kitaev chain (BdG)")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

axes[0].plot(M_values, chern, "o-", color="steelblue")
axes[0].axhline(0, color="black", lw=0.5)
axes[0].set_xlabel("M")
axes[0].set_ylabel("Chern number")
axes[0].set_title("Class A (Haldane): Z-valued")

axes[1].bar([str(lam) for lam in lambda_so_values], z2, color="firebrick")
axes[1].set_ylim(-0.2, 1.2)
axes[1].set_xlabel(r"$\lambda_{so}$")
axes[1].set_ylabel(r"$\mathbb{Z}_2$ invariant")
axes[1].set_title("Class AII (Kane-Mele): Z2-valued")

axes[2].semilogy(mu_values, np.maximum(min_gap, 1e-16), "o-", color="seagreen")
axes[2].axvline(-2.0, color="black", lw=0.5, ls="--")
axes[2].axvline(2.0, color="black", lw=0.5, ls="--")
axes[2].set_xlabel(r"$\mu$")
axes[2].set_ylabel("min |E| (end-mode gap)")
axes[2].set_title("Class D (Kitaev): Z2-valued")

fig.suptitle("Three symmetry classes, three invariant types, one classification scheme", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])

# %%
# The Kitaev chain's end-mode energy dips to numerical zero throughout
# :math:`|\mu| < 2t` (the topological region) and grows outside it, exactly
# tracking the same trivial/topological split the other two panels reach
# by entirely different routes -- one continuous parameter producing an
# unbounded integer, the other two producing only a two-valued flag,
# precisely as the tenfold way's classification predicts for their
# respective symmetry classes.

plt.show()
