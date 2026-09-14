r"""
Bateman's equations for radioactive decay chains
=====================================================

Real radioactive samples rarely decay in one step: uranium and thorium
each head long chains of successive daughters. Bateman (1910) solved the
resulting linear system in closed form, giving the population of the
:math:`k`-th species in a chain :math:`N_1\to N_2\to\cdots\to N_n` as an
explicit sum of exponentials in the decay constants
:math:`\lambda_1,\ldots,\lambda_n`. This example builds a three-species
chain with :func:`~physicskit.particle.decays.bateman_decay_chain`,
watches the parent deplete while an intermediate daughter first rises
and then falls (it has to be produced before it can decay away), and
checks total population conservation at every instant.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.particle.decays import bateman_decay_chain
from physicskit.particle.visualizers import animate_decay_chain_bars, plot_decay_chain

# %%
# A three-species chain: parent -> daughter -> stable granddaughter
# ------------------------------------------------------------------------
# Decay constants chosen so the parent decays slower than the
# intermediate daughter, giving the daughter's population a clear
# rise-then-fall ("transient equilibrium") shape; the final species is
# stable (decay constant exactly 0).
N0 = 1000.0
decay_constants = [0.15, 0.6, 0.0]
labels = ["Parent", "Daughter", "Stable granddaughter"]

t = np.linspace(0.0, 30.0, 300)
N = bateman_decay_chain(N0, decay_constants, t)

fig1, ax1 = plot_decay_chain(t, N, labels=labels)
ax1.set_title("A three-species decay chain")
fig1.tight_layout()

# %%
# Total population is conserved (nothing but transformation)
# -------------------------------------------------------------------
total = N.sum(axis=0)
print(f"N0 = {N0}")
print(f"total population N1+N2+N3 at t=0:  {total[0]:.6f}")
print(f"total population N1+N2+N3 at t=30: {total[-1]:.6f}  (should equal N0 -- decay only transforms species, it doesn't destroy atoms)")
print(f"max deviation from N0 over the whole run: {np.max(np.abs(total - N0)):.2e}")

# %%
# The daughter's rise-and-fall
# -----------------------------------
i_peak = np.argmax(N[1])
print(f"\ndaughter population peaks at t={t[i_peak]:.3f}, N2={N[1, i_peak]:.2f}")
print("(it must first be produced by the parent's decay before it can decay away itself --")
print(" a signature no single-species exponential can produce)")

anim = animate_decay_chain_bars(t, N, labels=labels)

plt.show()
