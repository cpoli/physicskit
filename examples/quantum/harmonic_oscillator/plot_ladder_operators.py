r"""
Dirac's ladder-operator method
==================================

Builds the harmonic oscillator's raising/lowering operators
:math:`\hat a^\dagger,\hat a` in a truncated Fock basis with
:func:`~physicskit.quantum.core.operators.annihilation_operator` and
:func:`~physicskit.quantum.core.operators.creation_operator`, verifies the
algebra :math:`[\hat a,\hat a^\dagger]=1`, and reads the spectrum
:math:`E_n=\hbar\omega(n+\tfrac12)` directly off the
:func:`~physicskit.quantum.core.operators.number_operator` -- reproducing
the oscillator's energy ladder without ever solving a differential
equation.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.quantum.chapters.harmonic_spin import HarmonicOscillator
from physicskit.quantum.core.operators import (
    annihilation_operator,
    commutator,
    creation_operator,
    is_hermitian,
    number_operator,
)

ho = HarmonicOscillator()
n_max = 12

a = annihilation_operator(n_max)
a_dag = creation_operator(n_max)
N = number_operator(n_max)

# The bulk of [a, a^dagger] is 1 (truncation only breaks the top level).
comm = commutator(a, a_dag)
print("[a, a^dagger] diagonal (should be 1, except the truncated top level):")
print(np.round(np.real(np.diag(comm)), 6))
print("N = a^dagger a is Hermitian:", is_hermitian(N))

# Build the spectrum by repeated raising from the vacuum |0>, annihilated by a.
vacuum = np.zeros(n_max, dtype=complex)
vacuum[0] = 1.0
print("a|0> = 0:", np.allclose(a @ vacuum, 0.0))

E_ladder = ho.hbar * ho.omega * (np.diag(N).real + 0.5)
E_exact = ho.energy(np.arange(n_max))

# %%
# The ladder-operator spectrum :math:`E_n=\hbar\omega(n+1/2)` against the
# exact eigenfunction energies, and the operator matrices themselves
# --------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(np.arange(n_max), E_exact, "o", label=r"$E_n$ from HarmonicOscillator.energy")
axes[0].plot(np.arange(n_max), E_ladder, "x", label=r"$\hbar\omega(N+1/2)$ from the ladder operators")
axes[0].set_xlabel("n")
axes[0].set_ylabel("E")
axes[0].set_title("Energy ladder built purely from operator algebra")
axes[0].legend(fontsize=8)

im = axes[1].imshow(np.abs(a), cmap="viridis")
axes[1].set_title(
    r"$|\hat a|$ in the truncated Fock basis"
    "\n(nonzero only one step below the diagonal)"
)
fig.colorbar(im, ax=axes[1], fraction=0.046)

fig.tight_layout()

print(f"max |E_ladder - E_exact| (excluding the truncated top level): {np.max(np.abs(E_ladder[:-1] - E_exact[:-1])):.2e}")
