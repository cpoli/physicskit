r"""
Dirac's quantization of the electromagnetic field: ladder operators
==========================================================================

Dirac gave the first fully quantum treatment of radiation by promoting
each field mode to a quantum harmonic oscillator, with non-commuting
annihilation and creation operators :math:`\hat a,\hat a^\dagger`
satisfying :math:`[\hat a,\hat a^\dagger]=1`. Photon number becomes the
oscillator's excitation number, :math:`\hat n=\hat a^\dagger\hat a`, with
:math:`\hat a` and :math:`\hat a^\dagger` respectively removing and
adding one field quantum -- exactly the ladder structure Einstein's 1905
light quanta had implied but never derived. This example builds
:math:`\hat a` and :math:`\hat a^\dagger` directly with
:func:`~physicskit.quantum.core.operators.annihilation_operator` and
:func:`~physicskit.quantum.core.operators.creation_operator`, verifies
the commutation relation and the ladder action on Fock states, and shows
the vacuum-fluctuation fact behind Dirac's other major result: even the
vacuum :math:`\lvert0\rangle` is not "nothing" to :math:`\hat a^\dagger`,
which is exactly what makes spontaneous emission unavoidable.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.quantum.core.operators import (
    annihilation_operator,
    commutator,
    creation_operator,
    number_operator,
)

# %%
# Building the ladder operators, and checking the commutator
# ------------------------------------------------------------------
n_max = 12
a = annihilation_operator(n_max)
a_dag = creation_operator(n_max)

comm = commutator(a, a_dag)
print("[a, a_dagger] on the truncated basis (should be the identity, except at the truncation edge):")
print(np.round(comm.real, 4))
print(f"\nmax deviation from identity, excluding the last (truncated) basis state: {np.max(np.abs(comm[:-1, :-1] - np.eye(n_max - 1))):.2e}")
print(f"deviation at the truncated edge (n_max-1): {comm[-1, -1].real:.4f} (a finite-basis artifact -- a true infinite Fock space has none)")

# %%
# The ladder action on Fock states
# --------------------------------------
# a|n> = sqrt(n)|n-1>, a_dagger|n> = sqrt(n+1)|n+1> -- verified directly
# as matrix-vector products against basis vectors, not assumed.
n_test = 4
psi_n = np.zeros(n_max, dtype=complex)
psi_n[n_test] = 1.0

lowered = a @ psi_n
raised = a_dag @ psi_n
lowered_idx, lowered_amp = np.argmax(np.abs(lowered)), np.abs(lowered).max()
raised_idx, raised_amp = np.argmax(np.abs(raised)), np.abs(raised).max()
print(f"\nstarting from |n={n_test}>:")
print(f"  a|n> has its only nonzero entry at index {lowered_idx}, amplitude {lowered_amp:.6f} (expected sqrt({n_test})={np.sqrt(n_test):.6f})")
print(f"  a_dagger|n> has its only nonzero entry at index {raised_idx}, amplitude {raised_amp:.6f} (expected sqrt({n_test + 1})={np.sqrt(n_test + 1):.6f})")

# %%
# The number operator's ladder-derived spectrum
# ----------------------------------------------------
N = number_operator(n_max)
eigenvalues = np.sort(np.linalg.eigvalsh(N).real)
print(f"\nnumber operator N=a_dagger*a eigenvalues: {np.round(eigenvalues, 6)}")
print("(exactly 0, 1, 2, ..., n_max-1 -- built entirely from the ladder operators above)")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(eigenvalues, "o-", color="steelblue")
ax.set_xlabel("eigenstate index")
ax.set_ylabel("photon number eigenvalue")
ax.set_title(r"Spectrum of $\hat N=\hat a^\dagger\hat a$: the discrete ladder")
fig.tight_layout()

# %%
# The vacuum is not "nothing" to a_dagger: the seed of spontaneous emission
# ------------------------------------------------------------------------------------
# <0|a a_dagger|0> = 1, even though <0|a_dagger a|0> = 0 (the vacuum has
# no photons to remove). This asymmetry -- a direct consequence of
# [a, a_dagger]=1 -- is exactly what Dirac showed drives spontaneous
# emission: an excited atom, coupled to the quantized field, is never
# coupled to "nothing" even when the field starts in vacuum.
vacuum = np.zeros(n_max, dtype=complex)
vacuum[0] = 1.0
n_expectation_vacuum = np.real(vacuum.conj() @ N @ vacuum)
a_adag_expectation = np.real(vacuum.conj() @ (a @ a_dag) @ vacuum)
print(f"\n<0|N|0> = {n_expectation_vacuum:.6f}  (no photons present)")
print(f"<0| a a_dagger |0> = {a_adag_expectation:.6f}  (nonzero! the vacuum still 'kicks back' on a_dagger)")
print("this single nonzero vacuum matrix element is the seed of spontaneous emission Dirac derived")
print("from first principles, rather than inserting Einstein's 1917 'A coefficient' by hand.")

plt.show()
