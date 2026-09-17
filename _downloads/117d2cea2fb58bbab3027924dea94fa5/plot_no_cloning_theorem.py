r"""
The no-cloning theorem: why a fixed unitary can copy a basis, not a state
================================================================================

Wootters, Zurek, and (independently) Dieks proved that no unitary process
can take an arbitrary, unknown quantum state and produce two independent
copies of it. The proof is a direct consequence of linearity: a unitary
that faithfully clones two particular states must, by linearity, act on
their superposition in a way that is *not* a faithful copy of that
superposition. This example makes the failure completely explicit with
the simplest possible candidate "cloning machine," the CNOT gate: it
successfully copies the computational basis states :math:`\lvert0\rangle`
and :math:`\lvert1\rangle`, but applied to a superposition it produces
exactly a Bell state -- :func:`physicskit.quantum.chapters.entanglement.bell_state` --
rather than two independent copies, entangling the two qubits instead of
cloning either one.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.quantum.chapters.entanglement import bell_state

# %%
# CNOT as a candidate cloning machine
# ------------------------------------------
# CNOT\|source>\|target=0> flips the target exactly when the source is
# \|1>: CNOT\|0>\|0>=\|0>\|0>, CNOT\|1>\|0>=\|1>\|1> -- both genuine
# copies, since the target ends up identical to the source in both cases.
CNOT = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=complex,
)

ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)


def clone_attempt(psi):
    """Apply CNOT to psi (x) |0>, the candidate cloning operation."""
    input_state = np.kron(psi, ket0)
    return CNOT @ input_state


out0 = clone_attempt(ket0)
out1 = clone_attempt(ket1)
print("CNOT|0>|0> =", np.round(out0.real, 3), " (matches |0>|0>: perfect copy of |0>)")
print("CNOT|1>|0> =", np.round(out1.real, 3), " (matches |1>|1>: perfect copy of |1>)")

# %%
# The same machine, applied to a superposition
# ---------------------------------------------------
# A faithful clone of \|+> = (\|0>+\|1>)/sqrt(2) would produce the PRODUCT
# state \|+>\|+> = (\|00>+\|01>+\|10>+\|11>)/2. What CNOT actually produces,
# by linearity from the two basis results above, is instead exactly a
# Bell state: entangled, not a pair of independent copies at all.
plus = (ket0 + ket1) / np.sqrt(2)
actual_output = clone_attempt(plus)
would_be_clone = np.kron(plus, plus)
bell = bell_state("phi+")

print(f"\nactual CNOT output on |+>|0>:      {np.round(actual_output.real, 4)}")
print(f"a faithful clone would have given: {np.round(would_be_clone.real, 4)}")
print(f"physicskit's own Bell state |phi+>: {np.round(bell.real, 4)}")
print(f"|actual output - Bell state|:       {np.linalg.norm(actual_output - bell):.2e}  (CNOT produced exactly a Bell state)")

fidelity = np.abs(np.vdot(would_be_clone, actual_output)) ** 2
print(f"\nfidelity between the actual output and a faithful clone: {fidelity:.4f}  (far below 1 -- cloning failed)")

# %%
# Cloning fidelity across every possible input state
# --------------------------------------------------------
# CNOT clones perfectly only the two states it was "tuned" for (theta=0
# and theta=pi below); everywhere else -- every unknown superposition an
# actual cloning machine would need to handle -- the fidelity to a
# faithful copy drops well below 1, vanishing entirely at the equator
# where the state is an equal superposition.
theta_values = np.linspace(0, np.pi, 200)
fidelities = []
for theta in theta_values:
    psi = np.cos(theta / 2) * ket0 + np.sin(theta / 2) * ket1
    actual = clone_attempt(psi)
    ideal_clone = np.kron(psi, psi)
    fidelities.append(np.abs(np.vdot(ideal_clone, actual)) ** 2)
fidelities = np.array(fidelities)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(np.degrees(theta_values), fidelities, color="firebrick")
ax.axvline(0, color="0.6", ls="--", lw=1)
ax.axvline(180, color="0.6", ls="--", lw=1)
ax.set_xlabel(r"input state angle $\theta$ (degrees), $|\psi\rangle=\cos(\theta/2)|0\rangle+\sin(\theta/2)|1\rangle$")
ax.set_ylabel("cloning fidelity")
ax.set_title("CNOT clones only the two states it was built for -- theta=0 and theta=180")
fig.tight_layout()

print(f"\nfidelity at theta=0 (|0>):   {fidelities[0]:.6f}")
print(f"fidelity at theta=90 (|+>):  {fidelities[len(fidelities) // 2]:.6f}")
print(f"fidelity at theta=180 (|1>): {fidelities[-1]:.6f}")
print("\nno fixed unitary reaches fidelity 1 across the whole range: exactly the no-cloning theorem's content.")

plt.show()
