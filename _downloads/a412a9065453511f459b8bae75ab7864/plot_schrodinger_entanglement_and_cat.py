r"""
Schrödinger names entanglement: the whole known, the parts not
==================================================================

Replying to Einstein, Podolsky and Rosen in 1935, Schrödinger named the
effect *Verschränkung*, entanglement, and called it "the characteristic
trait of quantum mechanics". His point: two systems can be in a
completely known joint state while neither one, on its own, has any
definite state at all. He dramatized it with a cat whose life is tied
to an undecayed atom.

This example makes his three observations quantitative with
:func:`~physicskit.quantum.chapters.entanglement.bell_state`: a Bell
state is pure while each half is maximally mixed; measuring one half in
*any* basis steers the other into a matching state (the "steering" he
described in the same paper); and the atom-cat system grows entangled as
the atom's decay probability rises, leaving the cat alone in a mixture,
not a superposition.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.quantum.chapters.entanglement import bell_state


def reduced_first(psi):
    """Reduced density matrix of the first qubit of a two-qubit pure state."""
    m = psi.reshape(2, 2)
    return m @ m.conj().T


def entropy(rho):
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-12]
    return float(-np.sum(w * np.log2(w)))


# %%
# Pure whole, mixed parts
# ---------------------------
# :math:`\lvert\Phi^+\rangle=(\lvert00\rangle+\lvert11\rangle)/\sqrt2` has
# purity 1. Tracing out either qubit leaves :math:`I/2`, purity 1/2, one
# full bit of entropy: nothing at all is known about the part.
phi_plus = bell_state("phi+")
rho_full = np.outer(phi_plus, phi_plus.conj())
rho_1 = reduced_first(phi_plus)
print(f"whole state: purity {np.trace(rho_full @ rho_full).real:.3f}, entropy {entropy(rho_full):.3f} bits")
print(f"one qubit:   purity {np.trace(rho_1 @ rho_1).real:.3f}, entropy {entropy(rho_1):.3f} bits")
print(f"reduced state = I/2: {np.allclose(rho_1, np.eye(2) / 2)}")

# %%
# Steering
# ------------
# Measure qubit 1 along an axis at angle :math:`\theta` in the x-z plane.
# For each outcome, qubit 2 is left in a pure state. For
# :math:`\Phi^+` it is the *same* state qubit 1 was found in: whichever
# basis Alice chooses, Bob's qubit ends up in that basis. Averaged over
# outcomes, Bob's state is still :math:`I/2`, so nothing can be
# signalled.
thetas = np.linspace(0, np.pi, 7)
fidelities = []
for th in thetas:
    up = np.array([np.cos(th / 2), np.sin(th / 2)])
    down = np.array([-np.sin(th / 2), np.cos(th / 2)])
    fid = []
    for outcome in (up, down):
        bob = outcome.conj() @ phi_plus.reshape(2, 2)
        bob /= np.linalg.norm(bob)
        fid.append(abs(np.vdot(outcome.conj(), bob)) ** 2)
    fidelities.append(min(fid))
print(f"\nBob's state matches Alice's outcome for every basis: min fidelity {min(fidelities):.6f}")

# %%
# The cat
# -----------
# An atom with decay probability :math:`p(t)=1-e^{-t/\tau}` is coupled to a
# cat: :math:`\sqrt{1-p}\,\lvert\text{atom},\text{alive}\rangle +
# \sqrt{p}\,\lvert\text{decayed},\text{dead}\rangle`. The cat's reduced
# state is diagonal, with no alive/dead coherence at all. Its
# entanglement with the atom peaks at one full bit when :math:`p = 1/2`,
# after one half-life.
t = np.linspace(0, 4, 400)  # in units of the atom's lifetime tau
p = 1 - np.exp(-t)
S_cat = []
for pk in p:
    psi = np.array([np.sqrt(1 - pk), 0.0, 0.0, np.sqrt(pk)])  # |undecayed, alive>, |decayed, dead>
    rho_cat = reduced_first(psi[[0, 2, 1, 3]])  # reorder so the cat is the first factor
    S_cat.append(entropy(rho_cat))
S_cat = np.array(S_cat)
rho_cat_half = reduced_first(np.array([1, 0, 0, 1]) / np.sqrt(2))
print(f"\ncat's reduced state at one half-life:\n{np.round(rho_cat_half.real, 3)}  (no off-diagonal coherence)")
print(f"maximum atom-cat entanglement {S_cat.max():.3f} bits at t = {t[np.argmax(S_cat)]:.3f} tau (ln 2 = {np.log(2):.3f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
labels = [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"]
ax1.imshow(rho_full.real, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
ax1.set_xticks(range(4), labels)
ax1.set_yticks(range(4), labels)
ax1.set_title(r"$\Phi^+$: pure, with coherence between $|00\rangle$ and $|11\rangle$", fontsize=9)
for (i, j), val in np.ndenumerate(rho_full.real):
    if abs(val) > 1e-9:
        ax1.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)
ax2.plot(t, p, color="0.5", ls="--", label="decay probability p(t)")
ax2.plot(t, S_cat, color="firebrick", label="atom-cat entanglement entropy [bits]")
ax2.axvline(np.log(2), color="k", ls=":", lw=0.8)
ax2.set_xlabel(r"time $t/\tau$")
ax2.set_title("Schrödinger's cat: entangled with the atom")
ax2.legend(fontsize=8)
fig.tight_layout()

plt.show()
