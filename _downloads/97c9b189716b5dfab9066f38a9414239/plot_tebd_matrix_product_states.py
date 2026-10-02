r"""
TEBD on matrix product states: spin chains beyond exact diagonalization
=======================================================================

The Hilbert space of :math:`N` spins grows as :math:`2^N`, but the
physically relevant states of 1D chains carry little entanglement and fit
in a matrix product state (MPS) of bond dimension :math:`\chi`. White's
DMRG (1992) found ground states this way; Vidal's time-evolving block
decimation (TEBD, 2003-2004) evolves an MPS in time by splitting
:math:`e^{-iH\delta t}` into two-site gates on even and odd bonds and
truncating each bond back to :math:`\chi` with an SVD. In imaginary time the
same algorithm finds ground states.

This example quenches an antiferromagnetic Néel state under the open
Heisenberg chain, checks TEBD against the exact evolution from
:mod:`physicskit.condensed.spin_chains` at :math:`N = 12`, follows the
entanglement growth that sets the cost, and then finds the ground state of
a 60-site chain, far beyond exact diagonalization, comparing its energy per
bond with Bethe's :math:`\tfrac14 - \ln 2`.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import expm_multiply

from physicskit.condensed.spin_chains import entanglement_entropy, xxz_hamiltonian
from physicskit.quantum.chapters.tensor_networks import MPS, tebd, xxz_bond_hamiltonians

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
Sz = np.diag([0.5, -0.5])

# %%
# Néel quench at N = 12: TEBD against exact time evolution
# --------------------------------------------------------
N = 12
H = xxz_hamiltonian(N, periodic=False)[0]
dt, steps = 0.02, 200
times = dt * np.arange(0, steps + 1, 10)
v0 = MPS.product_state([0, 1] * (N // 2)).to_dense()
exact = expm_multiply(-1j * H, v0, start=0.0, stop=times[-1], num=len(times), endpoint=True)
S_exact = np.array([entanglement_entropy(v, N, N // 2) for v in exact])
for chi, color in [(4, "#9ecae1"), (8, "#4292c6"), (64, "#08306b")]:
    psi = MPS.product_state([0, 1] * (N // 2))
    out = tebd(psi, xxz_bond_hamiltonians(N), dt, steps, chi_max=chi, measure_every=10, observables={"Sz": Sz})
    axes[0].plot(out["t"], out["entropy"], color=color, label=rf"TEBD, $\chi = {chi}$")
    if chi == 64:
        fidelity = abs(np.vdot(exact[-1], psi.to_dense()))
        sz_tebd = out["Sz"][:, 0]
axes[0].plot(times, S_exact, "k--", lw=1, label="exact")
axes[0].set_xlabel("time t (1/J)")
axes[0].set_ylabel("half-chain entanglement entropy")
axes[0].set_title("Entanglement growth after a Néel quench, N = 12")
axes[0].legend(fontsize=8)

sz_exact = []
for v in exact:
    p = np.abs(v) ** 2
    up = (np.arange(2**N) >> (N - 1)) & 1  # site 0 is the most significant bit
    sz_exact.append(np.sum(p * (up - 0.5)))
axes[1].plot(times, sz_exact, "k-", lw=1, label="exact")
axes[1].plot(times, sz_tebd, "o", ms=4, color="crimson", label=r"TEBD, $\chi = 64$")
axes[1].set_xlabel("time t (1/J)")
axes[1].set_ylabel(r"$\langle S^z_0 \rangle$")
axes[1].set_title(f"Edge magnetization; final fidelity {fidelity:.6f}")
axes[1].legend(fontsize=8)

# %%
# Ground state of a 60-site chain
# -------------------------------
# Imaginary-time TEBD with chi = 16 (the exact state would need 2^30). The
# energy per bond in the middle of the chain approaches Bethe's
# thermodynamic value 1/4 - ln 2 for the infinite chain.
N_big = 60
H_big = xxz_bond_hamiltonians(N_big)
gs = MPS.product_state([0, 1] * (N_big // 2))
for d, n_tau in ((0.1, 60), (0.05, 60)):
    tebd(gs, H_big, d, n_tau, chi_max=16, imaginary=True, measure_every=n_tau)
bond_E = gs.bond_energies(H_big)
bethe = 0.25 - np.log(2)
axes[2].plot(np.arange(N_big - 1) + 0.5, bond_E, ".-", color="navy", label="TEBD bond energies")
axes[2].axhline(bethe, color="crimson", ls="--", label=r"Bethe: $\frac{1}{4} - \ln 2$")
axes[2].set_xlabel("bond")
axes[2].set_ylabel(r"$\langle h_{j,j+1} \rangle$")
axes[2].set_title(f"N = {N_big} ground state, bond dimension {max(gs.bond_dimensions)}")
axes[2].legend(fontsize=8)
plt.tight_layout()
plt.show()
middle = bond_E[N_big // 2 - 6 : N_big // 2 + 6]
# open-chain dimerization alternates bond energies; average neighbouring pairs
e_mid = 0.5 * (middle[::2] + middle[1::2]).mean()
print(f"quench fidelity at t = {times[-1]}: {fidelity:.8f}")
print(f"mid-chain energy per bond {e_mid:.4f}; Bethe {bethe:.4f}")

# %%
# Check
# -----
assert fidelity > 0.99999
np.testing.assert_allclose(sz_tebd, sz_exact, atol=1e-4)
assert abs(e_mid - bethe) < 0.01
