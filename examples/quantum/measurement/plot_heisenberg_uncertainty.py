r"""
Heisenberg's uncertainty principle
======================================

Checks the Heisenberg bound :math:`\Delta x\,\Delta p \ge \hbar/2` directly
from harmonic-oscillator wavefunctions with
:func:`~physicskit.quantum.utils.measure.uncertainty`, and verifies the
underlying canonical commutation relation :math:`[\hat x,\hat p]=i\hbar` in a
truncated Fock basis with
:func:`~physicskit.quantum.core.operators.commutator`,
:func:`~physicskit.quantum.core.operators.position_operator`, and
:func:`~physicskit.quantum.core.operators.momentum_operator`.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.quantum.chapters.harmonic_spin import HarmonicOscillator
from physicskit.quantum.core.operators import commutator, momentum_operator, position_operator
from physicskit.quantum.utils.measure import uncertainty

ho = HarmonicOscillator()
x = np.linspace(-12, 12, 4000)

n_values = np.arange(0, 8)
dx_vals, dp_vals, prod_vals = [], [], []
for n in n_values:
    psi_n = ho.eigenfunction(int(n), x)
    dx, dp, prod = uncertainty(x, psi_n, hbar=ho.hbar)
    dx_vals.append(dx)
    dp_vals.append(dp)
    prod_vals.append(prod)

# %%
# :math:`\Delta x\,\Delta p` for Fock states :math:`|n\rangle`, always at or
# above the Heisenberg bound (equality only at :math:`n=0`)
# --------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(n_values, prod_vals, "o-", label=r"$\Delta x\,\Delta p$ (numeric)")
axes[0].axhline(ho.hbar / 2, color="gray", ls="--", label=r"Heisenberg bound $\hbar/2$")
axes[0].set_xlabel("n")
axes[0].set_ylabel(r"$\Delta x \, \Delta p$")
axes[0].set_title(
    r"Fock state $|n\rangle$: uncertainty grows with $n$,"
    "\nnever dropping below the bound"
)
axes[0].legend(fontsize=8)

# %%
# The canonical commutation relation, verified in a truncated Fock basis
# ----------------------------------------------------------------------------

n_max = 40
x_op = position_operator(n_max, m=ho.m, omega=ho.omega, hbar=ho.hbar)
p_op = momentum_operator(n_max, m=ho.m, omega=ho.omega, hbar=ho.hbar)
comm = commutator(x_op, p_op)
target = 1j * ho.hbar * np.eye(n_max)

# Truncation breaks the relation only in the last diagonal entry (raising
# out of the truncated basis is cut off there); the bulk matches i*hbar*I
# exactly.
bulk = slice(0, n_max - 1)
max_err_bulk = np.max(np.abs(comm[bulk, bulk] - target[bulk, bulk]))
print(f"max |[x,p] - i*hbar*I| over the bulk (excluding the last row/col): {max_err_bulk:.2e}")

im = axes[1].imshow(np.imag(comm[bulk, bulk]), cmap="RdBu_r", vmin=-1.2, vmax=1.2)
axes[1].set_title(
    r"Im $[\hat x,\hat p]$ in a truncated Fock basis"
    "\n(should be $\\hbar$ on the diagonal)"
)
fig.colorbar(im, ax=axes[1], fraction=0.046)

fig.tight_layout()

# %%
# The squeezed-vacuum momentum uncertainty away from the squeezing axis
# --------------------------------------------------------------------------
#
# :meth:`~physicskit.quantum.chapters.harmonic_spin.HarmonicOscillator.squeezed_vacuum_wavefunction`
# takes a squeezing angle :math:`\phi` that rotates the uncertainty
# ellipse in phase space: at :math:`\phi=0` position is squeezed, at
# :math:`\phi=\pi/2` momentum is, and in between the squeezing sits
# along a rotated quadrature, so :math:`\Delta x` and :math:`\Delta p`
# both depend on :math:`r` and :math:`\phi`, and their product exceeds
# :math:`\hbar/2` away from the two axes even though the state is still
# a minimum-uncertainty state for its own rotated quadratures.

r_values = np.linspace(0.0, 1.2, 40)
phi_values = np.linspace(0.0, np.pi / 2, 40)
dp_map = np.zeros((len(r_values), len(phi_values)))
prod_ratio = np.zeros_like(phi_values)
x_sq = np.linspace(-10, 10, 1500)
for i, r_val in enumerate(r_values):
    for j, phi_val in enumerate(phi_values):
        psi_rp = ho.squeezed_vacuum_wavefunction(x_sq, r_val, phi_val)
        _dx_rp, dp_rp, prod_rp = uncertainty(x_sq, psi_rp, hbar=ho.hbar)
        dp_map[i, j] = dp_rp
        if i == 0:
            prod_ratio[j] = prod_rp / ho.hbar

fig2, ax2 = plt.subplots(figsize=(6.5, 5))
im2 = ax2.pcolormesh(phi_values, r_values, dp_map, shading="auto", cmap="magma")
ax2.set_xlabel(r"squeezing angle $\phi$")
ax2.set_ylabel(r"squeezing amplitude $r$")
ax2.set_title(r"Squeezed vacuum: $\Delta p(r,\phi)$" "\n(anti-squeezed at phi=0, squeezed at phi=pi/2)")
fig2.colorbar(im2, ax=ax2, label=r"$\Delta p$")
fig2.tight_layout()

print(f"Delta p(r={r_values[-1]:.2f}, phi=0) = {dp_map[-1, 0]:.4f}   Delta p(r={r_values[-1]:.2f}, phi={phi_values[-1]:.2f}) = {dp_map[-1, -1]:.4f}")
print("Delta x*Delta p/hbar along the r=0 row (the vacuum, whatever phi):")
print("  ", np.round(prod_ratio[:: len(prod_ratio) // 4], 4))
