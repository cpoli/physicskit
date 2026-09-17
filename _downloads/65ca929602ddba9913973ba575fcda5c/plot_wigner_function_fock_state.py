r"""
Wigner's phase-space distribution: negativity of a Fock state
===================================================================

Eugene Wigner introduced a quasi-probability distribution :math:`W(x,p)`
that represents a quantum state jointly over the dimensionless quadratures
:math:`x = (a+a^\dagger)/\sqrt2` and :math:`p = (a-a^\dagger)/(i\sqrt2)`,
reproducing the correct marginal probability densities upon integration
along either axis, while permitting *negative* values elsewhere -- a
signature with no classical phase-space analogue.
:func:`~physicskit.optics.quantum_optics.compute_wigner_function` evaluates
:math:`W(x,p) = \sum_{n,m}\rho_{nm}K_{nm}(x,p)` on a phase-space grid from
the density matrix :math:`\rho=\lvert\psi\rangle\langle\psi\rvert` of an
arbitrary Fock-basis state vector; for a number (Fock) state
:math:`\lvert n\rangle` (diagonal :math:`\rho`) this reduces to the
closed form

.. math::

    W_n(x,p) = \frac{(-1)^n}{\pi}\, e^{-(x^2+p^2)}\, L_n\!\left[2(x^2+p^2)\right],

with :math:`L_n` the Laguerre polynomial -- for the single-photon state
:math:`n=1` studied below, :math:`W_1(0,0) = -1/\pi`, the negative dip at
the phase-space origin. :func:`~physicskit.optics.quantum_optics.wigner_negativity`
integrates :math:`\max(0,-W)` over phase space to quantify the negative
volume that marks a state as nonclassical -- zero for any classical state,
strictly positive for a single-photon Fock state.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.optics.quantum_optics import compute_wigner_function, fock_state, wigner_negativity

# %%
# The single-photon Fock state :math:`|1\rangle`, and its Wigner function
# -----------------------------------------------------------------------

cutoff = 10
psi = fock_state(1, cutoff)  # single-photon Fock state |1>
x = np.linspace(-4, 4, 121)
p = np.linspace(-4, 4, 121)
W = compute_wigner_function(psi, x, p)

N_W = wigner_negativity(W, x, p)

# %%
# The Wigner function of :math:`|1\rangle` dips below zero right at the phase-space
# origin -- a region no classical probability distribution could occupy --
# and its integrated negative volume N_W is decisively nonzero, unlike any
# classical (or coherent, or thermal) state.

fig, ax = plt.subplots(figsize=(5, 4))
vmax = np.abs(W).max()
im = ax.contourf(x, p, W.T, levels=40, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
fig.colorbar(im, ax=ax, label="W(x, p)")
ax.set_xlabel("x")
ax.set_ylabel("p")
ax.set_title("Wigner function of |n=1>: negative rings, no classical analogue")
fig.tight_layout()

i0, j0 = np.argmin(np.abs(x)), np.argmin(np.abs(p))
print(f"Wigner negativity N_W = {N_W:.4f}  (zero for any classical state)")
print(f"W(0, 0) = {W[i0, j0]:.4f}  (exact value: -1/pi = {-1 / np.pi:.4f})")

vac = fock_state(0, cutoff)
W_vac = compute_wigner_function(vac, x, p)
N_W_vac = wigner_negativity(W_vac, x, p)
print(f"for comparison, the vacuum |0> (classical): N_W = {N_W_vac:.6f}")
