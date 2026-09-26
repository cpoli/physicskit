r"""
Pauli's exclusion principle and spin matrices
================================================

In 1925 Pauli explained the shell structure of atoms with two rules: an
electron needs a fourth quantum number with just two values, and no two
electrons may share all four. Uhlenbeck and Goudsmit identified the
fourth number as spin, and in 1927 Pauli wrote spin as a two-component
wavefunction acted on by the matrices

.. math::

    \sigma_x = \begin{pmatrix}0&1\\1&0\end{pmatrix},\quad
    \sigma_y = \begin{pmatrix}0&-i\\i&0\end{pmatrix},\quad
    \sigma_z = \begin{pmatrix}1&0\\0&-1\end{pmatrix}.

This example checks the Pauli-matrix algebra with
:func:`~physicskit.quantum.core.operators.spin_operator`, shows the
spinor's sign change under a full turn, builds the antisymmetric
two-electron wavefunction that enforces exclusion, and fills atomic
shells to recover the noble-gas numbers.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from physicskit.quantum.core.operators import commutator, spin_operator

# %%
# The Pauli algebra
# ---------------------
# :math:`\sigma_i\sigma_j = \delta_{ij}I + i\epsilon_{ijk}\sigma_k`: each
# matrix squares to the identity, they anticommute, and their commutators
# close into the angular-momentum algebra :math:`[S_x,S_y]=i\hbar S_z`.
sig = {a: 2 * spin_operator(a, s=0.5) for a in "xyz"}
I2 = np.eye(2)
eps = {("x", "y"): "z", ("y", "z"): "x", ("z", "x"): "y"}
ok = all(np.allclose(sig[a] @ sig[a], I2) for a in "xyz")
for (a, b), c in eps.items():
    ok &= np.allclose(sig[a] @ sig[b], 1j * sig[c]) and np.allclose(sig[b] @ sig[a], -1j * sig[c])
print(f"sigma_i sigma_j = delta_ij + i eps_ijk sigma_k for all pairs: {ok}")
print(f"[S_x, S_y] = i S_z: {np.allclose(commutator(sig['x'] / 2, sig['y'] / 2), 1j * sig['z'] / 2)}")
print(f"eigenvalues of every sigma: {np.linalg.eigvalsh(sig['x'])}  (two states only)")

# %%
# A full turn flips the sign
# ------------------------------
# Rotating a spin-1/2 state by :math:`\theta` about z applies
# :math:`e^{-i\theta\sigma_z/2}`. At :math:`\theta=2\pi` the state comes
# back as :math:`-\psi`; only a :math:`4\pi` turn is the identity. No
# classical vector behaves this way.
thetas = np.linspace(0, 4 * np.pi, 200)
overlap = np.array([np.trace(expm(-1j * t * sig["z"] / 2)).real / 2 for t in thetas])
print(f"U(2 pi) = {np.round(expm(-1j * 2 * np.pi * sig['z'] / 2).real, 6).tolist()}")

# %%
# Exclusion from antisymmetry
# -------------------------------
# Two electrons in a box, in orbitals :math:`\phi_a` and :math:`\phi_b`.
# The total wavefunction must change sign when the electrons are
# swapped. In the spin triplet the spatial part is the Slater
# determinant :math:`\phi_a(x_1)\phi_b(x_2)-\phi_b(x_1)\phi_a(x_2)`: it
# vanishes whenever :math:`x_1 = x_2`, and entirely if :math:`a = b`.
# That is the exclusion principle.
x = np.linspace(0, 1, 200)
phi = lambda k: np.sqrt(2) * np.sin(k * np.pi * x)  # noqa: E731
X1, X2 = np.meshgrid(x, x, indexing="ij")
a, b = 1, 2
triplet = (np.outer(phi(a), phi(b)) - np.outer(phi(b), phi(a))) / np.sqrt(2)
singlet = (np.outer(phi(a), phi(b)) + np.outer(phi(b), phi(a))) / np.sqrt(2)
same = np.outer(phi(a), phi(a)) - np.outer(phi(a), phi(a))
print(f"\nSlater determinant with both electrons in orbital {a}: max |psi| = {np.abs(same).max():.1e}")
print(f"triplet |psi|^2 on the diagonal x1 = x2: max {np.max(np.diag(triplet) ** 2):.1e}")

# %%
# Filling the shells
# ----------------------
# Each hydrogen-like orbital :math:`(n,l,m_l)` holds two electrons, one
# per spin state, so subshell :math:`l` holds :math:`2(2l+1)`. Filling
# in the Madelung order (increasing :math:`n+l`, then :math:`n`) and
# stopping at each completed *p* subshell gives the noble gases.
subshells = sorted(((n, l) for n in range(1, 8) for l in range(n)), key=lambda nl: (nl[0] + nl[1], nl[0]))
Z, noble = 0, []
for n, l in subshells:
    Z += 2 * (2 * l + 1)
    if l == 1 or (n, l) == (1, 0):
        noble.append(Z)
print(f"\nclosed shells (noble gases) at Z = {noble[:6]}  (He, Ne, Ar, Kr, Xe, Rn)")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
ax1.plot(thetas / np.pi, overlap, color="steelblue")
ax1.axhline(0, color="0.7", lw=0.8)
for t in (2, 4):
    ax1.axvline(t, color="0.5", ls=":")
ax1.set_xlabel(r"rotation angle $\theta/\pi$")
ax1.set_ylabel(r"$\langle\psi|U(\theta)|\psi\rangle$ (averaged)")
ax1.set_title(r"Spinors: $-1$ after $2\pi$, $+1$ after $4\pi$")
for ax, psi, title in [(ax2, singlet, "spin singlet (symmetric space)"), (ax3, triplet, "spin triplet (antisymmetric space)")]:
    ax.pcolormesh(X1, X2, psi**2, cmap="magma", shading="auto")
    ax.plot([0, 1], [0, 1], "w--", lw=0.8)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.set_title(title + r": $|\psi(x_1,x_2)|^2$", fontsize=9)
fig.tight_layout()

plt.show()
