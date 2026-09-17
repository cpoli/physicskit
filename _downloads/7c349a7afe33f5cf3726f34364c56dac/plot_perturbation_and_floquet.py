r"""
Stark and Zeeman splitting: static perturbation theory
==============================================================

Two static, first-order perturbative effects on the hydrogen atom. In a
magnetic field :math:`B`, the weak-field Zeeman effect shifts each
:math:`\lvert l, m_l; s, m_s\rangle` sublevel by

.. math::

    \Delta E = \mu_B B\,(m_l + g_s m_s);

in a uniform electric field :math:`F`, the linear Stark effect splits
hydrogen's 4-fold degenerate :math:`n=2` shell (using parabolic quantum
numbers :math:`n_1, n_2, m` with :math:`n_1+n_2+\lvert m\rvert+1=n`) by

.. math::

    \Delta E = \tfrac{3}{2}\, n\, (n_1 - n_2)\, a_0 F.

The Zeeman effect is the textbook example of non-degenerate perturbation
theory; the linear Stark effect of hydrogen specifically is the
*degenerate* variety, made possible only by hydrogen's accidental level
degeneracy in :math:`l`.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.quantum.chapters.perturbation import stark_n2_quartet, zeeman_spectrum

# %%
# The Stark quartet and the Zeeman splitting
# --------------------------------------------------------------------------
# Left: the three Stark-shifted energies of the :math:`n=2` quartet vs.
# field :math:`F` (the :math:`m=\pm1` sublevels stay unshifted since only
# the :math:`m=0` pair couples to the field). Right: the Zeeman sublevel
# shifts vs. :math:`B` for :math:`l=0,1,2`.

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Linear Stark effect: hydrogen n=2 quartet
F_values = np.linspace(-0.02, 0.02, 50)
shifts = np.array([[lv.shift for lv in stark_n2_quartet(F)] for F in F_values])
for i in range(shifts.shape[1]):
    axes[0].plot(F_values, shifts[:, i])
axes[0].set_xlabel("Field F")
axes[0].set_ylabel("Energy shift")
axes[0].set_title("Linear Stark effect: hydrogen n=2 quartet")

# Zeeman splitting
B_values = np.linspace(0, 2, 40)
for l in [0, 1, 2]:
    levels = np.array([zeeman_spectrum(l, B) for B in B_values])
    for i in range(levels.shape[1]):
        axes[1].plot(B_values, levels[:, i], color=f"C{l}", alpha=0.6, label=(f"l={l}" if i == 0 else None))
axes[1].set_xlabel("Magnetic field B")
axes[1].set_ylabel("Energy shift")
axes[1].set_title("Zeeman sublevel splitting")
axes[1].legend(fontsize=8)

fig.tight_layout()

print(f"Stark n=2 shifts at F={F_values[-1]:.3f}: {shifts[-1]}")
print(f"Zeeman l=1 shifts at B={B_values[-1]:.3f}: {zeeman_spectrum(1, B_values[-1])}")
