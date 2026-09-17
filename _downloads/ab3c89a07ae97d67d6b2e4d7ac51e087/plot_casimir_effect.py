r"""
The Casimir energy grows weaker as plates separate
========================================================

Hendrik Casimir showed in 1948 that two uncharged, perfectly conducting
parallel plates in vacuum should attract each other: only electromagnetic
vacuum modes that fit an integer number of wavelengths between the
plates are allowed there, so the confined zero-point energy is slightly
lower than in the unbounded vacuum outside, producing a net inward force.
This is a simplified, standard textbook treatment -- a 1D scalar field
confined between two "plates" a distance :math:`d` apart, not a full 3D
electromagnetic calculation.
:func:`~physicskit.fields.quantum_fields.casimir_mode_frequencies` builds
the resulting discrete standing-wave spectrum,

.. math::

    \omega_n = \frac{n\pi c}{d}, \qquad n = 1, 2, 3, \dots,

in place of the continuum of frequencies allowed in unbounded vacuum.
The bare zero-point sum :math:`\tfrac12\sum_n \omega_n` diverges, so
:func:`~physicskit.fields.quantum_fields.casimir_energy_1d` regularizes
it with an exponential (Abel-summation) cutoff, subtracts the same
divergent piece a plate-free continuum calculation would produce under
the same regulator, and takes the cutoff to zero, leaving the finite,
physical remainder

.. math::

    E(d) \to -\frac{\pi c}{24 d},

the standard 1D massless-field Casimir energy -- reproducing the
qualitative Casimir signature (an attractive, separation-dependent
zero-point energy) directly.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.fields import animate_casimir_modes, casimir_energy_1d, casimir_mode_frequencies

# %%
# Sweep the plate separation and animate the mode spectrum and energy
# --------------------------------------------------------------------------

d_values = np.linspace(1.0, 6.0, 40)

anim = animate_casimir_modes(d_values)
plt.show()

# %%
# To save the animation to a file instead of (or in addition to)
# displaying it interactively, use e.g.::
#
#     anim.save("casimir_effect.gif", writer="pillow", fps=12)

# %%
# The discrete spectrum grows denser (approaching the continuum) as the
# plates separate, and the regularized zero-point energy is negative
# (attractive) everywhere, growing less negative -- weaker confinement of
# vacuum energy -- as the separation grows.

energies = np.array([casimir_energy_1d(d) for d in d_values])
modes_at_small_d = casimir_mode_frequencies(d_values[0], n_max=5)
modes_at_large_d = casimir_mode_frequencies(d_values[-1], n_max=5)

print(f"first 5 mode frequencies at d={d_values[0]:.2f}: {np.round(modes_at_small_d, 3)}")
print(f"first 5 mode frequencies at d={d_values[-1]:.2f}: {np.round(modes_at_large_d, 3)} (denser -> continuum)")
print(f"Casimir energy: E(d={d_values[0]:.2f})={energies[0]:.4f}, E(d={d_values[-1]:.2f})={energies[-1]:.4f}")
print(f"force is attractive everywhere (energy increases monotonically with d): {bool(np.all(np.diff(energies) > 0))}")
