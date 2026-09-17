r"""
The TKNN Invariant: Exactly Quantized Chern Numbers
==========================================================

Thouless, Kohmoto, Nightingale, and den Nijs showed that the quantized
Hall conductance is a topological invariant: the integral of the Berry
curvature of the occupied Bloch bands over the Brillouin zone, now called
the (first) Chern number :math:`C`. Because :math:`C` can only change when
a bulk gap closes, it is exact and disorder-independent -- the moment band
theory became *topological* band theory.
:func:`~physicskit.condensed.topology.compute_berry_curvature` and
:func:`~physicskit.condensed.topology.compute_chern_number` implement the
Fukui-Hatsugai-Suzuki lattice discretization of this integral, returning
exactly quantized integers for any gapped Bloch Hamiltonian.
"""

import numpy as np

from physicskit.condensed.models import haldane_model
from physicskit.condensed.topology import compute_berry_curvature, compute_chern_number
from physicskit.condensed.visualizers import plot_berry_curvature

# %%
# Any gapped two-band Bloch Hamiltonian -- here, Haldane's model
# --------------------------------------------------------------------
# The FHS algorithm needs nothing but a function ``H(k1, k2)``; it makes
# no assumption about the model beyond a spectral gap.

H = lambda k1, k2: haldane_model(k1, k2, t=1.0, t2=0.2, phi=np.pi / 2, M=0.0)

# %%
# Berry curvature concentrated near the gapped Dirac points
# ------------------------------------------------------------------
# The per-plaquette Berry curvature of the lower band is sharply peaked
# near the honeycomb lattice's two (former) Dirac points, where the
# time-reversal-breaking mass gap is smallest.

F = compute_berry_curvature(H, grid_size=30, band_index=0)
fig, ax = plot_berry_curvature(F)
ax.set_title("Berry curvature of the lower Haldane band")

# %%
# Integrating the curvature gives an exact integer
# -----------------------------------------------------
# Summing the curvature over the whole Brillouin zone and dividing by
# :math:`2\pi` returns an *exact* integer for every band -- not
# approximately quantized, but exactly, by construction of the FHS link
# variables.

C = compute_chern_number(H, grid_size=30)
print("Chern numbers (per band):", C)
print(f"raw curvature sum / 2*pi for the lower band: {F.sum() / (2 * np.pi):.10f}")
