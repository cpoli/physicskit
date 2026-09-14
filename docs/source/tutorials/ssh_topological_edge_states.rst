:orphan:

The SSH Model and Topological Edge States
============================================

The Su-Schrieffer-Heeger (SSH) chain is the simplest possible topological
band model: a 1D dimerized chain with alternating hopping amplitudes
:math:`v` (intracell) and :math:`w` (intercell). It is the standard
introduction to bulk-boundary correspondence.

Bulk bands and the Zak phase
------------------------------

:func:`physicskit.condensed.models.ssh_hamiltonian` gives the 2x2 Bloch
Hamiltonian. Its gap closes only at :math:`v = w`; away from that point,
the chain is in one of two gapped phases distinguished by the
:func:`physicskit.condensed.topology.zak_phase`, quantized by chiral
symmetry to :math:`0` or :math:`\pi`:

.. code-block:: python

   from physicskit.condensed.models import ssh_hamiltonian
   from physicskit.condensed.topology import zak_phase

   trivial = lambda k: ssh_hamiltonian(k, v=1.0, w=0.5)       # v > w
   topological = lambda k: ssh_hamiltonian(k, v=0.5, w=1.0)   # v < w

   print(zak_phase(trivial))       # ~0
   print(zak_phase(topological))   # ~pi

Cutting the chain open
------------------------

Bulk topology alone is not directly observable. The signature of a
nontrivial Zak phase appears at the *boundary*, once translational
symmetry is broken. :func:`physicskit.condensed.models.ssh_lattice_hamiltonian`
builds the same chain as a real-space
:class:`~physicskit.condensed.tight_binding.Hamiltonian`, and
:func:`physicskit.condensed.tight_binding.build_ribbon` truncates it into
a finite, open wire:

.. code-block:: python

   import numpy as np
   from physicskit.condensed.models import ssh_lattice_hamiltonian
   from physicskit.condensed.tight_binding import build_ribbon

   H = ssh_lattice_hamiltonian(v=0.5, w=1.0)
   H_wire = build_ribbon(H, open_direction=0, n_cells=30)
   spectrum = np.linalg.eigvalsh(H_wire(np.array([])))
   print(np.abs(spectrum).min())  # ~1e-9: a mid-gap, near-zero mode

In the topological phase (:math:`v < w`), a pair of eigenvalues sit
exponentially close to zero energy, split only by the finite length of
the chain -- one state localized at each end. In the trivial phase
(:math:`v > w`), no such state exists; the spectrum is fully gapped.

Visualizing the edge state
-----------------------------

:func:`physicskit.condensed.visualizers.plot_edge_state_density` plots the
probability density of the closest-to-zero eigenstate, revealing the
expected exponential localization :math:`|\psi(x)|^2 \sim e^{-2x/\xi}` at
the chain's boundary:

.. code-block:: python

   from physicskit.condensed.visualizers import plot_edge_state_density

   fig, ax, density = plot_edge_state_density(H_wire, k_parallel=[])

See Also
--------

- :doc:`graphene_and_haldane` for the 2D analog (Chern insulators).
- :func:`physicskit.condensed.models.kitaev_chain_bdg_real_space` for the
  superconducting cousin of this construction: Majorana zero modes.
