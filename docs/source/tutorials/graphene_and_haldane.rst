:orphan:

Graphene and the Haldane Model
===============================

This tutorial builds up from the semimetallic Dirac cones of graphene to
the first Chern insulator: Haldane's 1988 model of the quantum anomalous
Hall effect without a net magnetic field.

Graphene's Dirac cones
-----------------------

:func:`physicskit.condensed.models.graphene_hamiltonian` is the nearest-neighbor
tight-binding Bloch Hamiltonian on the honeycomb lattice, in the reduced
crystal-momentum convention used throughout :mod:`physicskit.condensed`
(see :mod:`physicskit.condensed.tight_binding`). Its two bands touch
linearly at the corners of the Brillouin zone:

.. code-block:: python

   import numpy as np
   from physicskit.condensed.models import graphene_hamiltonian

   K = np.array([2 * np.pi / 3, 4 * np.pi / 3])
   print(np.linalg.eigvalsh(graphene_hamiltonian(*K)))       # ~[0, 0]
   print(np.linalg.eigvalsh(graphene_hamiltonian(*(K + [1e-3, 0]))))  # +-0.00087

The vanishing gap at ``K`` and its time-reversed partner is protected by
inversion and time-reversal symmetry together. Breaking either one opens a gap.

Breaking time reversal: the Haldane model
-------------------------------------------

:func:`physicskit.condensed.models.haldane_model` adds a complex
next-nearest-neighbor hopping :math:`t_2 e^{i\phi}`, circulating in
opposite senses on the two sublattices. This breaks time-reversal symmetry
*without* any net magnetic flux through the unit cell -- the hallmark of
the quantum anomalous Hall effect:

.. code-block:: python

   from physicskit.condensed.models import haldane_model
   from physicskit.condensed.topology import compute_chern_number

   H = lambda k1, k2: haldane_model(k1, k2, t=1.0, t2=0.2, phi=np.pi / 2, M=0.0)
   print(compute_chern_number(H, grid_size=30))  # [1, -1]

The lower band now carries Chern number :math:`C = +1`: a topological
invariant that cannot change under any smooth, gap-preserving deformation
of the Hamiltonian. Adding a large enough sublattice mass ``M`` closes and
reopens the gap in a *trivial* way, driving the system back to
:math:`C = 0`:

.. code-block:: python

   H_trivial = lambda k1, k2: haldane_model(k1, k2, t=1.0, t2=0.2, phi=np.pi / 2, M=2.0)
   print(compute_chern_number(H_trivial, grid_size=30))  # [0, 0]

Visualizing the Berry curvature
---------------------------------

The Chern number is the integral of the Berry curvature over the
Brillouin zone. :func:`physicskit.condensed.topology.compute_berry_curvature`
exposes the curvature field itself, which
:func:`physicskit.condensed.visualizers.plot_berry_curvature` renders as a
heatmap -- the curvature concentrates near the (former) Dirac points:

.. code-block:: python

   from physicskit.condensed.topology import compute_berry_curvature
   from physicskit.condensed.visualizers import plot_berry_curvature

   F = compute_berry_curvature(H, grid_size=40, band_index=0)
   fig, ax = plot_berry_curvature(F)

See Also
--------

- :doc:`ssh_topological_edge_states` for the 1D analog (SSH chain).
- :doc:`/history/condensed_breakthroughs` for the historical context of the
  TKNN invariant and Haldane's 1988 construction.
