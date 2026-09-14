:orphan:

Creating a Quantized Vortex in a Rotating BEC
================================================

This tutorial uses imaginary-time propagation of the Gross-Pitaevskii
equation (GPE) to find the ground state of a trapped Bose-Einstein
condensate, and then to demonstrate quantized vortex nucleation under
rotation -- the microscopic building block of an Abrikosov vortex lattice.

The non-rotating ground state
--------------------------------

Imaginary-time propagation (:math:`\tau=it`) turns the Schrodinger-like
GPE into a gradient-flow equation that relaxes any initial state toward a
stationary point of the energy. :func:`physicskit.fields.quantum_fields.gpe_relax`
implements this via a split-step scheme, renormalizing the wavefunction to
a fixed particle number after every step:

.. code-block:: python

   import numpy as np
   from physicskit.fields.quantum_fields import harmonic_trap_grid, gpe_relax, gpe_energy

   n, length, g = 64, 12.0, 4.0
   X, Y, KX, KY, K2 = harmonic_trap_grid(n, length)
   V = 0.5 * (X ** 2 + Y ** 2)

   psi0 = np.exp(-0.5 * (X ** 2 + Y ** 2)).astype(complex)
   psi_ground = gpe_relax(psi0, V, g, dtau=5e-4, steps=4000, X=X, Y=Y, K2=K2)

Imprinting a vortex
-----------------------

A singly-quantized vortex is a wavefunction with a :math:`2\pi` phase
winding around a point where the density vanishes.
:func:`physicskit.fields.quantum_fields.gpe_imprint_vortex` seeds one by
multiplying in a factor :math:`(x-x_0)+i(y-y_0)`:

.. code-block:: python

   from physicskit.fields.quantum_fields import gpe_imprint_vortex

   psi0_vortex = gpe_imprint_vortex(psi0, X, Y, [(0.0, 0.0)])
   psi_vortex = gpe_relax(psi0_vortex, V, g, dtau=5e-4, steps=4000, X=X, Y=Y, K2=K2)

Relaxation confirms this is a genuine stationary GPE solution: the phase
winding survives (checked with
:func:`physicskit.fields.quantum_fields.count_vortices`), and the density
vanishes exactly at the core, visible in
:func:`physicskit.fields.visualizers.plot_bec_density` and
:func:`~physicskit.fields.visualizers.plot_bec_phase`.

The critical rotation frequency
-----------------------------------

A vortex costs extra kinetic energy at the core, so at zero rotation the
vortex-free state has lower energy. In the *rotating* frame, however, the
vortex carries angular momentum :math:`\langle L_z\rangle \approx 1`
(in units where the vortex-free state has zero), so its rotating-frame
energy :math:`E - \Omega L_z` decreases faster with :math:`\Omega` than
the vortex-free state's. The two cross at the critical frequency

.. math::

   \Omega_c = \frac{E_{\text{vortex}} - E_{\text{vortex-free}}}{\langle L_z\rangle_{\text{vortex}} - \langle L_z\rangle_{\text{vortex-free}}},

above which nucleating the vortex is energetically favorable -- exactly
the textbook criterion for vortex nucleation in a rotating superfluid:

.. code-block:: python

   from physicskit.fields.quantum_fields import gpe_energy

   E0 = gpe_energy(psi_ground, V, g, X, Y, K2)
   E1 = gpe_energy(psi_vortex, V, g, X, Y, K2)
   Omega_c = (E1["total"] - E0["total"]) / (E1["angular_momentum"] - E0["angular_momentum"])
   print(Omega_c)  # ~0.88, comfortably below the trap frequency of 1.0

Multiple vortices, seeded at several positions via
``gpe_imprint_vortex(psi0, X, Y, [(x1, y1), (x2, y2), ...])`` and relaxed
the same way, are the building blocks of the Abrikosov-like vortex
lattices observed experimentally in rotating BECs.

See Also
--------

- :doc:`/history/fields_breakthroughs` for the history of the
  Gross-Pitaevskii equation and the first observed BEC vortex lattices.
