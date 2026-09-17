:orphan:

Wigner Functions and Squeezed Light
=======================================

This tutorial uses :mod:`physicskit.optics.quantum_optics` to compute the
Wigner quasi-probability distribution of a few standard quantum optical
states, and shows how it distinguishes classical light from two genuinely
quantum effects: negativity (a single photon) and squeezing (reduced
noise below the vacuum level in one quadrature).

A classical benchmark: the coherent state
----------------------------------------------

A coherent state :math:`\lvert\alpha\rangle` is the quantum state closest
to an ideal classical light wave: its Wigner function is a simple
non-negative Gaussian bump, centered away from the origin.

.. code-block:: python

   import numpy as np
   from physicskit.optics.quantum_optics import (
       coherent_state, fock_state, squeezed_state,
       compute_wigner_function, wigner_negativity,
   )

   x = np.linspace(-5, 5, 120)
   p = np.linspace(-5, 5, 120)

   psi_coh = coherent_state(1.5, cutoff=30)
   W_coh = compute_wigner_function(psi_coh, x, p)
   print(wigner_negativity(W_coh, x, p))  # ~0 -- indistinguishable from classical

:func:`~physicskit.optics.quantum_optics.wigner_negativity` integrates
:math:`|\min(W, 0)|` over phase space; for the coherent state it vanishes
to numerical precision, exactly as expected of a state with a classical
(non-negative) phase-space description.

A single photon: genuine negativity
------------------------------------------

The Fock state :math:`\lvert 1\rangle` has no classical analogue at all.
Its Wigner function dips *below zero* right at the phase-space origin --
a signature with no classical probability-distribution interpretation:

.. code-block:: python

   psi1 = fock_state(1, cutoff=30)
   W1 = compute_wigner_function(psi1, x, p)

   i0, j0 = np.argmin(np.abs(x)), np.argmin(np.abs(p))
   print(W1[i0, j0])            # ~ -0.31, close to the exact value -1/pi
   print(wigner_negativity(W1, x, p))  # ~0.21, decisively nonzero

The exact value at the origin is :math:`W_1(0,0) = -1/\pi \approx -0.318`;
the small discrepancy from the printed value is ordinary grid
discretization error on this 120-point sampling.

Squeezed light: noise below the vacuum limit
---------------------------------------------------

A squeezed vacuum state redistributes the vacuum's inherent quantum noise
unevenly between the two quadratures: it narrows the noise below the
standard vacuum (shot-noise) level in one quadrature, at the cost of
proportionally amplifying it in the other, keeping the Heisenberg product
fixed. :func:`~physicskit.optics.quantum_optics.squeezed_state` builds
this via the squeeze-then-displace convention; with squeezing parameter
:math:`\xi = r` real and positive, the :math:`x`-quadrature variance
shrinks by the standard factor :math:`e^{-2r}`:

.. code-block:: python

   def variance_x(W, x_grid, p_grid):
       marginal = np.trapezoid(W, p_grid, axis=1)
       marginal /= np.trapezoid(marginal, x_grid)
       mean = np.trapezoid(x_grid * marginal, x_grid)
       return np.trapezoid((x_grid - mean) ** 2 * marginal, x_grid)

   psi_vac = fock_state(0, cutoff=30)
   W_vac = compute_wigner_function(psi_vac, x, p)
   print(variance_x(W_vac, x, p))  # ~0.5, the vacuum (shot-noise) level

   r = 0.8
   psi_sq = squeezed_state(r, alpha=0.0, cutoff=40)
   W_sq = compute_wigner_function(psi_sq, x, p)
   print(variance_x(W_sq, x, p))          # ~0.101
   print(0.5 * np.exp(-2 * r))            # ~0.101 -- matches the e^{-2r} law

This is the effect Slusher and coworkers first observed experimentally in
1985 (see :doc:`/history/optics_breakthroughs`): noise pushed below the
vacuum level in one quadrature is exactly what makes squeezed light
useful for precision interferometry (including gravitational-wave
detectors), where the vacuum's own quantum fluctuations would otherwise
set the sensitivity floor.

Visualizing the phase-space surface
------------------------------------------

:func:`physicskit.optics.visualizers.interactive_wigner_surface` renders
any of these as an interactive 3D Plotly surface, making the single
photon's central dip and the squeezed state's elongated ellipse directly
visible:

.. code-block:: python

   from physicskit.optics.visualizers import interactive_wigner_surface

   fig = interactive_wigner_surface(W1, x, p, title="Single photon |1>")
   fig.show()

See Also
--------

- :doc:`/history/optics_breakthroughs` for Wigner's 1932 introduction of
  the phase-space quasi-probability distribution and Slusher's 1985
  observation of squeezed light.
- :doc:`/api/optics`
