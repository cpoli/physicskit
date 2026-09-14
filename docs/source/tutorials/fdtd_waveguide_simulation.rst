:orphan:

Simulating Wave Propagation with FDTD
========================================

This tutorial builds up the finite-difference time-domain (FDTD) method on
a Yee grid, from a single 1D pulse to a 2D field with an absorbing boundary.

A pulse at the speed of light
--------------------------------

:func:`physicskit.fields.electrodynamics.fdtd_1d` implements the leapfrog
Yee update for a 1D plane wave. Launching a smooth Gaussian pulse with an
impedance-matched initial condition (so it travels purely in one
direction) confirms the scheme reproduces the vacuum speed of light:

.. code-block:: python

   import numpy as np
   from physicskit.fields.electrodynamics import fdtd_1d, courant_limit_1d, C0, EPS0, MU0

   N, dx = 800, 1e-3
   dt = 0.99 * courant_limit_1d(dx)
   eta0 = np.sqrt(MU0 / EPS0)
   x0, sigma = 100, 25
   Ez0 = np.exp(-((np.arange(N) - x0) ** 2) / (2 * sigma ** 2))
   xh = np.arange(N - 1) + 0.5
   Hy0 = np.exp(-((xh - x0) ** 2) / (2 * sigma ** 2)) / eta0
   eps_r, mu_r = np.ones(N), np.ones(N)

   Ez, Hy = fdtd_1d(Ez0, Hy0, eps_r, mu_r, steps=360, dt=dt, dx=dx)
   peak = np.argmax(Ez)
   print((peak - x0) * dx / (360 * dt) / C0)  # ~1.0

A single sharp point source, by contrast, contains high-frequency content
that FDTD schemes propagate at the wrong speed (numerical dispersion) --
always launch pulses several grid cells wide.

Absorbing the boundary instead of reflecting it
---------------------------------------------------

By default, :func:`~physicskit.fields.electrodynamics.fdtd_1d` terminates
the grid with a hard electric wall (PEC), which reflects outgoing waves
completely. :func:`physicskit.fields.electrodynamics.pml_conductivity_profile`
builds a graded-conductivity layer -- a simplified, non-split-field
approximation to Berenger's Perfectly Matched Layer -- that instead damps
the field smoothly to zero:

.. code-block:: python

   from physicskit.fields.electrodynamics import pml_conductivity_profile

   sigma = pml_conductivity_profile(N, pml_width=60, dx=dx)
   Ez_absorbed, Hy_absorbed = fdtd_1d(Ez0, Hy0, eps_r, mu_r, steps=1600, dt=dt, dx=dx, sigma=sigma)

Compare the peak field magnitude over time against the ``sigma=None``
(hard-wall) case: the hard wall keeps the pulse's energy exactly, while
the graded layer's peak field visibly decays as the pulse is absorbed at
the boundary.

Two dimensions and the Poynting vector
------------------------------------------

:func:`physicskit.fields.electrodynamics.fdtd_2d_tmz` extends the same
leapfrog update to a 2D TMz-mode field ``(Ez, Hx, Hy)``, and
:func:`physicskit.fields.electrodynamics.poynting_vector_tmz` computes the
resulting energy flux :math:`\mathbf{S} = \mathbf{E}\times\mathbf{H}`,
which :func:`physicskit.fields.visualizers.plot_poynting_field` renders as
a quiver plot -- useful for reading off the direction of power flow
through a simulated waveguide or cavity.

See Also
--------

- :doc:`/history/fields_breakthroughs` for the history of the Yee/FDTD
  algorithm and Berenger's PML.
