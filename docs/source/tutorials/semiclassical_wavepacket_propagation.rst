:orphan:

Semiclassical Wavepacket Propagation
=======================================

This tutorial builds up the Van Vleck and Herman-Kluk semiclassical
propagators from a single classical trajectory to a full multi-trajectory
wavepacket propagation, checking each step against an exactly solvable
system (the harmonic oscillator) along the way.

A trajectory and its monodromy matrix
------------------------------------------

:func:`physicskit.semiclassical.core.propagators.propagate_trajectory_monodromy_action`
integrates a classical trajectory together with its **monodromy
matrix** -- the linearized map from small deviations in the initial
position and momentum to deviations in the final position and momentum
-- and its classical action, all in one Numba-compiled pass:

.. code-block:: python

   import numpy as np
   from numba import njit
   from physicskit.semiclassical.core.propagators import propagate_trajectory_monodromy_action

   m, omega = 1.0, 1.0
   params = np.array([m * omega ** 2])
   dVdx = njit(lambda q, params: params[0] * q)
   d2Vdx2 = njit(lambda q, params: params[0])
   V = njit(lambda q, params: 0.5 * params[0] * q ** 2)

   t = 1.3
   q_t, p_t, M, S, _ = propagate_trajectory_monodromy_action(
       q0=1.0, p0=0.3, dVdx=dVdx, d2Vdx2=d2Vdx2, V=V, m=m, dt=t / 4000, steps=4000, params=params
   )
   print(M)  # matches [[cos(t), sin(t)], [-sin(t), cos(t)]] to ~1e-6

Because the classical flow is Hamiltonian, :math:`\det M = 1` exactly
(Liouville's theorem) -- a good sanity check on any trajectory.

The Van Vleck propagator, checked against an exact result
------------------------------------------------------------

:func:`~physicskit.semiclassical.core.propagators.van_vleck_propagator_1d`
turns that trajectory into a semiclassical propagator amplitude
:math:`K(q_t,t;q_0,0)`. Because the harmonic oscillator's classical
action is exactly quadratic, the WKB expansion this formula comes from
truncates exactly, and the result matches the harmonic oscillator's
*exact* quantum (Mehler) propagator to within :math:`10^{-5}`:

.. code-block:: python

   from physicskit.semiclassical.core.propagators import van_vleck_propagator_1d

   q_t, K = van_vleck_propagator_1d(q0=1.0, p0=0.3, dVdx=dVdx, d2Vdx2=d2Vdx2, V=V, m=m, dt=t / 4000, steps=4000, params=params)
   K_exact = (
       np.sqrt(m * omega / (2j * np.pi * np.sin(omega * t)))
       * np.exp(1j * m * omega / (2 * np.sin(omega * t)) * ((q_t ** 2 + 1.0 ** 2) * np.cos(omega * t) - 2 * q_t * 1.0))
   )
   print(abs(K - K_exact))  # ~1e-6

Herman-Kluk: summing many frozen Gaussians
------------------------------------------------

A single trajectory connects one fixed start point to one fixed end
point. To propagate a whole *wavepacket*,
:func:`~physicskit.semiclassical.core.propagators.herman_kluk_propagate_wavepacket`
instead launches one classical trajectory -- each carrying a
fixed-width ("frozen") Gaussian -- from every point of a phase-space
grid covering the initial state, and sums their contributions:

.. code-block:: python

   from physicskit.semiclassical.core.propagators import (
       frozen_gaussian_1d, herman_kluk_propagate_wavepacket,
   )

   x = np.linspace(-6, 6, 400)
   psi_t = herman_kluk_propagate_wavepacket(
       qc0=1.0, pc0=0.0, gamma=1.0, dVdx=dVdx, d2Vdx2=d2Vdx2, V=V, m=m,
       dt=0.01, steps=100, x_eval=x, n_grid=61, n_sigma=7.0, params=params,
   )

At very short times this reduces to the frozen-Gaussian basis's
resolution of the identity and reconstructs the unpropagated initial
state almost exactly. Away from that limit, this particular
quadrature-grid evaluation of the Herman-Kluk integral does not exactly
conserve :math:`\int|\psi|^2\,dx` the way the true continuous
phase-space integral would for an at-most-quadratic potential --
:func:`physicskit.semiclassical.visualizers.propagators.plot_classical_trajectory_on_wigner`
is a good way to sanity-check a propagated state against the exact
Wigner function of a comparison calculation, and convergence in
``n_grid``, ``n_sigma``, and ``gamma`` should always be checked for
serious use.
