:orphan:

Reconstructing a Spectrum from a Single Periodic Orbit
=========================================================

This tutorial reconstructs the quantum energy spectrum of a bound 1D
system purely from the classical action and period of its one family of
periodic orbits -- the Gutzwiller trace formula -- and then looks at a
genuinely 2D chaotic system, the stadium billiard, where individual
eigenstates can still show a visible imprint of one particular unstable
orbit.

The exact 1D trace formula
-------------------------------

For a one-dimensional bound potential, Einstein-Brillouin-Keller (EBK)
quantization places energy levels exactly where the classical action
:math:`S(E)` (:func:`physicskit.semiclassical.core.wkb.wkb_action`)
satisfies :math:`S(E_n)/\hbar=(n+\tfrac12)\pi`. Poisson-summing that
discrete spectrum over the quantum number converts it into a sum over
classical orbit *repetitions* -- the one-dimensional Gutzwiller trace
formula, and (unlike the general multi-dimensional formula) exact:

.. code-block:: python

   import numpy as np
   from scipy.signal import find_peaks
   from physicskit.semiclassical.core.gutzwiller import gutzwiller_density_of_states
   from physicskit.semiclassical.core.wkb import bohr_sommerfeld_energies

   V = lambda x: 0.5 * x ** 2
   E_grid = np.linspace(0.2, 4.5, 600)
   dos = gutzwiller_density_of_states(E_grid, V, m=1.0, x_min=-20, x_max=20)

   peak_idx, _ = find_peaks(dos, height=0.3 * dos.max())
   print(E_grid[peak_idx])  # [0.5, 1.5, 2.5, 3.5]
   print(bohr_sommerfeld_energies(V, m=1.0, x_min=-20, x_max=20, n_max=4))  # matches exactly

:func:`physicskit.semiclassical.visualizers.gutzwiller.plot_density_of_states`
plots ``dos`` against ``E_grid`` with the exact levels marked, showing
each oscillating term of the sum sharpening the reconstructed peaks as
more orbit repetitions (``r_max``) are included.

Quantum scars in the stadium billiard
-------------------------------------------

In a genuinely chaotic system, periodic orbits are isolated and
unstable rather than forming a continuous family, and the general
(multi-dimensional) Gutzwiller formula applies instead --
:func:`physicskit.semiclassical.core.gutzwiller.gutzwiller_amplitude_from_monodromy`
gives the corresponding stability-weighted amplitude,
:math:`1/\sqrt{|2-\operatorname{tr}M|}`, for any such orbit's monodromy
matrix. The Bunimovich stadium billiard
(:class:`physicskit.quantum.chapters.potentials.StadiumBilliard2D`) is
the classic testbed: almost every trajectory disperses chaotically off
its curved end-caps, but the family of "bouncing ball" orbits --
bouncing straight up and down between the flat top and bottom walls --
is only marginally unstable, and Heller (1984) found that several
low-lying eigenstates concentrate their density visibly along exactly
this family:

.. code-block:: python

   from physicskit.quantum.chapters.potentials import StadiumBilliard2D
   from physicskit.semiclassical.systems.scarring import (
       bouncing_ball_energies, bouncing_ball_orbit_points, scar_enhancement,
   )

   sb = StadiumBilliard2D(L=1.0, R=0.5)
   energies, wavefunctions, X, Y, mask = sb.solve(n_points=90, n_states=6)

   print(bouncing_ball_energies(R=0.5, n_max=3))  # leading-order prediction

   x0 = 0.0
   for psi in wavefunctions:
       eta = scar_enhancement(psi ** 2, X, Y, mask, x0=x0, half_width=0.05)
       print(eta)  # > 1 for a state scarred on the x0 bouncing-ball orbit

:func:`physicskit.semiclassical.visualizers.scarring.plot_scar_map` heatmaps
any eigenstate's density with
:func:`~physicskit.semiclassical.systems.scarring.bouncing_ball_orbit_points`
drawn on top, making the enhancement directly visible; the interactive
:func:`~physicskit.semiclassical.visualizers.scarring.plot_scar_map_interactive`
is useful for zooming into the fine structure of a strongly scarred state.
