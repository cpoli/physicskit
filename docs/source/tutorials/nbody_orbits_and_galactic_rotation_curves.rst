N-Body Orbits, Hohmann Transfers, and Galactic Rotation Curves
====================================================================

This tutorial covers three scales of gravitational dynamics with
:mod:`physicskit.astro.nbody`, :mod:`physicskit.astro.orbital_mechanics`,
and :mod:`physicskit.astro.galactic_dynamics`: a symplectically-integrated
two-body orbit, an interplanetary transfer maneuver, and the flat
rotation curve that first pointed astronomers toward dark matter.

A symplectically-integrated circular orbit
------------------------------------------------

:class:`~physicskit.astro.nbody.NBodySystem` advances any number of
gravitating bodies with a leapfrog (symplectic) integrator -- the same
family of integrator prized elsewhere in this library for long-term
orbital stability. Two equal masses on a circular orbit around their
common center of mass is the simplest nontrivial test:

.. code-block:: python

   import numpy as np
   from physicskit.astro.nbody import NBodySystem

   G, m, d = 1.0, 1.0, 2.0
   v = np.sqrt(G * m / (2 * d))  # circular-orbit speed for this separation
   positions = np.array([[d / 2, 0, 0], [-d / 2, 0, 0]])
   velocities = np.array([[0, v, 0], [0, -v, 0]])
   system = NBodySystem(positions, velocities, masses=np.array([m, m]), G=G)

   E0 = system.total_energy()
   period = 2 * np.pi * (d / 2) / v
   system.simulate(dt=period / 200, n_steps=2000)  # 10 full orbits
   E1 = system.total_energy()
   print((E1 - E0) / E0)  # ~1.6e-12 -- energy conserved to near machine precision

Ten complete orbits leave the total energy essentially unchanged --
exactly the long-term stability a symplectic integrator is chosen for,
where a naive (non-symplectic) integrator would accumulate a steady
energy drift over the same run.

A Hohmann transfer between two orbits
-------------------------------------------

:func:`~physicskit.astro.orbital_mechanics.hohmann_transfer` computes the
two burns of the most fuel-efficient two-impulse transfer between
circular orbits -- the maneuver actually used to send spacecraft between
planets. In units where 1 AU is the length unit, 1 year is the time unit,
and the Sun's gravitational parameter is :math:`\mu_\odot=4\pi^2`, an
Earth-to-Mars transfer (:math:`r_1=1\,{\rm AU}`,
:math:`r_2=1.524\,{\rm AU}`) looks like:

.. code-block:: python

   from physicskit.astro.orbital_mechanics import hohmann_transfer

   mu_sun = 4 * np.pi ** 2
   dv1, dv2, transfer_time = hohmann_transfer(r1=1.0, r2=1.524, mu=mu_sun)
   print(dv1, dv2, transfer_time)
   # 0.6215 0.5590 0.7089  (speed changes in AU/yr, time in years)

A transfer time of roughly 0.71 years -- about 8.5 months -- matches the
real-world figure quoted for Hohmann-transfer trajectories to Mars.

Galactic rotation curves and dark matter
-----------------------------------------------

:func:`~physicskit.astro.galactic_dynamics.circular_velocity`, given any
enclosed-mass profile, gives the orbital speed a test particle needs to
stay on a circular orbit at each radius. For an ordinary (luminous, point
or centrally-concentrated) mass distribution this speed falls off as
:math:`1/\sqrt r` at large radius -- but for a Navarro-Frenk-White dark
matter halo, :func:`~physicskit.astro.galactic_dynamics.nfw_enclosed_mass`
keeps growing (logarithmically) with radius, and the two effects very
nearly cancel:

.. code-block:: python

   from physicskit.astro.galactic_dynamics import nfw_enclosed_mass, circular_velocity

   rho_s, r_s = 1.0, 1.0
   r = np.array([2, 5, 10, 20, 30]) * r_s
   v = circular_velocity(r, lambda rr: nfw_enclosed_mass(rr, rho_s, r_s))
   print(v)
   # [1.6474 1.552  1.3678 1.1465 1.0164]

Beyond a few scale radii, the rotation curve is nearly flat (the speed at
30 :math:`r_s` is still within about 16% of its value at 5 :math:`r_s`,
rather than falling off by a large factor as Keplerian rotation around a
fixed point mass would) -- exactly the puzzle Rubin and Ford's
measurements presented in the 1970s, resolved by exactly this kind of
extended, mostly-invisible mass distribution wrapped around every galaxy.

See Also
--------

- :doc:`/history/astro_breakthroughs`
- :doc:`/api/astro`
