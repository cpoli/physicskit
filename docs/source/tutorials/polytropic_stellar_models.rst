:orphan:

Polytropic Stellar Models and the Lane-Emden Equation
==========================================================

This tutorial uses :mod:`physicskit.astro.stellar_structure` to build
simple self-gravitating stellar models from the Lane-Emden equation, and
to check two of astrophysics' best-known mass scales.

Solving the Lane-Emden equation
------------------------------------

A polytrope -- a self-gravitating gas sphere with equation of state
:math:`P=K\rho^{1+1/n}` -- has a dimensionless hydrostatic-equilibrium
profile :math:`\theta(\xi)` governed by the Lane-Emden equation, solved
by :func:`~physicskit.astro.stellar_structure.lane_emden`:

.. code-block:: python

   from physicskit.astro.stellar_structure import lane_emden

   for n in [0.0, 1.0, 1.5, 3.0]:
       xi, theta = lane_emden(n)
       print(n, xi[-1])
   # 0.0 2.449...  (exact: sqrt(6))
   # 1.0 3.1416...  (exact: pi)
   # 1.5 3.6538...
   # 3.0 6.8968...

The point where :math:`\theta` first reaches zero is the star's surface
(density hits zero there); larger polytropic index ``n`` -- a softer,
more centrally-concentrated equation of state -- pushes that dimensionless
surface radius further out. The :math:`n=0` (uniform density) and
:math:`n=1` cases have simple closed forms,
:math:`\theta=1-\xi^2/6` and :math:`\theta=\sin\xi/\xi`, matching the
numerical surfaces :math:`\sqrt6` and :math:`\pi` exactly -- a good
sanity check that the solver is correct.

Building a physical star
------------------------------

:class:`~physicskit.astro.stellar_structure.PolytropicStar` turns a
Lane-Emden solution into a physical star, given a polytropic constant
``K`` and central density ``rho_c``:

.. code-block:: python

   from physicskit.astro.stellar_structure import PolytropicStar

   star = PolytropicStar(n=1.5, K=1.0, rho_c=1.0)
   print(star.radius)  # 1.6297  (n=1.5 is a reasonable model for a fully
   print(star.mass)    # 3.0279  # convective star, e.g. a low-mass red dwarf)

:math:`n=1.5` is the classic model for a fully convective star; ``n=3``
(the Eddington standard model, and the same index behind the
Chandrasekhar mass below) gives a more centrally concentrated star of the
same central density and polytropic constant:

.. code-block:: python

   star3 = PolytropicStar(n=3.0, K=1.0, rho_c=1.0)
   print(star3.radius, star3.mass)  # 3.891  4.557 -- bigger and more massive
                                     # at the same rho_c, K

The Chandrasekhar mass and the mass-luminosity relation
--------------------------------------------------------------

An :math:`n=3` polytrope supported by relativistic electron degeneracy
pressure -- a white dwarf -- has a maximum possible mass independent of
its central density, the Chandrasekhar limit,
:func:`~physicskit.astro.stellar_structure.chandrasekhar_mass`:

.. code-block:: python

   from physicskit.astro.stellar_structure import chandrasekhar_mass, main_sequence_luminosity

   print(chandrasekhar_mass(mu_e=2.0))  # 1.4575 solar masses

matching the textbook value of roughly 1.4 solar masses for a
carbon-oxygen white dwarf (:math:`\mu_e=2`) -- above this mass, electron
degeneracy pressure can no longer hold the star up against its own
gravity, and it either collapses further or, if enough mass accretes onto
it from a companion, detonates as a Type Ia supernova.

For ordinary main-sequence stars, luminosity climbs steeply with mass --
:func:`~physicskit.astro.stellar_structure.main_sequence_luminosity`
implements the empirical :math:`L\propto M^{3.5}` scaling:

.. code-block:: python

   print(main_sequence_luminosity(1.0))    # 1.0 (the Sun, by definition)
   print(main_sequence_luminosity(10.0))   # 3162 -- a 10-solar-mass star is
                                            # over 3000x as luminous

This steep scaling is why massive stars burn through their nuclear fuel
so much faster than the Sun, despite starting with more of it -- and
therefore live dramatically shorter lives.

See Also
--------

- :doc:`/history/astro_breakthroughs`
- :doc:`/api/astro`
