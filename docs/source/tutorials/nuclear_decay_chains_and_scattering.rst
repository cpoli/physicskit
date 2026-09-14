Radioactive Decay Chains, Rutherford Scattering, and Nuclear Binding
=========================================================================

This tutorial covers three classic pieces of nuclear physics with
:mod:`physicskit.particle.decays`, :mod:`physicskit.particle.scattering`,
and :mod:`physicskit.particle.nuclear`: how a chain of successive
radioactive decays evolves, how alpha particles scatter off a nucleus,
and why iron sits at the peak of nuclear stability.

A three-species decay chain
--------------------------------

A radioactive chain :math:`1\to2\to3` -- an unstable parent decaying to
an unstable daughter, which itself decays to a stable granddaughter --
is governed by the Bateman equations, implemented directly by
:func:`~physicskit.particle.decays.bateman_decay_chain`:

.. code-block:: python

   import numpy as np
   from physicskit.particle.decays import bateman_decay_chain

   t = np.array([0.0, 2.0, 5.0, 10.0, 20.0])
   N = bateman_decay_chain(N0=1000.0, decay_constants=[0.5, 0.2, 0.05], t=t)
   print(np.round(N, 1))
   # [[1000.     367.9     82.1      6.7      0. ]
   #  [   0.     504.1    476.3    214.3     30.5]
   #  [   0.     123.4    397.1    602.8    504.3]]

Each row is one species' population over time: the parent (row 0) decays
away monotonically; the daughter (row 1) first *builds up* as it's fed by
the parent, then decays away itself once the parent is nearly gone; the
granddaughter (row 2) keeps accumulating population throughout, since it
is treated as stable in this example (its own decay constant, 0.05, is
just slow on this timescale). Species 1 alone, by construction, matches
the plain exponential law
:func:`~physicskit.particle.decays.radioactive_decay_number` exactly --
the whole chain machinery reduces to the simple case when there's nothing
downstream feeding back.

Rutherford scattering off a gold nucleus
---------------------------------------------

Geiger and Marsden's classic experiment fired alpha particles
(:math:`Z_1=2`) at a thin gold foil (:math:`Z_2=79`).
:func:`~physicskit.particle.scattering.rutherford_dsigma_domega`
reproduces the steep angular falloff Rutherford predicted:

.. code-block:: python

   from physicskit.particle.scattering import rutherford_dsigma_domega, impact_parameter

   theta = np.radians([10, 30, 60, 90])
   dsigma = rutherford_dsigma_domega(theta, Z1=2, Z2=79, E_kin=5.0)
   print(dsigma)
   # [5.760e+01 7.406e-01 5.317e-02 1.329e-02]

The cross section at 10 degrees is roughly 4000 times larger than at 90
degrees -- almost all alpha particles pass through with only a small
deflection, and only a rare few, at small impact parameter, scatter
sharply backward. That steep falloff obeys the model's exact
:math:`1/\sin^4(\theta/2)` law: multiplying it back out gives the same
constant at every angle,

.. code-block:: python

   print(dsigma * np.sin(theta / 2) ** 4)
   # [0.00332342 0.00332342 0.00332342 0.00332342]  -- constant, as expected

and the corresponding classical impact parameters,
:func:`~physicskit.particle.scattering.impact_parameter`, shrink
smoothly as the scattering angle grows -- a closer pass means a sharper
deflection:

.. code-block:: python

   print(impact_parameter(theta, Z1=2, Z2=79, E_kin=5.0))
   # [1.318 0.430 0.200 0.115]

Where nuclear stability peaks
-----------------------------------

:func:`~physicskit.particle.nuclear.binding_energy_per_nucleon`
implements the semi-empirical (Weizsacker) mass formula, whose competing
volume, surface, Coulomb, and asymmetry terms combine to produce a broad
peak in stability around the iron group:

.. code-block:: python

   from physicskit.particle.nuclear import binding_energy_per_nucleon

   print(binding_energy_per_nucleon(2, 4))     # He-4:   5.71 MeV/nucleon
   print(binding_energy_per_nucleon(26, 56))   # Fe-56:  8.85 MeV/nucleon
   print(binding_energy_per_nucleon(82, 208))  # Pb-208: 7.86 MeV/nucleon

Iron-56 sits close to the top of this curve: light nuclei gain binding
energy per nucleon by fusing (the volume term dominates, not yet offset
by much surface energy), while heavy nuclei lose it by fissioning (the
Coulomb repulsion between an ever-larger number of protons eventually
wins) -- the same competition that makes stellar fusion stop producing
net energy once its core has burned up to the iron group.

See Also
--------

- :doc:`/history/particle_breakthroughs`
- :doc:`/api/particle`
