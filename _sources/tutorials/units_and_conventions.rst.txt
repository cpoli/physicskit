Units and Conventions
=====================

physicskit doesn't force one unit system on every field. Each subpackage
computes in the units its own literature uses, so its formulas look like
the ones in the papers you are checking against: :math:`r_\text{ISCO} = 6M`
rather than :math:`6GM/c^2`, and :math:`E_n = n + \tfrac12` rather than
:math:`(n + \tfrac12)\hbar\omega`. The cost is that a bare number such as
``6.0`` only has a physical meaning once you say what the unit is. This
page lists the convention of each subpackage, and shows how
:mod:`physicskit.units` makes the missing scale choice explicit, converts
results to SI and back, and keeps the units attached when results are
saved.

Conventions by subpackage
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Subpackage
     - Convention
     - What you still have to choose
   * - :mod:`~physicskit.astro`
     - gravitational, :math:`G = 1`
     - a length *and* a mass scale (e.g. 1 pc and 1 :math:`M_\odot`);
       :func:`~physicskit.units.astro_units`
   * - :mod:`~physicskit.relativity`
     - geometrized, :math:`G = c = 1`
     - one scale, conventionally the mass :math:`M`;
       :func:`~physicskit.units.geometrized_units`. The black-hole classes
       use :math:`M = 1`; :class:`~physicskit.relativity.BinaryMerger`
       uses 1 m as the length unit (``geometrized_units(length_m=1.0)``);
       :class:`~physicskit.relativity.FLRWCosmology` works in km/s/Mpc
       and Gyr.
   * - :mod:`~physicskit.quantum`
     - :math:`\hbar = 1` (usually with :math:`m = 1`)
     - a mass and a length (or energy) scale;
       :func:`~physicskit.units.quantum_units`
   * - :mod:`~physicskit.statphys`
     - :math:`k_B = 1`, couplings :math:`J = 1`
     - an energy (equivalently temperature) scale; also a length and mass
       for molecular dynamics; :func:`~physicskit.units.statphys_units`
   * - :mod:`~physicskit.plasma`
     - SI for single-particle, MHD and wave functions; normalized
       (:math:`\omega_{pe} = \lambda_D = 1`) for the PIC code
     - nothing for the SI functions; a density and temperature to give
       the PIC units physical meaning
   * - :mod:`~physicskit.fluids`
     - any consistent system you pass in; idealized problems are
       non-dimensional
     - your own units
   * - :mod:`~physicskit.rmt`, :mod:`~physicskit.chaos`
     - dimensionless
     - nothing
   * - :mod:`~physicskit.classical`, :mod:`~physicskit.condensed`,
       :mod:`~physicskit.fields`, :mod:`~physicskit.optics`,
       :mod:`~physicskit.particle`, :mod:`~physicskit.semiclassical`
     - constants are explicit parameters (``hbar=1.0``, ``g=9.81``, ...)
       or units are stated per function (e.g. GeV in parts of
       :mod:`~physicskit.particle`)
     - read the function's docstring

:mod:`physicskit.constants` is the single source of SI values (CODATA,
via :mod:`scipy.constants`) that all of this is built on.

Choosing the scale: :mod:`physicskit.units`
-------------------------------------------

Setting constants to 1 leaves some scales free: :math:`G = c = 1` leaves
one and :math:`G = 1` leaves two. Each preset in
:mod:`physicskit.units` takes the free scales as required keyword
arguments. None of them has a default, because a default would pick a
physical system for you without saying so:

.. code-block:: python

   import physicskit.constants as const
   from physicskit.units import geometrized_units

   sun = geometrized_units(mass_kg=const.SOLAR_MASS_KG)     # M = 1 solar mass
   print(f"{sun.scale('length'):.1f} m, {sun.scale('time') * 1e6:.3f} us")
   # 1476.7 m, 4.926 us

   print(f"ISCO: {sun.to_si(6.0, 'length') / 1e3:.2f} km")  # r = 6M
   # ISCO: 8.86 km

   print(f"1 ms = {sun.from_si(1.0e-3, 'time'):.1f} M")
   # 1 ms = 203.0 M

The same :math:`r = 6M` is 8.86 km for a solar-mass black hole and about
58 billion km (385 au) for the :math:`6.5\times10^9\,M_\odot` black hole
at the centre of M87. The code is identical; only the scale changes.

A dimension the scales don't determine raises an error instead of
returning a number. With :math:`k_B = 1` and only an energy scale, an
Ising-model temperature converts but a length doesn't:

.. code-block:: python

   from physicskit.units import statphys_units

   ising = statphys_units(temperature_k=300.0)   # coupling J = k_B * 300 K
   print(f"T_c = {ising.to_si(2.269, 'temperature'):.1f} K")
   # T_c = 680.7 K

   ising.to_si(1.0, "length")
   # ValueError: unit system 'statphys (k_B=1)' does not fix the scale of dimension L^1;
   # supply another scale (e.g. length_m or mass_kg) when building it

Choosing the electron mass and the Bohr radius for :math:`\hbar = 1`
gives Hartree atomic units, so a ground-state energy of :math:`-1/2`
becomes the hydrogen binding energy:

.. code-block:: python

   from physicskit.units import quantum_units

   au = quantum_units(mass_kg=const.ELECTRON_MASS, length_m=5.29177210903e-11)
   print(f"{const.joules_to_ev(au.to_si(0.5, 'energy')):.3f} eV")
   # 13.606 eV

Any other combination can be built with
:func:`~physicskit.units.natural_units`, for example particle-physics
units with :math:`\hbar = c = 1` and energies in GeV:

.. code-block:: python

   from physicskit.units import natural_units

   gev = natural_units("hbar", "c", energy_j=1e9 * const.ELECTRONVOLT)
   print(f"1/GeV = {gev.to_si(1.0, 'length') * 1e15:.4f} fm = {gev.to_si(1.0, 'time'):.3e} s")
   # 1/GeV = 0.1973 fm = 6.582e-25 s

How it works: a :class:`~physicskit.units.UnitSystem` stores *anchors*,
pairs of a :class:`~physicskit.units.Dimension` and the SI size of one
unit. Each constant set to 1 is one anchor and each chosen scale is
another. Solving the anchors in log space gives the SI size of one unit
of any dimension they determine. Anchors that contradict each other,
such as :math:`c = 1` together with unrelated length and time scales,
are rejected when the system is built.

Keeping units with results
--------------------------

:class:`physicskit.results.Result` holds a time series (or any array
output) together with its metadata, the dimension of each array, and the
unit system they are expressed in. The ``from_*`` adapters in
:mod:`physicskit.results` wrap the result types the subpackages already
return (integrator tuples, the classical ``SimulationResult``, quantum
``EigenResult``, rmt ``Spectrum``, chaos map orbits). You can also
build one directly:

.. code-block:: python

   import numpy as np
   import physicskit.relativity as rel
   from physicskit.results import Result

   bh = rel.SchwarzschildBlackHole(M=1.0)
   orbit = bh.integrate_geodesic(bh.circular_orbit_initial_state(10.0), dtau=0.5, n_steps=2000)

   result = Result(
       times=orbit["t"],
       states=np.column_stack([orbit["r"], orbit["phi"]]),
       metadata={"r0": 10.0},
       units={"times": "time"},
       unit_system=sun,
   )
   print(f"{result.times[-1]:.1f} M = {result.to_si().times[-1] * 1e3:.3f} ms")
   # 1195.2 M = 5.887 ms

:func:`physicskit.io.save` and :func:`~physicskit.io.load` write and
read the whole thing, unit system included, as ``.npz`` or (with
``pip install "physicskit[hdf5]"``) HDF5:

.. code-block:: python

   from physicskit import io

   io.save(result, "orbit.npz")
   back = io.load("orbit.npz")
   print(back, back.unit_system == sun)
   # Result(times=(2001,), states=(2001, 2), unit_system='relativity (G=c=1)') True

Interoperating with pint
------------------------

If you already use :mod:`pint`, install ``physicskit[units]`` and convert
at the boundary with :func:`~physicskit.units.to_pint` and
:func:`~physicskit.units.from_pint`. The dimension of a pint quantity is
read from the quantity itself:

.. code-block:: python

   import pint
   from physicskit.units import from_pint, to_pint

   ureg = pint.UnitRegistry()
   print(to_pint(6.0, "length", sun, registry=ureg).to("km"))
   # 8.860018146200808 kilometer

   print(f"{from_pint(1.0 * ureg.millisecond, sun):.1f}")
   # 203.0

Relation to older helpers
-------------------------

A few conversion helpers predate :mod:`physicskit.units` and remain
supported: :func:`physicskit.constants.gravitational_unit_system` (the
same numbers as :func:`~physicskit.units.astro_units`),
:func:`~physicskit.constants.energy_to_temperature` and the eV
converters, and :mod:`physicskit.relativity.utils.constants`
(``solar_masses_to_geometrized`` and friends, for the 1-metre geometrized
units used by :class:`~physicskit.relativity.BinaryMerger`). For anything
else, :mod:`physicskit.units` is the general tool.

Reference
---------

.. automodule:: physicskit.units
   :members:

.. automodule:: physicskit.results
   :members:

.. automodule:: physicskit.io
   :members:
