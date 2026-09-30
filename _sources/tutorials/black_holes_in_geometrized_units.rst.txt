Black Holes, Orbits and Ringdowns in Geometrized Units
======================================================

:mod:`physicskit.relativity` works in geometrized units,
:math:`G = c = 1`. Mass, length and time then share a single unit, so
the Schwarzschild radius is just :math:`2M` and the innermost stable
orbit is at :math:`6M`. Formulas become clean, but a result like
":math:`r = 6`" means nothing physical until you choose what
:math:`M` is. This tutorial solves a sequence of classic
general-relativity problems in :math:`M = 1` units and then converts
each answer to metres, seconds, arcseconds or hertz with
:func:`physicskit.units.geometrized_units`. Choosing :math:`M` is always
an explicit step, never a default.

Orbits around a Schwarzschild black hole
----------------------------------------

A test particle with specific angular momentum :math:`L` around a mass
:math:`M` moves in the effective potential

.. math::

   V_\text{eff}(r) = \left(1 - \frac{2M}{r}\right)\left(1 + \frac{L^2}{r^2}\right).

Unlike the Newtonian potential, :math:`V_\text{eff}` has a *maximum*
as well as a minimum. Setting :math:`dV/dr = 0` gives circular orbits
at :math:`r = \frac{L^2}{2M}\left(1 \pm \sqrt{1 - 12M^2/L^2}\right)`,
which for :math:`L = 4M` is :math:`r = 12M` (stable) and :math:`r = 4M`
(unstable):

.. code-block:: python

   import numpy as np
   import physicskit.relativity as rel

   bh = rel.SchwarzschildBlackHole(M=1.0)
   r = np.linspace(2.5, 30.0, 27501)
   V = bh.effective_potential(r, 4.0)
   turning = np.where(np.diff(np.sign(np.diff(V))) != 0)[0] + 1
   print("extrema of V_eff at r =", r[turning].round(3))
   # extrema of V_eff at r = [ 4. 12.]

   print(rel.KerrBlackHole(M=1.0, a=0.0).isco_radius())
   # 6.0

The two orbits merge when :math:`L^2 = 12M^2`, at :math:`r = 6M`. Inside
that radius, the innermost stable circular orbit (ISCO), no stable
circular orbits exist and matter plunges. This is the inner edge of an
accretion disk.

Perihelion precession, strong and weak
--------------------------------------

Bound orbits in general relativity don't close. Each periapsis moves
ahead by an angle that, far from the hole, is Einstein's 1915 result
:math:`\Delta\phi = 6\pi M/[a(1-e^2)]`. Integrating an actual geodesic
and measuring the advance with
:meth:`~physicskit.relativity.SchwarzschildBlackHole.perihelion_precession`
shows how that formula breaks down close to the hole:

.. code-block:: python

   y0 = bh.eccentric_orbit_initial_state(r0=40.0, eccentricity_boost=0.1)
   orbit = bh.integrate_geodesic(y0, dtau=0.5, n_steps=40000)
   measured = np.median(bh.perihelion_precession(orbit))

   r_peri, r_apo = orbit["r"].min(), orbit["r"].max()
   a, e = (r_peri + r_apo) / 2, (r_apo - r_peri) / (r_apo + r_peri)
   weak = bh.weak_field_precession_per_orbit(a, e)
   print(f"r = {r_peri:.2f}..{r_apo:.2f} M, measured {measured:.4f} rad/orbit, weak-field {weak:.4f}")
   # r = 25.95..40.00 M, measured 0.7002 rad/orbit, weak-field 0.5988

At :math:`r \approx 26\text{-}40\,M` the orbit advances by 0.70 rad per
revolution, 17% more than the leading-order formula predicts. The exact
value, from the elliptic integral
:math:`\Delta\phi = 2\int_{u_a}^{u_p} du/\sqrt{2Mu^3 - u^2 + 2Mu/L^2 +
(E^2-1)/L^2} - 2\pi` between the apsides :math:`u = 1/r`, is 0.7012,
so the geodesic integrator is within 0.15%. The small remaining error
comes from locating each periapsis on the discrete proper-time grid.

Converting to the Solar System: Mercury and the 1919 eclipse
------------------------------------------------------------

Now give :math:`M` a physical value. With :math:`M = M_\odot`, one
geometrized unit is :math:`GM_\odot/c^2` of length and
:math:`GM_\odot/c^3` of time, and
:func:`~physicskit.units.geometrized_units` does the conversion:

.. code-block:: python

   import physicskit.constants as const
   from physicskit.units import geometrized_units

   sun = geometrized_units(mass_kg=const.SOLAR_MASS_KG)
   print(f"1 M = {sun.scale('length'):.1f} m = {sun.scale('time') * 1e6:.3f} us")
   # 1 M = 1476.7 m = 4.926 us

   bh = rel.SchwarzschildBlackHole(M=1.0)                 # M = 1 is now the Sun
   a_mercury = sun.from_si(5.7909e10, "length")           # semi-major axis, in units of M
   dphi = bh.weak_field_precession_per_orbit(a_mercury, 0.2056)
   orbits_per_century = 36525 / 87.969
   print(f"Mercury: {np.degrees(dphi) * 3600 * orbits_per_century:.2f} arcsec/century")
   # Mercury: 42.98 arcsec/century

   deflection = bh.light_deflection_angle(sun.from_si(6.957e8, "length"))   # b = R_sun
   print(f"Sun-grazing light: {np.degrees(deflection) * 3600:.3f} arcsec")
   # Sun-grazing light: 1.751 arcsec

These are the two classic tests of general relativity. The first is the
43″/century anomaly in Mercury's orbit that Newtonian perturbation
theory could not explain (Le Verrier, 1859), which Einstein accounted
for in 1915. The second is the 1.75″ bending of starlight that
Eddington's 1919 eclipse expedition measured, twice the Newtonian
value. The weak-field formulas are accurate here because Mercury's orbit
is :math:`4\times10^7\,M` from the Sun.

Spinning black holes: how efficient is an accretion disk?
---------------------------------------------------------

For a Kerr black hole of spin :math:`a`, the prograde ISCO moves inward
(Bardeen, Press and Teukolsky, 1972). Gas that spirals slowly through a
thin disk down to the ISCO has radiated
:math:`1 - E_\text{ISCO}` of its rest-mass energy by the time it plunges.
Separately, the Penrose process can extract the hole's rotational
energy, up to :math:`1 - M_\text{irr}/M`:

.. code-block:: python

   for a in [0.0, 0.5, 0.9, 0.998]:
       bh = rel.KerrBlackHole(M=1.0, a=a)
       r_isco = bh.isco_radius(prograde=True)
       E_isco, _ = bh.circular_orbit_conserved_quantities(r_isco, prograde=True)
       print(f"a={a:<5}  r_isco = {r_isco:.3f} (retrograde {bh.isco_radius(prograde=False):.3f}),  "
             f"disk efficiency = {1 - E_isco:.1%},  Penrose max = {bh.max_penrose_efficiency():.1%}")

   # a=0.0    r_isco = 6.000 (retrograde 6.000),  disk efficiency = 5.7%,  Penrose max = 0.0%
   # a=0.5    r_isco = 4.233 (retrograde 7.555),  disk efficiency = 8.2%,  Penrose max = 3.4%
   # a=0.9    r_isco = 2.321 (retrograde 8.717),  disk efficiency = 15.6%,  Penrose max = 15.3%
   # a=0.998  r_isco = 1.237 (retrograde 8.994),  disk efficiency = 32.1%,  Penrose max = 27.1%

The non-spinning value :math:`1 - \sqrt{8/9} = 5.7\%` is already about
eight times the 0.7% that hydrogen fusion releases. Near-maximal spin
(:math:`a = 0.998`, Thorne's 1974 limit, set by the hole capturing
photons from its own disk) raises it to 32%. This is why accreting black
holes power the most luminous objects in the universe.

Gravitational-wave ringdown: GW150914 in hertz
----------------------------------------------

:class:`~physicskit.relativity.BinaryMerger` takes absolute masses,
because a waveform's amplitude depends on the physical scale. It uses
geometrized units with *one metre* as the unit of length, which is
``geometrized_units(length_m=1.0)``. The relativity subpackage's
:func:`~physicskit.relativity.utils.constants.solar_masses_to_geometrized`
gives the masses in those units. For the first detected merger,
GW150914 (36 + 29 solar masses at about 410 Mpc; LIGO/Virgo, *Phys.
Rev. Lett.* 116, 061102, 2016):

.. code-block:: python

   from physicskit.relativity.utils.constants import solar_masses_to_geometrized

   meters = geometrized_units(length_m=1.0)
   m1, m2 = solar_masses_to_geometrized(36.0), solar_masses_to_geometrized(29.0)
   merger = rel.BinaryMerger(m1, m2, distance=410e6 * const.PARSEC_M)

   M_f, a_f = merger.remnant_estimate()
   f_qnm, tau = merger.qnm_frequency_damping()
   print(f"remnant: {M_f / solar_masses_to_geometrized(1.0):.1f} Msun, spin a/M = {a_f / M_f:.2f}")
   print(f"ringdown: f = {meters.to_si(f_qnm, 'frequency'):.0f} Hz, tau = {meters.to_si(tau, 'time') * 1e3:.2f} ms")
   # remnant: 61.8 Msun, spin a/M = 0.68
   # ringdown: f = 276 Hz, tau = 3.71 ms

The remnant estimate is deliberately simple, but it lands close to
LIGO's inferred :math:`62\,M_\odot` and :math:`a/M = 0.67`. The
dominant quasinormal mode, from the Berti-Cardoso-Will (2006) fits,
rings at a few hundred hertz and decays within a few milliseconds, the
signal LIGO recorded at the end of the chirp. Because the conversion
goes through ``meters.to_si``, the factor of :math:`c` between a
geometrized frequency (in 1/m) and hertz cannot be forgotten.

Neutron stars and the maximum mass
----------------------------------

The Tolman-Oppenheimer-Volkoff equations describe a static star in
general relativity. Unlike Newtonian stars, TOV stars have a
*maximum* mass: beyond some central density, adding matter makes the
star less massive and unstable, so it collapses to a black hole.
:class:`~physicskit.relativity.NeutronStar` solves them for a polytrope
:math:`P = K\rho_0^\Gamma`. The standard numerical-relativity benchmark
:math:`K = 100`, :math:`\Gamma = 2` is quoted in units where
:math:`G = c = M_\odot = 1`, which is the same ``sun`` system as above:

.. code-block:: python

   star = rel.NeutronStar(K=100.0, Gamma=2.0)
   rho_c = np.linspace(0.5e-3, 5e-3, 60)
   masses, radii = star.mass_radius_curve(rho_c)
   i = np.argmax(masses)
   print(f"M_max = {masses[i]:.3f} Msun at R = {radii[i]:.2f} ({sun.to_si(radii[i], 'length') / 1e3:.1f} km), "
         f"rho_c = {rho_c[i]:.2e}")
   # M_max = 1.637 Msun at R = 7.64 (11.3 km), rho_c = 3.17e-03

:math:`M_\text{max} = 1.637\,M_\odot` is the benchmark's known maximum
mass, and it comes out in solar masses only because we chose
:math:`M_\odot` as the unit. The same :math:`K = 100` in a different
mass unit describes a different equation of state.

Where to go next
----------------

- :class:`~physicskit.relativity.PointMassLens` and the ray tracer in
  :mod:`physicskit.relativity.core.raytracer` turn the deflection angle
  into Einstein rings and black-hole shadows.
- :class:`~physicskit.relativity.FLRWCosmology` works in cosmological
  units (km/s/Mpc, Gyr) instead, for example ``FLRWCosmology().age_gyr()``.
- :doc:`units_and_conventions` explains the unit systems used across
  physicskit and how :mod:`physicskit.units` makes the choice of scale
  explicit.
- The relativity :doc:`history </history/relativity_breakthroughs>` covers
  each of these results, from Mercury to GW170817.
