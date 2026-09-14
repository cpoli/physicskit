Breakthroughs in General Relativity
===================================

.. epigraph::

   "Spacetime tells matter how to move; matter tells spacetime how to curve."
   -- John Archibald Wheeler

General relativity replaced Newton's instantaneous force of gravity with a
single geometric idea: mass and energy curve spacetime, and free-falling
objects simply follow the straightest paths available in that curved
geometry. From that one idea, worked out by Einstein over a decade of false
starts, followed black holes, an expanding universe, and ripples in
spacetime itself that would not be directly observed for another century.
The chronology behind :mod:`physicskit.relativity` traces that century, from
Einstein's 1915 field equations to the Event Horizon Telescope's first
image of a black hole's shadow in 2019, with a pointer to the corresponding
implementation in this package at each stop.

.. contents:: Timeline
   :local:
   :depth: 1

1915 -- Einstein's Field Equations
----------------------------------

After nearly a decade spent extending special relativity to include
gravity and acceleration, Albert Einstein presented the final form of the
general theory of relativity to the Prussian Academy of Sciences in
November 1915. Its centerpiece, the field equations,

.. math::

   G_{\mu\nu} \equiv R_{\mu\nu} - \frac{1}{2} R\, g_{\mu\nu} = 8\pi T_{\mu\nu},

relate the Einstein tensor :math:`G_{\mu\nu}` -- built from the curvature of
spacetime -- to the energy-momentum tensor :math:`T_{\mu\nu}` of whatever
matter and energy is present, in geometrized units :math:`G=c=1`. In
vacuum, where :math:`T_{\mu\nu}=0`, the equations reduce to
:math:`G_{\mu\nu}=0`: spacetime curves in response to nothing but its own
prior curvature, a self-consistency condition every metric in this package
must ultimately satisfy.

*Implementation:* :func:`physicskit.relativity.core.tensors.einstein_tensor`
computes :math:`G_{\mu\nu}` directly from an arbitrary metric function by
numerical differentiation, built on top of
:func:`~physicskit.relativity.core.tensors.christoffel_symbols`,
:func:`~physicskit.relativity.core.tensors.riemann_tensor`, and
:func:`~physicskit.relativity.core.tensors.ricci_tensor`; evaluating it on
the Schwarzschild metric and finding it numerically zero is this package's
own sanity check of Einstein's vacuum equation.

*References:* A. Einstein, "Die Feldgleichungen der Gravitation,"
Sitzungsberichte der Königlich Preußischen Akademie der Wissenschaften,
844-847 (25 Nov. 1915).

.. minigallery:: ../../examples/relativity/spacetime_geometry/plot_curvature_engine_validation.py

1915 -- Mercury's Perihelion Precession
---------------------------------------

Urbain Le Verrier noticed, as early as 1859, that Mercury's orbit precesses
about 43 arcseconds per century faster than Newtonian gravity -- accounting
for every known planetary perturbation -- could explain. The anomaly went
unexplained for over half a century; a hypothetical planet, "Vulcan",
orbiting inside Mercury to account for the extra tug, was searched for and
never found. On 18 November 1915, days before he finished the general
theory itself, Einstein computed the correction general relativity predicts
for a test mass orbiting a central body,

.. math::

   \Delta\phi = \frac{6\pi M}{a\,(1-e^2)}

per orbit, where :math:`a` is the orbit's semi-major axis and :math:`e` its
eccentricity. Plugging in Mercury's orbital elements reproduced the missing
43 arcseconds per century exactly. Einstein later told a colleague the
result gave him heart palpitations -- it was general relativity's first
successful confrontation with data, four years before Eddington's eclipse
expedition made the theory famous.

*Implementation:*
:meth:`physicskit.relativity.chapters.schwarzschild.SchwarzschildBlackHole.weak_field_precession_per_orbit`
computes exactly this closed-form precession;
:meth:`~physicskit.relativity.chapters.schwarzschild.SchwarzschildBlackHole.integrate_geodesic`
together with
:meth:`~physicskit.relativity.chapters.schwarzschild.SchwarzschildBlackHole.perihelion_precession`
recovers the same number the other way, by directly integrating an
eccentric geodesic from
:meth:`~physicskit.relativity.chapters.schwarzschild.SchwarzschildBlackHole.eccentric_orbit_initial_state`
and measuring the excess angle swept between successive perihelion
passages.

*References:* A. Einstein, "Erklärung der Perihelbewegung des Merkur aus
der allgemeinen Relativitätstheorie," Sitzungsberichte der Preußischen
Akademie der Wissenschaften, 831-839 (18 Nov. 1915).

.. minigallery:: ../../examples/relativity/schwarzschild/plot_orbit_precession.py

1916 -- Schwarzschild's Exact Solution
--------------------------------------

Weeks after Einstein's paper appeared -- and while serving as an artillery
officer on the Russian front in the First World War -- Karl Schwarzschild
found the first exact solution to the field equations: the spacetime
around a single non-rotating, spherically symmetric mass,

.. math::

   ds^2 = -\left(1-\frac{2M}{r}\right)dt^2 + \left(1-\frac{2M}{r}\right)^{-1}dr^2
        + r^2 d\theta^2 + r^2\sin^2\theta\, d\phi^2.

Schwarzschild sent his solution to Einstein, who was reportedly astonished
that an exact answer existed at all. The metric's coordinate singularity
at :math:`r=2M` -- initially dismissed as unphysical -- would only be
understood decades later as an event horizon, the boundary of a black hole.
Schwarzschild died of an autoimmune illness contracted at the front just a
few months after publishing it.

*Implementation:* :func:`physicskit.relativity.core.tensors.schwarzschild_metric`
is exactly this line element;
:class:`physicskit.relativity.chapters.schwarzschild.SchwarzschildBlackHole`
wraps it with the derived quantities Schwarzschild's solution predicts --
the ``horizon_radius`` :math:`r_s=2M`, ``photon_sphere_radius``
:math:`r_{ph}=3M`, and ``isco_radius`` :math:`r_{\text{ISCO}}=6M`.

*References:* K. Schwarzschild, "Über das Gravitationsfeld eines
Massenpunktes nach der Einsteinschen Theorie," Sitzungsberichte der
Königlich Preußischen Akademie der Wissenschaften, 189-196 (1916).

.. minigallery::
   ../../examples/relativity/spacetime_geometry/plot_flamm_paraboloid.py
   ../../examples/relativity/spacetime_geometry/plot_kruskal_penrose_diagrams.py

1916-1918 -- Einstein Predicts Gravitational Waves
---------------------------------------------------

Barely a year after finishing general relativity, Einstein linearized the
field equations about flat spacetime and found that they admit wave
solutions: small ripples in the metric that propagate at the speed of
light, sourced by a time-varying mass quadrupole moment and radiating away
energy at a rate

.. math::

   P \sim \frac{d^3 Q_{ij}}{dt^3}\frac{d^3 Q^{ij}}{dt^3},

the quadrupole formula. His first 1916 paper on the subject contained an
error -- a spurious factor of two from an incorrect gauge choice -- which
he corrected two years later in a second paper laying out the quadrupole
formula essentially as it is used today. Einstein himself remained
privately unsure whether the waves were physically real or merely a
coordinate artifact, a question not fully settled within the relativity
community until the 1950s and 1960s. It would take until 1974 for the
first (indirect) astrophysical evidence that they exist, and until 2015
for a direct detection.

*Implementation:* :meth:`physicskit.relativity.chapters.gw_merger.BinaryMerger.inspiral_frequency`
and :meth:`~physicskit.relativity.chapters.gw_merger.BinaryMerger.inspiral_strain`
implement exactly this leading-order (Newtonian quadrupole) radiation
formula for a binary source, the direct numerical descendant of Einstein's
corrected 1918 result.

*References:* A. Einstein, "Näherungsweise Integration der Feldgleichungen
der Gravitation," Sitzungsberichte der Königlich Preußischen Akademie der
Wissenschaften, 688-696 (1916); A. Einstein, "Über Gravitationswellen,"
Sitzungsberichte der Preußischen Akademie der Wissenschaften, 154-167
(1918) (correcting the 1916 paper's factor-of-two error).

.. minigallery:: ../../examples/relativity/gravitational_waves/plot_binary_merger_chirp.py

1919 -- The Eddington Eclipse Expedition
----------------------------------------

General relativity predicts that light passing near a massive body is
deflected by twice the angle Newtonian gravity (treating light as a
particle) would give,

.. math::

   \delta\phi \approx \frac{4M}{b},

for a photon with impact parameter :math:`b`. Frank Dyson, the Astronomer
Royal, organized two expeditions to observe the total solar eclipse of 29
May 1919 -- the only time stars close enough to the Sun's limb are visible
at all -- so that a result from one site could be checked against the
other. Arthur Eddington and Edwin Cottingham traveled to the island of
Principe, while Andrew Crommelin and Charles Davidson took a second set of
plates at Sobral, Brazil; contrary to the popular account that credits
Eddington alone, it was Crommelin and Davidson's Sobral data -- the sharper
and more numerous of the two plate sets -- that gave the more decisive
measurement. Both sites measured a deflection consistent with Einstein's
value, not Newton's, and Dyson, Eddington, and Davidson jointly published
the combined result. The announcement, later that year, made Einstein a
household name overnight and stands as general relativity's first
observational triumph.

*Implementation:* :meth:`physicskit.relativity.chapters.schwarzschild.SchwarzschildBlackHole.light_deflection_angle`
implements exactly this weak-field formula;
:func:`physicskit.relativity.chapters.lensing.exact_deflection_angle`
numerically integrates the full null geodesic for comparison, capturing the
strong-field corrections that become significant close to the photon
sphere, where the weak-field formula above breaks down.

*References:* F. W. Dyson, A. S. Eddington, and C. Davidson, "A
Determination of the Deflection of Light by the Sun's Gravitational
Field, from Observations Made at the Total Eclipse of May 29, 1919," Phil.
Trans. R. Soc. A 220, 291-333 (1920) (results announced Nov. 1919; the
paper itself appeared the following year).

.. minigallery:: ../../examples/relativity/schwarzschild/plot_light_bending_and_shadow.py

1922 -- Friedmann, Lemaitre, and the Expanding Universe
-------------------------------------------------------

Alexander Friedmann showed, in 1922, that Einstein's field equations admit
non-static, homogeneous, isotropic cosmological solutions -- in direct
contradiction to Einstein's own preference (at the time) for a static
universe, which had led him to introduce the cosmological constant
:math:`\Lambda` specifically to prevent expansion. Georges Lemaitre
independently rediscovered these solutions in 1927 and connected them to
the redshifts already being observed in distant galaxies, two years before
Hubble's own 1929 observational confirmation. The Friedmann equation,

.. math::

   H(a)^2 \equiv \left(\frac{\dot a}{a}\right)^2
       = H_0^2\left[\Omega_r a^{-4} + \Omega_m a^{-3} + \Omega_k a^{-2} + \Omega_\Lambda\right],

governs how the universe's scale factor :math:`a(t)` evolves under the
combined gravity of radiation, matter, curvature, and dark energy.

*Implementation:* :func:`physicskit.relativity.core.tensors.flrw_metric`
implements the FLRW line element itself;
:class:`physicskit.relativity.chapters.cosmology.FLRWCosmology` and its
:meth:`~physicskit.relativity.chapters.cosmology.FLRWCosmology.hubble_parameter`
method solve exactly this Friedmann equation, with
:meth:`~physicskit.relativity.chapters.cosmology.FLRWCosmology.luminosity_distance_mpc`
and :meth:`~physicskit.relativity.chapters.cosmology.FLRWCosmology.age_gyr`
built on top of it.

*References:* A. Friedmann, "Über die Krümmung des Raumes," Z. Phys. 10,
377-386 (1922); G. Lemaître, Annales de la Société Scientifique de
Bruxelles A47, 49-59 (1927); E. Hubble, "A Relation between Distance and
Radial Velocity among Extra-Galactic Nebulae," PNAS 15(3), 168-173 (1929).

.. minigallery:: ../../examples/relativity/cosmology/plot_flrw_expansion.py

1936 -- Einstein Rings and Gravitational Lensing
------------------------------------------------

Einstein had privately worked out, as early as 1912, that a massive body
could act as a gravitational lens, bending and magnifying the light of a
background source -- but he did not publish the result until 1936, and
even then considered it of little practical interest, since the angular
separations involved seemed hopelessly small for any real star to resolve.
When the observer, lens, and source line up exactly, the lensed image forms
a complete ring around the lens, of angular radius

.. math::

   \theta_E = \sqrt{\frac{4M\, D_{LS}}{D_L D_S}},

the Einstein radius. Any other alignment splits a single source into two
images of unequal brightness. Fritz Zwicky pointed out in 1937 that entire
galaxies, not just stars, could lens background sources strongly enough to
observe -- the first galaxy-scale lens was found in 1979, and the first
complete Einstein ring imaged in 1988.

*Implementation:* :class:`physicskit.relativity.chapters.lensing.PointMassLens`
and its :meth:`~physicskit.relativity.chapters.lensing.PointMassLens.einstein_angle`,
:meth:`~physicskit.relativity.chapters.lensing.PointMassLens.image_angles`,
and :meth:`~physicskit.relativity.chapters.lensing.PointMassLens.magnification`
methods solve exactly this thin-lens problem, including the Paczynski
microlensing magnification formula used in exoplanet and dark-object
surveys.

*References:* A. Einstein, "Lens-Like Action of a Star by the Deviation of
Light in the Gravitational Field," Science 84(2188), 506-507 (1936); F.
Zwicky, "Nebulae as Gravitational Lenses," Phys. Rev. 51, 290 (1937); first
observed lens: D. Walsh, R. F. Carswell, and R. J. Weymann, Nature 279,
381-384 (1979); first Einstein ring: J. N. Hewitt et al., Nature 333,
537-540 (1988).

.. minigallery:: ../../examples/relativity/lensing/plot_einstein_ring_and_microlensing.py

1939 -- Tolman-Oppenheimer-Volkoff and Neutron Star Structure
-------------------------------------------------------------

Richard Tolman, and then J. Robert Oppenheimer and George Volkoff, derived
the relativistic generalization of Newtonian hydrostatic equilibrium for a
self-gravitating fluid sphere -- the equation describing the interior
structure of a neutron star, matter compressed to nuclear density and held
up against collapse purely by neutron degeneracy pressure. Unlike the
Newtonian case, the TOV equation predicts a maximum possible mass: past a
critical central density, adding more matter only accelerates collapse,
and no degeneracy pressure can halt it -- the star must become a black
hole. This maximum mass (loosely, the "Tolman-Oppenheimer-Volkoff limit")
remains an active area of nuclear-equation-of-state research today.

*Implementation:* :class:`physicskit.relativity.chapters.neutron_star.NeutronStar`
and its :meth:`~physicskit.relativity.chapters.neutron_star.NeutronStar.solve`
method integrate exactly the TOV equations outward from the stellar center
to the surface;
:meth:`~physicskit.relativity.chapters.neutron_star.NeutronStar.mass_radius_curve`
traces out the resulting mass-radius relation and its maximum-mass turning
point.

*References:* R. C. Tolman, "Static Solutions of Einstein's Field Equations
for Spheres of Fluid," Phys. Rev. 55, 364-373 (1939); J. R. Oppenheimer and
G. M. Volkoff, "On Massive Neutron Cores," Phys. Rev. 55, 374-381 (1939).

.. minigallery:: ../../examples/relativity/neutron_star/plot_tov_mass_radius.py

1963 -- Kerr's Rotating Black Hole Solution
-------------------------------------------

For nearly fifty years after Schwarzschild, no one found an exact solution
describing a *rotating* black hole -- the astrophysically realistic case,
since essentially every collapsing star has some angular momentum. Roy
Kerr found one in 1963, characterized by just two parameters, mass
:math:`M` and spin :math:`a=J/M`. The Kerr solution predicts frame
dragging: spacetime itself is dragged around the rotating hole, so
strongly near the horizon that no observer, however powerful their engine,
can remain at fixed angular position within the ergosphere -- the region
:math:`r_+ < r < M+\sqrt{M^2-a^2\cos^2\theta}` outside the horizon where
this effect appears.

*Implementation:* :func:`physicskit.relativity.core.tensors.kerr_metric_bl`
implements the Kerr metric in Boyer-Lindquist coordinates;
:class:`physicskit.relativity.chapters.kerr.KerrBlackHole` and its
:meth:`~physicskit.relativity.chapters.kerr.KerrBlackHole.frame_dragging_angular_velocity`
and :meth:`~physicskit.relativity.chapters.kerr.KerrBlackHole.ergosphere_radius`
methods compute exactly these effects, and
:meth:`~physicskit.relativity.chapters.kerr.KerrBlackHole.isco_radius`
gives the spin-dependent innermost stable circular orbit.

*References:* R. P. Kerr, "Gravitational Field of a Spinning Mass as an
Example of Algebraically Special Metrics," Phys. Rev. Lett. 11, 237-238
(1963).

.. minigallery:: ../../examples/relativity/kerr/plot_ergosphere_and_penrose.py

1965 -- Penrose's Singularity Theorems
--------------------------------------

Schwarzschild's :math:`r=0` singularity and the FLRW big bang had both
appeared inside solutions with an enormous amount of built-in symmetry,
leading many, including Einstein himself, to suspect that singularities
were merely an artifact of that idealized symmetry -- an assumption that
real, lumpy, rotating collapsing matter would surely avoid. Roger Penrose
proved otherwise in 1965: using global, topological methods rather than
solving the field equations explicitly, he showed that once a *trapped
surface* forms -- a closed surface from which even outgoing light rays are
converging inward -- and ordinary energy conditions hold, at least one
causal geodesic in the resulting spacetime must be incomplete. No symmetry
assumption is needed. For a simple, spherically symmetric illustration of
that incompleteness, a particle dropped from rest at radius :math:`R`
reaches the singularity in finite proper time,

.. math::

   \tau_{r=0} = \frac{\pi}{2}\sqrt{\frac{R^3}{2M}},

rather than taking forever, as a merely coordinate-induced artifact would.
Stephen Hawking adapted the same methods to the time-reversed case the
following year, showing an initial big-bang singularity is likewise
unavoidable given only that the universe contains enough matter and is
expanding. Penrose received a share of the 2020 Nobel Prize in Physics "for
the discovery that black hole formation is a robust prediction of the
general theory of relativity."

*Implementation:*
:meth:`physicskit.relativity.chapters.schwarzschild.SchwarzschildBlackHole.eccentric_orbit_initial_state`
with ``eccentricity_boost=1.0`` starts a geodesic at rest -- zero angular
velocity -- at radius :math:`r_0`;
:meth:`~physicskit.relativity.chapters.schwarzschild.SchwarzschildBlackHole.integrate_geodesic`
evolves it inward and shows the elementary, coordinate-bound version of
the puzzle directly: the particle's own proper time :math:`\tau` climbs
smoothly and stays finite all the way to the horizon, while
:math:`dt/d\tau` grows without bound over the same interval -- Schwarzschild
coordinates are singular exactly at :math:`r=2M`, so this coordinate-time
integrator cannot be pushed past the horizon at all. Reaching the true
singularity needs the closed-form :math:`\tau_{r=0}` above, evaluated
directly from the same radial energy-conservation equation the integrator
solves, independent of which time coordinate is used to parametrize the
fall -- not a further numerical integration past the point where these
particular coordinates break down.

*References:* R. Penrose, "Gravitational Collapse and Space-Time
Singularities," Phys. Rev. Lett. 14, 57-59 (1965); S. W. Hawking,
"Singularities in the Universe," Proc. R. Soc. A 294, 511-521 (1966).

.. minigallery:: ../../examples/relativity/schwarzschild/plot_radial_infall_and_singularity.py

1969 -- Penrose's Energy-Extraction Process
-------------------------------------------

Roger Penrose showed that the ergosphere is not just a curiosity of
rotating spacetime but a genuine energy resource: inside it, the timelike
Killing vector associated with time-translation symmetry becomes
spacelike, so a particle there can have *negative* energy as measured by
an observer at infinity. If a particle entering the ergosphere splits in
two, with one negative-energy fragment falling into the horizon, energy
conservation forces the escaping fragment to carry away *more* energy than
the original particle had -- extracting rotational energy from the black
hole itself. The process is capped by Hawking's area theorem: the hole's
irreducible mass, :math:`M_{\text{irr}}=\sqrt{Mr_+/2}`, can never decrease,
limiting the maximum extractable fraction to about 29% for a maximally
spinning hole.

*Implementation:* :meth:`physicskit.relativity.chapters.kerr.KerrBlackHole.penrose_energy_gain`
computes the escaping fragment's energy gain for a given ergosphere split;
:meth:`~physicskit.relativity.chapters.kerr.KerrBlackHole.max_penrose_efficiency`
gives exactly this 29%-at-extremal-spin efficiency ceiling.

*References:* R. Penrose, "Gravitational Collapse: The Role of General
Relativity," Rivista del Nuovo Cimento, Numero Speciale I, 252-276 (1969).

.. minigallery:: ../../examples/relativity/kerr/plot_penrose_energy_extraction.py

1959-1962 -- The ADM (3+1) Formalism
-------------------------------------

Chronologically, the ADM formalism predates the Kerr and Penrose results
above; it is placed here instead because it belongs thematically with the
numerical-relativity thread this chronology picks back up at 1974 and,
decisively, in 2005. Richard Arnowitt, Stanley Deser, and Charles Misner
recast general relativity as an initial-value problem: instead of treating
spacetime as a single four-dimensional block satisfying the field
equations everywhere at once, they sliced it into a stack of spacelike
three-dimensional hypersurfaces, each carrying a spatial metric
:math:`\gamma_{ij}` and extrinsic curvature :math:`K_{ij}`, related to
their neighbors by a lapse function :math:`\alpha` and shift vector
:math:`\beta^i` that describe how the slicing threads through spacetime.
Einstein's ten field equations split into evolution equations for
:math:`\gamma_{ij}` and :math:`K_{ij}` plus four constraint equations
(Hamiltonian and momentum) that must hold on every slice. This "3+1"
decomposition -- three spatial dimensions evolving in a time coordinate --
is the starting point of essentially every numerical-relativity simulation
since, including the codes that produced the 2005 binary-black-hole-merger
breakthrough below and, decades later, the waveform templates behind
LIGO's detections.

*Connection:* :mod:`physicskit.relativity` does not itself implement a 3+1
evolution scheme -- every metric here (Schwarzschild, Kerr, FLRW) is
treated as a known analytic background rather than evolved numerically
from initial data, so there is no lapse/shift/extrinsic-curvature
machinery to point to. The package's role in this chronology begins one
step downstream of ADM, with the merger *outcomes* -- remnant mass, spin,
and ringdown -- that full numerical-relativity evolutions of the ADM
equations first computed; see the 2005 entry below and
:class:`physicskit.relativity.chapters.gw_merger.BinaryMerger`.

*References:* R. Arnowitt, S. Deser, and C. W. Misner, "The Dynamics of
General Relativity," in *Gravitation: An Introduction to Current
Research*, ed. L. Witten (Wiley, 1962), pp. 227-265; originally developed
in a series of Physical Review papers, e.g. R. Arnowitt, S. Deser, and
C. W. Misner, "Dynamical Structure and Definition of Energy in General
Relativity," Phys. Rev. 116, 1322 (1959).

.. minigallery:: ../../examples/relativity/gravitational_waves/plot_binary_merger_chirp.py

1974 -- The Hulse-Taylor Binary Pulsar
--------------------------------------

General relativity predicts that any accelerating mass quadrupole radiates
gravitational waves, carrying away orbital energy -- but the effect is
so weak that no laboratory source could ever hope to detect it directly.
Russell Hulse and Joseph Taylor found an astrophysical source strong
enough to reveal the effect indirectly: PSR B1913+16, a pulsar in a tight
orbit with another neutron star, discovered in 1974. Over the following
years, precise pulsar timing showed the pair's orbital period shrinking at
exactly the rate

.. math::

   \frac{dP}{dt} \propto -\frac{m_1 m_2 (m_1+m_2)}{a^{4}}

predicted by Peters' 1964 formula for gravitational-wave energy loss. The
pulsar was discovered in 1974; Taylor and Weisberg's landmark timing
analysis, published in 1982, already showed the orbital decay tracking
general relativity's prediction to about 1%, and later refinements of the
same pulsar-timing dataset -- Taylor and Weisberg again in 1989, and
Weisberg and Taylor's subsequent updates through the mid-2000s -- tightened
the agreement to the oft-quoted 0.2% figure. Together this stood as the
first evidence, albeit indirect, that gravitational waves are real. Hulse
and Taylor received the 1993 Nobel Prize in Physics for the discovery.

*Implementation:* :meth:`physicskit.relativity.chapters.gw_merger.BinaryMerger.semi_major_axis_decay_rate`
implements exactly this Peters-formula orbital shrinkage rate;
:meth:`~physicskit.relativity.chapters.gw_merger.BinaryMerger.period_decay_rate`
converts it to the observable :math:`dP/dt` that Hulse and Taylor measured.

*References:* R. A. Hulse and J. H. Taylor, "Discovery of a Pulsar in a
Binary System," ApJ Letters 195, L51-L53 (1975) (discovered 1974, published
1975); P. C. Peters, "Gravitational Radiation and the Motion of Two Point
Masses," Phys. Rev. 136, B1224-B1232 (1964); J. H. Taylor and J. M.
Weisberg, "A New Test of General Relativity: Gravitational Radiation and
the Binary Pulsar PSR 1913+16," ApJ 253, 908-920 (1982) (~1% agreement);
J. H. Taylor and J. M. Weisberg, "Further Experimental Tests of Relativistic
Gravity Using the Binary Pulsar PSR 1913+16," ApJ 345, 434-450 (1989) and
subsequent Weisberg & Taylor updates (~2004-2010) for the tighter ~0.2%
figure.

.. minigallery:: ../../examples/relativity/gravitational_waves/plot_hulse_taylor_decay.py

1976 -- Gravity Probe A and Relativistic Timekeeping
----------------------------------------------------

General relativity predicts that a clock deeper in a gravitational
potential runs slower than one farther out -- gravitational time dilation.
Robert Vessot and Martin Levine put this to its most precise test yet in
June 1976, launching a hydrogen maser clock on a suborbital rocket to
10,000 km altitude and comparing its rate against an identical clock on the
ground throughout the flight: the Gravity Probe A experiment confirmed the
predicted rate shift to about one part in 10,000. The rocket flew in June
1976, but the definitive analysis was not published until four years later
-- Vessot, Levine, and collaborators' 1980 paper is the citation behind the
"1976" result, and readers looking for a contemporaneous 1976 paper will
not find one. That same effect is not merely a laboratory curiosity today
-- every GPS satellite clock runs measurably faster than clocks on the
ground it serves, and the discrepancy must be corrected for by design, or
accumulated position errors would reach several kilometers per day.

*Implementation:* :func:`physicskit.relativity.chapters.timekeeping.clock_rate_factor`
computes exactly this relativistic clock-rate shift for an orbiting clock;
:func:`~physicskit.relativity.chapters.timekeeping.gps_relativistic_offset_per_day`
gives the resulting daily timing correction GPS satellites apply.

*References:* R. F. C. Vessot, M. W. Levine, et al., "Test of Relativistic
Gravitation with a Space-Borne Hydrogen Maser," Phys. Rev. Lett. 45,
2081-2084 (1980) (experiment flown June 1976, results published 1980).

.. minigallery:: ../../examples/relativity/schwarzschild/plot_gps_relativistic_correction.py

2005-2006 -- Numerical Relativity Solves the Binary Black Hole Merger
-----------------------------------------------------------------------

For thirty years after the ADM formalism made numerical evolution of the
field equations possible in principle, every attempt to simulate two black
holes spiraling together and merging crashed before reaching merger --
numerical instabilities, chiefly around the singularities themselves, grew
without bound and destroyed the simulation. Frank Pretorius broke the
deadlock in 2005 with a new formulation (generalized harmonic coordinates,
combined with excision of the singular region) that evolved a binary black
hole cleanly through inspiral, merger, and ringdown for the first time.
Within months, two independent groups -- Campanelli, Lousto, Marronetti,
and Zlochower, and separately Baker, Centrella, Choi, Koppitz, and van
Meter -- found a second, more broadly adopted route to the same result, the
"moving puncture" method, which let punctures representing the black holes
move freely across the numerical grid without the manual excision
Pretorius's approach required. Together these breakthroughs turned binary
black hole mergers from an analytically inaccessible strong-field problem
into a solvable one, making it possible, for the first time, to compute
the full inspiral-merger-ringdown waveform -- including the merger itself,
which no post-Newtonian or perturbative approximation can reach -- directly
from the field equations. The precomputed template banks and surrogate
waveform models this made possible are exactly what let LIGO recognize
GW150914 as a binary black hole merger the moment it arrived, a decade
later.

*Implementation:* :meth:`physicskit.relativity.chapters.gw_merger.BinaryMerger.remnant_estimate`
anchors its equal-mass, non-spinning endpoint (:math:`M_f \approx 0.95 M`,
:math:`a_f/M_f \approx 0.69`) to exactly the merger remnant mass and spin
that these numerical-relativity simulations first measured; this package's
own docstring is explicit that :meth:`remnant_estimate` is "not a
precision numerical-relativity surrogate" but a simple interpolation
toward that well-measured NR result, and
:meth:`~physicskit.relativity.chapters.gw_merger.BinaryMerger.full_waveform`
stitches an inspiral onto a quasinormal-mode ringdown into the same
qualitative inspiral-merger-ringdown structure that these simulations were
the first to compute in full.

*References:* F. Pretorius, "Evolution of Binary Black-Hole Spacetimes,"
Phys. Rev. Lett. 95, 121101 (2005); M. Campanelli, C. O. Lousto, P.
Marronetti, and Y. Zlochower, "Accurate Evolutions of Orbiting
Black-Hole Binaries without Excision," Phys. Rev. Lett. 96, 111101 (2006);
J. G. Baker, J. Centrella, D.-I. Choi, M. Koppitz, and J. van Meter,
"Gravitational-Wave Extraction from an Inspiraling Configuration of
Merging Black Holes," Phys. Rev. Lett. 96, 111102 (2006).

.. minigallery:: ../../examples/relativity/gravitational_waves/plot_binary_merger_chirp.py

2015 -- LIGO and the Direct Detection of Gravitational Waves
------------------------------------------------------------

On 14 September 2015, the twin LIGO detectors observed a transient signal
-- a rising "chirp" in frequency and amplitude followed by a brief
ringdown -- matching, to remarkable precision, the predicted waveform of
two black holes of roughly 36 and 29 solar masses spiraling together and
merging into a single, still-quivering black hole, radiating about three
solar masses' worth of energy as gravitational waves in a fraction of a
second. GW150914 was the first direct detection of gravitational waves,
a full century after Einstein predicted their existence, and the first
direct observation of a binary black hole merger. The 2017 Nobel Prize in
Physics recognized the LIGO detection.

*Implementation:* :class:`physicskit.relativity.chapters.gw_merger.BinaryMerger`
implements exactly this inspiral-merger-ringdown sequence:
:meth:`~physicskit.relativity.chapters.gw_merger.BinaryMerger.inspiral_strain`
gives the chirping inspiral waveform,
:meth:`~physicskit.relativity.chapters.gw_merger.BinaryMerger.qnm_frequency_damping`
and :meth:`~physicskit.relativity.chapters.gw_merger.BinaryMerger.ringdown_strain`
give the remnant's damped-sinusoid ringdown, and
:meth:`~physicskit.relativity.chapters.gw_merger.BinaryMerger.full_waveform`
stitches the two regimes together into a single GW150914-like signal.

*References:* B. P. Abbott et al. (LIGO Scientific Collaboration and
Virgo Collaboration), "Observation of Gravitational Waves from a Binary
Black Hole Merger," Phys. Rev. Lett. 116, 061102 (2016).

.. minigallery:: ../../examples/relativity/gravitational_waves/plot_binary_merger_chirp.py

2017 -- GW170817 and the Dawn of Multi-Messenger Astronomy
----------------------------------------------------------

On 17 August 2017, LIGO and Virgo detected the inspiral of a *binary
neutron star* system -- far lighter than any binary black hole seen
before, and correspondingly slower-chirping, remaining in the detectors'
sensitive band for over a minute rather than a fraction of a second. Just
1.7 seconds after the merger, the Fermi and INTEGRAL satellites detected a
short gamma-ray burst from the same patch of sky, and observatories
worldwide went on to track the fading kilonova afterglow for weeks --
confirming that short gamma-ray bursts are neutron-star mergers, and that
these mergers forge heavy elements like gold and platinum via rapid
neutron capture (the r-process), long suspected but never before directly
observed. It was the first gravitational-wave event with a confirmed
electromagnetic counterpart, opening the era of multi-messenger
astronomy, and its independently measured distance and inferred recession
velocity gave a new, "standard siren" measurement of the Hubble constant.
The measured chirp mass,

.. math::

   \mathcal{M} = \frac{(m_1 m_2)^{3/5}}{(m_1+m_2)^{1/5}} \approx 1.186\,M_\odot,

was immediately recognizable as two neutron stars rather than two black
holes: far below any black hole binary LIGO had detected, and consistent
with the narrow mass range neutron stars are observed to occupy.

*Implementation:* :class:`physicskit.relativity.chapters.gw_merger.BinaryMerger`
with :math:`m_1, m_2 \approx 1.4\,M_\odot` reproduces this chirp mass via
its :attr:`~physicskit.relativity.chapters.gw_merger.BinaryMerger.chirp_mass`
property, and its :meth:`~physicskit.relativity.chapters.gw_merger.BinaryMerger.inspiral_frequency`
sweep shows exactly why such light neutron stars chirp for so much longer
than GW150914's ~30-solar-mass black holes did -- long enough for the
joint gravitational-wave and gamma-ray sky localization that made the
electromagnetic follow-up possible; the mass-radius relation from
:class:`physicskit.relativity.chapters.neutron_star.NeutronStar` traced by
:meth:`~physicskit.relativity.chapters.neutron_star.NeutronStar.mass_radius_curve`
is exactly the equation-of-state relation that observations like
GW170817's tidal deformability measurement are used to constrain.

*References:* B. P. Abbott et al. (LIGO Scientific Collaboration and Virgo
Collaboration), "GW170817: Observation of Gravitational Waves from a
Binary Neutron Star Inspiral," Phys. Rev. Lett. 119, 161101 (2017);
multi-messenger follow-up: "Multi-messenger Observations of a Binary
Neutron Star Merger," ApJ Letters 848, L12 (2017).

.. minigallery:: ../../examples/relativity/gravitational_waves/plot_gw170817_neutron_star_merger.py

2019 -- Event Horizon Telescope: The First Black Hole Image
-----------------------------------------------------------

On 10 April 2019, the Event Horizon Telescope collaboration released the
first direct image of a black hole's shadow: the supermassive black hole
at the center of the galaxy M87, roughly 6.5 billion solar masses,
imaged by linking radio dishes across the globe into a single
Earth-diameter interferometer. The image showed a dark central shadow
ringed by bright, asymmetric emission from infalling plasma -- direct
visual confirmation of the photon sphere and the shadow it casts, exactly
as general relativity predicts, cast at a distance :math:`D` at angular
radius

.. math::

   \theta_{\text{shadow}} \approx \frac{b_c}{D} = \frac{3\sqrt{3}\,M}{D}.

The collaboration went on to image Sagittarius A*, the far smaller and
closer supermassive black hole at the center of our own galaxy, in 2022.
Reinhard Genzel and Andrea Ghez shared the other half of the 2020 Nobel
Prize in Physics for tracking stellar orbits around Sagittarius A* closely
enough, over decades, to establish that it must be an extremely compact,
extremely massive dark object.

*Implementation:*
:attr:`physicskit.relativity.chapters.schwarzschild.SchwarzschildBlackHole.critical_impact_parameter`
gives exactly this shadow-casting impact parameter :math:`b_c=3\sqrt3\,M`;
:func:`physicskit.relativity.core.raytracer.render_shadow_image` and its
rotating-black-hole counterpart
:func:`~physicskit.relativity.core.kerr_raytracer.render_kerr_shadow_image`
backward ray-trace this shadow-and-photon-ring signature pixel by pixel
from a virtual camera, and
:func:`~physicskit.relativity.visualizers.shadow_render.plot_black_hole_shadow`
renders the result, reproducing the qualitative dark-disk-with-bright-ring
appearance of the EHT's M87* and Sagittarius A* images.

*References:* Event Horizon Telescope Collaboration, "First M87 Event
Horizon Telescope Results. I. The Shadow of the Supermassive Black Hole,"
ApJ Letters 875, L1 (2019) (first of a six-paper series); Sgr A* results:
Event Horizon Telescope Collaboration, "First Sagittarius A* Event Horizon
Telescope Results. I. The Shadow of the Supermassive Black Hole in the
Center of the Milky Way," ApJ Letters 930, L12 (2022).

.. minigallery::
   ../../examples/relativity/schwarzschild/plot_light_bending_and_shadow.py
   ../../examples/relativity/kerr/plot_kerr_shadow_and_redshift.py

See Also
--------

- :doc:`/api/relativity`
- :doc:`/api/gallery/relativity/index`
