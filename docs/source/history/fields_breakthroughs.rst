Breakthroughs in Classical and Quantum Field Theory
======================================================

.. epigraph::

   "For the great glory of research into that most excellent gift, light."
   -- inscription James Clerk Maxwell chose for a scientific instrument

From a single solitary wave chased down a Scottish canal to quantized
vortices spun up in a cloud of ultracold atoms, the physics behind
:mod:`physicskit.fields` spans a century and a half of discovering that
continuous media -- electromagnetic fields and macroscopic quantum
wavefunctions alike -- support robust, localized, particle-like
excitations. (Fluid dynamics, once part of this same chronology, now has
its own package and history: see :doc:`/history/fluid_breakthroughs`.)
This chronology traces that thread. Every stop has a pointer
to the corresponding implementation in this package, a structural diagram
of the system it describes, and a short, runnable example reproducing the
milestone's signature observable.

.. contents:: Timeline
   :local:
   :depth: 1

1834 -- Russell's Wave of Translation
------------------------------------------

Watching a boat stop abruptly in the Union Canal near Edinburgh, John
Scott Russell observed a single, smooth heap of water detach from the
bow wave and roll on for miles down the canal, "apparently without change
of form or diminution of speed." He rode after it on horseback for two
miles before losing it. Russell's "great wave of translation" was the
first documented observation of what would later be called a soliton: a
solitary wave that propagates without dispersing, decades before any
theory explained why.

*Implementation:* :func:`physicskit.fields.solitons.kdv_soliton` and
:func:`~physicskit.fields.solitons.kdv_evolve` reproduce the exact shape
that would eventually explain Russell's canal-side observation -- see
the 1895 Korteweg-de Vries entry below. See it in
:doc:`/api/gallery/fields/solitons/plot_kdv_soliton_translation`.

*References:* Russell's observation dates to 1834, but his own account
of it was not published until a decade later: J. Scott Russell, "Report
on Waves," Report of the 14th Meeting of the British Association for the
Advancement of Science (York, 1844), pp. 311-390.

.. minigallery:: ../../examples/fields/solitons/plot_kdv_soliton_translation.py

1865 -- Maxwell's Equations
--------------------------------

James Clerk Maxwell unified electricity, magnetism, and optics into four
coupled partial differential equations for the electric and magnetic
fields, and showed that they admit wave solutions propagating at a speed
determined purely by the vacuum permittivity and permeability,
:math:`c = 1/\sqrt{\varepsilon_0\mu_0}` -- matching the already-measured
speed of light so closely that Maxwell concluded light itself *is* an
electromagnetic wave. This single insight collapsed three previously
separate branches of physics into one field theory, the template every
subsequent field theory (fluid, soliton, or quantum) in this module
follows.

*Implementation:* :data:`physicskit.fields.electrodynamics.C0` is defined
exactly this way, and :func:`physicskit.fields.electrodynamics.fdtd_1d`
verifies numerically that a propagating pulse moves at precisely this speed.

*References:* J. C. Maxwell, "A Dynamical Theory of the Electromagnetic
Field," Phil. Trans. R. Soc. Lond. 155, 459-512 (1865).

.. minigallery:: ../../examples/fields/electrodynamics/plot_maxwell_speed_of_light.py

1887 -- Hertz Confirms Electromagnetic Waves
--------------------------------------------------

Twenty-two years after Maxwell's equations predicted that oscillating
electric charges should radiate energy as a wave traveling at the speed
of light, Heinrich Hertz built the apparatus to check it: an induction
coil driving a spark gap between two rod-shaped conductors -- an
oscillating electric dipole antenna -- and, meters away, a simple wire
loop with its own tiny spark gap as a detector. Between 1887 and 1888,
Hertz not only detected sparks jumping across that distant receiver in
step with the transmitter, proving the waves were real, but went on to
show they could be reflected, refracted, polarized, and diffracted just
like light, and that their measured speed matched the speed of light to
within experimental error. It was the first direct experimental proof
that light and what would later be called radio waves are the same
electromagnetic phenomenon, and it opened the door to wireless
telegraphy within a decade.

*Implementation:* :func:`physicskit.fields.electrodynamics.oscillating_dipole_source`
builds exactly Hertz's transmitter -- a soft, sinusoidally driven point
source -- for :func:`~physicskit.fields.electrodynamics.fdtd_2d_tmz_evolve`,
which records the outgoing ``Ez`` field as it radiates away from the
antenna on the Yee grid. See it in
:doc:`/api/gallery/fields/electrodynamics/plot_dipole_radiation`.

*References:* H. Hertz, "Über sehr schnelle electrische Schwingungen,"
Ann. Phys. 267, 421-448 (1887); H. Hertz, "Über Strahlen elektrischer
Kraft," Sitzungsber. Preuss. Akad. Wiss. Berlin (1888); collected in
*Electric Waves* (1893; English translation 1900).

.. minigallery:: ../../examples/fields/electrodynamics/plot_dipole_radiation.py

1895 -- The Korteweg-de Vries Equation
--------------------------------------------

Diederik Korteweg and Gustav de Vries derived a single nonlinear partial
differential equation, :math:`u_t + 6uu_x + u_{xxx} = 0`, for shallow-water
waves, finally explaining Russell's wave mathematically: a balance between
nonlinear steepening (:math:`uu_x`) and linear dispersion
(:math:`u_{xxx}`) allows exactly this kind of shape-preserving solitary
wave to exist. It would take another seventy years, and the discovery
that colliding KdV solitons pass through each other completely
unchanged (an "elastic collision" utterly unlike ordinary nonlinear
waves), before the deeper mathematical structure -- complete integrability
-- was understood.

*Implementation:* :func:`physicskit.fields.solitons.kdv_soliton` and
:func:`~physicskit.fields.solitons.kdv_evolve` reproduce exactly this
elastic two-soliton collision.

*References:* D. J. Korteweg and G. de Vries, "On the Change of Form of
Long Waves Advancing in a Rectangular Canal, and on a New Type of Long
Stationary Waves," Phil. Mag. Series 5, 39(240), 422-443 (1895). The
equation itself was not entirely new: Joseph Boussinesq had derived an
essentially equivalent equation eighteen years earlier (J. Math. Pures
Appl. 17, 55-108, 1877), a priority nuance that does not disturb
Korteweg and de Vries' name on the result they made famous.

.. minigallery:: ../../examples/fields/solitons/plot_kdv_elastic_collision.py

1926 -- The Schrodinger Equation
--------------------------------------

Erwin Schrodinger proposed a wave equation, :math:`i\hbar\partial_t\psi =
-\tfrac{\hbar^2}{2m}\nabla^2\psi + V\psi`, governing the evolution of a
quantum-mechanical wavefunction, unifying de Broglie's matter-wave
hypothesis with a workable equation of motion. Its most basic prediction
-- that a localized particle's wavefunction inevitably spreads out over
time -- was as radical as it was verifiable. Adding a single nonlinear
self-interaction term to this equation -- as done independently for light
in an optical fiber (Hasegawa and Tappert, 1973) and for a
Bose-Einstein condensate (Gross and Pitaevskii, 1961) -- is exactly what
turns it into the soliton-bearing nonlinear Schrodinger equation used
throughout this module.

*Implementation:* :func:`physicskit.fields.solitons.nls_evolve` solves
precisely the linear Schrodinger term (:math:`\tfrac{1}{2}\psi_{xx}`)
alongside the added nonlinearity :math:`g|\psi|^2\psi`.

*References:* E. Schrodinger, "Quantisierung als Eigenwertproblem" (I-IV),
Ann. Phys. 384-386 (1926); E. Schrodinger, Phys. Rev. 28, 1049 (1926). For
the added nonlinear term: A. Hasegawa and F. Tappert, Appl. Phys. Lett.
23, 142-144 (1973); E. P. Gross, Nuovo Cimento 20, 454-457 (1961); L. P.
Pitaevskii, Sov. Phys. JETP 13, 451-454 (1961).

.. minigallery:: ../../examples/fields/solitons/plot_nls_wavepacket_spreading.py

1938 -- The Frenkel-Kontorova Model and the Sine-Gordon Equation
--------------------------------------------------------------------

Yakov Frenkel and Tatiana Kontorova modeled a crystal dislocation as a
chain of atoms coupled to their neighbors and sitting in a periodic
substrate potential -- a discrete chain of coupled pendula, in the
mechanical analogy. In the continuum limit this becomes the Sine-Gordon
equation, :math:`u_{tt} - u_{xx} + \sin u = 0`, whose kink solution (a
single :math:`2\pi` twist propagating along the chain) is a topological
soliton: it cannot be untwisted by any local, continuous deformation,
only by another kink of opposite polarity annihilating it. The same
equation later reappeared as the effective theory of magnetic flux
quanta threading a long Josephson junction.

*Implementation:* :func:`physicskit.fields.solitons.sine_gordon_kink` and
:func:`~physicskit.fields.solitons.sine_gordon_evolve` construct and
propagate exactly this topologically protected kink. See it in
:doc:`/api/gallery/fields/solitons/plot_sine_gordon_kink`.

*References:* Ya. I. Frenkel and T. Kontorova, Zh. Eksp. Teor. Fiz. 8,
1340 (1938) [also published as Phys. Z. Sowjetunion 13, 1 (1938)].

.. minigallery:: ../../examples/fields/solitons/plot_sine_gordon_kink.py

1948 -- Casimir's Vacuum Force
-------------------------------------

Working at Philips, Hendrik Casimir showed that two uncharged, perfectly
conducting parallel plates placed in vacuum should attract each other,
purely as a consequence of quantum field theory: between the plates,
only electromagnetic vacuum modes whose wavelength fits an integer number
of times in the gap are allowed, so the zero-point energy density trapped
between the plates is slightly lower than in the unbounded vacuum outside
them, and the plates are pushed together by the resulting pressure
imbalance,

.. math::

   \frac{F}{A} = -\frac{\pi^2\hbar c}{240\,d^4},

for plate separation :math:`d`. No experiment of the day could resolve a
force this small; a first, rough confirmation came only in 1958 (Marcus
Sparnaay), and a precise measurement -- agreeing with Casimir's formula to
about five percent -- had to wait until Steve Lamoreaux's 1997
torsion-pendulum experiment. The Casimir effect remains one of the few
places where the reality of the electromagnetic vacuum's zero-point
energy is visible as a macroscopic, measurable force.

*Implementation:* :func:`physicskit.fields.quantum_fields.casimir_mode_frequencies`
builds the analogous discrete standing-wave spectrum of a 1D scalar field
confined between two "plates" a distance ``d`` apart (:math:`\omega_n =
n\pi c/d`, a simplified stand-in for the full 3D electromagnetic
calculation), and :func:`~physicskit.fields.quantum_fields.casimir_energy_1d`
regularizes and sums its zero-point energy via an exponential cutoff,
reproducing exactly the qualitative signature Casimir predicted: a
finite, negative (attractive), separation-dependent vacuum energy that
grows less negative as the plates move apart. See it in
:doc:`/api/gallery/fields/quantum_fields/plot_casimir_effect`.

*References:* H. B. G. Casimir, Proc. K. Ned. Akad. Wet. 51, 793-795
(1948); M. J. Sparnaay, Physica 24, 751-764 (1958) (first, rough
confirmation); S. K. Lamoreaux, Phys. Rev. Lett. 78, 5-8 (1997)
(definitive measurement).

.. minigallery:: ../../examples/fields/quantum_fields/plot_casimir_effect.py

1949-1955 -- Quantized Vortices in Superfluid Helium
------------------------------------------------------------

Lars Onsager (1949) and, independently, Richard Feynman (1955) proposed
that the frictionless flow of superfluid helium-4 could only rotate by
threading itself with discrete vortex lines, each carrying exactly one
quantum of circulation :math:`\oint\mathbf{v}\cdot d\boldsymbol{\ell} =
h/m`. Direct experimental confirmation followed in 1961 (William Vinen),
decades before the same underlying physics -- a macroscopic wavefunction
forced to carry angular momentum only in integer units -- could be
observed directly in a rotating Bose-Einstein condensate.

*Implementation:* the same quantization is what
:func:`physicskit.fields.quantum_fields.count_vortices` detects, as an
exact integer phase winding around each vortex core;
:func:`~physicskit.fields.visualizers.plot_bec_density` and
:func:`~physicskit.fields.visualizers.plot_bec_phase` make that winding
directly visible, side by side, as a density dip pierced by a phase
singularity.

*References:* L. Onsager, remark during discussion, Nuovo Cimento
Suppl. 6, 279-287 (1949); R. P. Feynman, Chapter II in *Progress in Low
Temperature Physics*, Vol. 1 (North-Holland, 1955); W. F. Vinen,
Proc. R. Soc. Lond. A 260, 218-236 (1961) (experimental confirmation).

.. minigallery:: ../../examples/fields/quantum_fields/plot_superfluid_vortex_quantization.py

1961 -- Gross-Pitaevskii Theory
------------------------------------

Eugene Gross and Lev Pitaevskii independently derived a mean-field
equation for the macroscopic wavefunction of a dilute Bose-Einstein
condensate: a nonlinear Schrodinger equation with a cubic
self-interaction term, :math:`i\hbar\partial_t\psi =
[-\tfrac{\hbar^2}{2m}\nabla^2 + V + g|\psi|^2]\psi`. Decades before a real
BEC was ever created in a laboratory (1995), the Gross-Pitaevskii
equation gave theorists a concrete, quantitative tool for predicting its
structure -- including, crucially, that it should support quantized
vortices exactly analogous to those in superfluid helium.

*Implementation:* :func:`physicskit.fields.quantum_fields.gpe_relax` solves
exactly this equation, via imaginary-time propagation, for both static
and rotating traps; :func:`~physicskit.fields.quantum_fields.gpe_energy`
evaluates the resulting condensate's energy functional, and
:func:`~physicskit.fields.visualizers.plot_bec_density` renders the
relaxed ground-state density.

*References:* E. P. Gross, Nuovo Cimento 20, 454-457 (1961); L. P.
Pitaevskii, Sov. Phys. JETP 13, 451-454 (1961).

.. minigallery:: ../../examples/fields/quantum_fields/plot_gpe_ground_state_relaxation.py

1965 -- Zabusky and Kruskal Coin "Soliton"
-------------------------------------------------

Trying to numerically resolve a puzzle left open by the 1955
Fermi-Pasta-Ulam-Tsingou experiment -- why a nonlinear chain of
oscillators stubbornly failed to thermalize -- Norman Zabusky and Martin
Kruskal simulated the continuum (KdV) limit of that chain on a computer,
and watched a generic initial disturbance "fission" into a rank-ordered
train of solitary waves that then collided and emerged completely
unscathed, each recovering its exact original shape and speed. They
coined the name "soliton" for this particle-like behavior, in direct
analogy to a proton or electron, launching soliton theory as a distinct
field of mathematical physics.

*Implementation:* :func:`physicskit.fields.solitons.kdv_evolve`,
started from a single generic pulse rather than an exact soliton,
reproduces the very fission phenomenon that gave solitons their name.

*References:* N. J. Zabusky and M. D. Kruskal, Phys. Rev. Lett. 15,
240-243 (1965). The recurrence puzzle they set out to resolve traces to
E. Fermi, J. Pasta, and S. Ulam, "Studies of Nonlinear Problems," Los
Alamos report LA-1940 (1955).

.. minigallery:: ../../examples/fields/solitons/plot_kdv_soliton_fission.py

1966 -- Kane Yee's FDTD Algorithm
----------------------------------------

Kane Yee introduced a deceptively simple idea for solving Maxwell's
equations numerically: stagger the electric and magnetic field components
on interleaved spatial and temporal grids (a "Yee cell"), so that each
field's curl is naturally centered on the other's location. This
staggering makes the resulting leapfrog update both second-order accurate
and, remarkably, satisfy a discrete analog of Gauss's law automatically.
Fifty years later, the finite-difference time-domain (FDTD) method remains
the workhorse for simulating antennas, waveguides, and photonic devices.

*Implementation:* :func:`physicskit.fields.electrodynamics.fdtd_1d` and
:func:`~physicskit.fields.electrodynamics.fdtd_2d_tmz` are direct
implementations of the Yee scheme.
:func:`~physicskit.fields.electrodynamics.poynting_vector_tmz` computes
the resulting energy-flux (Poynting) field directly from the same
staggered :math:`E_z,H_x,H_y` components, and
:func:`~physicskit.fields.visualizers.plot_poynting_field` renders it,
showing the radiating dipole's energy actually flowing outward rather
than just its field amplitude oscillating in place.

*References:* K. S. Yee, IEEE Trans. Antennas Propag. 14(3), 302-307
(1966).

.. minigallery:: ../../examples/fields/electrodynamics/plot_yee_fdtd_point_source.py

The same staggered-grid update handles a wave crossing a material
interface, or an idealized closed cavity, without any change to its core
leapfrog rule: :func:`~physicskit.fields.electrodynamics.dielectric_slab`
supplies a permittivity map to :func:`fdtd_2d_tmz` so a wave partially
reflects and partially transmits at a dielectric interface, and
:func:`~physicskit.fields.electrodynamics.tmz_cavity_mode` gives the
analytic :math:`TM_{mn}` standing wave of a rectangular PEC cavity to
seed the same solver with, rather than launching a traveling pulse.
:func:`~physicskit.fields.electrodynamics.fdtd_2d_tmz_evolve` is the
snapshot-recording variant of :func:`fdtd_2d_tmz` used to animate all of
these with :func:`physicskit.fields.visualizers.animate_field_2d`. See
the medium crossing in
:doc:`/api/gallery/fields/electrodynamics/plot_dielectric_slab_propagation`
and the standing cavity mode in
:doc:`/api/gallery/fields/electrodynamics/plot_em_cavity_modes`.

1967 -- The Inverse Scattering Transform
-------------------------------------------------

Clifford Gardner, John Greene, Martin Kruskal, and Robert Miura discovered
that the Korteweg-de Vries equation could be solved exactly for arbitrary
initial data, via a remarkable trick: treat the initial waveform as a
potential in a linear Schrodinger scattering problem, evolve the
resulting (trivial, linear) scattering data forward in time, and
reconstruct the nonlinear solution at any later time by inverting the
scattering problem. This "inverse scattering transform" finally explained
*why* KdV solitons collide elastically -- each corresponds to a bound
state whose scattering data, and hence identity, never changes -- and
revealed KdV as merely the first example of a much broader class of
exactly integrable nonlinear equations, the Sine-Gordon and nonlinear
Schrodinger equations included. For Sine-Gordon, the technique yields
closed-form multi-soliton solutions directly -- including an exact
kink-antikink collision -- rather than requiring numerical evolution at all.

*Implementation:* the elastic collisions reproduced by
:func:`physicskit.fields.solitons.kdv_evolve` and the exact kink solutions
in :func:`~physicskit.fields.solitons.sine_gordon_evolve` are both
consequences of precisely this integrability.

*References:* C. S. Gardner, J. M. Greene, M. D. Kruskal, and R. M.
Miura, Phys. Rev. Lett. 19, 1095-1097 (1967).

.. minigallery:: ../../examples/fields/solitons/plot_sine_gordon_kink_antikink_ist.py

1972 -- Zakharov's Collapse of Langmuir Waves
------------------------------------------------------

Studying intense plasma oscillations (Langmuir waves), Vladimir Zakharov
showed in "Collapse of Langmuir Waves" that the same focusing nonlinear
Schrodinger equation behind Hasegawa and Tappert's stable optical
solitons (see 1973, below) has a second, much more violent regime: when
the initial wave packet is narrow and intense enough, in two or three
spatial dimensions nonlinear self-focusing can overwhelm dispersion so
completely that the packet contracts toward a genuine mathematical
singularity in finite time, rather than settling into a shape-preserving
soliton. Zakharov's "wave collapse" is not a curiosity specific to
plasmas: the identical mechanism governs the catastrophic self-focusing
of sufficiently intense laser beams in nonlinear optical media, and the
analogous collapse of an attractively interacting Bose-Einstein
condensate once its density exceeds a critical threshold.

*Implementation:* :func:`physicskit.fields.quantum_fields.gpe_evolve`,
called with a vanishing trapping potential and an attractive
self-interaction (``g < 0`` in this module's sign convention), places a
tall, narrow initial packet directly in this focusing regime: real-time
split-step propagation shows it visibly contracting into a narrower,
taller peak, the finite-grid stand-in for approaching (never quite
reaching, since no numerical grid can resolve an actual singularity) the
collapse Zakharov predicted. See it in
:doc:`/api/gallery/fields/quantum_fields/plot_nls_self_focusing_collapse`.

*References:* V. E. Zakharov, Zh. Eksp. Teor. Fiz. 62, 1745-1759 (1972)
[Sov. Phys. JETP 35, 908-914 (1972)].

.. minigallery:: ../../examples/fields/quantum_fields/plot_nls_self_focusing_collapse.py

1973 -- The Kosterlitz-Thouless Transition
--------------------------------------------------

John Kosterlitz and David Thouless explained how a two-dimensional
superfluid or magnet can support order without the long-range order
forbidden in 2D by thermal fluctuations: below a critical temperature,
thermally created vortices bind tightly into vortex-antivortex pairs of
opposite circulation, whose combined velocity fields cancel at long
range, leaving superfluid order intact. Above that temperature the pairs
unbind into a free plasma of independent vortices and antivortices,
destroying superfluidity through purely topological defect proliferation
rather than any conventional symmetry-breaking transition. Applied to a
rotating, quasi-two-dimensional Bose-Einstein condensate, this is the
same vortex-antivortex physics that governs how the isolated,
same-signed vortices of the 1949-1955 entry above and the corotating
lattice below (1995-2001) first nucleate and unbind from the condensate's
thermal component.

*Implementation:* this module's :func:`physicskit.fields.quantum_fields.count_vortices`
detects individual vortices and antivortices alike, as the same signed
:math:`\pm 1` phase windings that pair up (or unbind) in the
Kosterlitz-Thouless picture; simulating the thermal binding-unbinding
transition itself is outside this package's current scope.

*References:* J. M. Kosterlitz and D. J. Thouless, J. Phys. C 6,
1181-1203 (1973).

.. minigallery:: ../../examples/fields/quantum_fields/plot_vortex_antivortex_pair.py

1973 -- Optical Solitons
------------------------------

Akira Hasegawa and Fred Tappert showed theoretically that the same balance
of nonlinearity and dispersion behind Russell's water wave -- now governed
by the nonlinear Schrodinger equation rather than KdV -- allows light
pulses in an optical fiber to propagate as solitons, self-correcting
against the pulse-spreading dispersion that otherwise limits every
long-distance optical communication line. Experimentally confirmed within
a decade, optical solitons remain a foundational idea in nonlinear fiber
optics.

*Implementation:* :func:`physicskit.fields.solitons.nls_bright_soliton`
and :func:`~physicskit.fields.solitons.nls_evolve` reproduce exactly this
shape-preserving envelope propagation. See it in
:doc:`/api/gallery/fields/solitons/plot_nls_optical_soliton`.

*References:* A. Hasegawa and F. Tappert, Appl. Phys. Lett. 23, 142-144
(1973) (theory); L. F. Mollenauer, R. H. Stolen, and J. P. Gordon, Phys.
Rev. Lett. 45, 1095-1098 (1980) (first experimental observation).

.. minigallery:: ../../examples/fields/solitons/plot_nls_optical_soliton.py

1974 -- Wilson's Lattice Confinement of Quarks
--------------------------------------------------

Kenneth Wilson, in "Confinement of Quarks," put Yang-Mills gauge theory
onto a discrete spacetime lattice -- gauge fields living as link
variables between neighboring lattice sites -- in a way that preserves
exact gauge invariance non-perturbatively, something no continuum
perturbative expansion could offer. In the resulting theory's
strong-coupling limit, Wilson showed that the potential energy of a
static quark-antiquark pair grows *linearly* with their separation,
:math:`V(r) \propto r`, rather than falling off with distance the way an
ordinary Coulomb field does. A linearly rising potential means pulling
two quarks apart costs an ever-increasing amount of energy -- infinite at
infinite separation -- so an isolated colored quark can never be pulled
free of its partner; this is confinement, the reason free quarks have
never been observed in isolation. The physical picture behind the linear
potential is that, unlike an ordinary field that spreads out in all
directions, the gluon field lines are squeezed into a narrow "flux tube"
of roughly constant cross-section connecting the two charges, so the
field energy per unit length -- and hence the total energy -- is the same
everywhere along the tube, giving exactly the linear growth Wilson's
lattice calculation found.

*Implementation:* :func:`physicskit.fields.electrodynamics.flux_tube_field_1d`
and :func:`~physicskit.fields.electrodynamics.flux_tube_energy_density_2d`
reproduce only this qualitative confinement signature -- a field confined
to a fixed-cross-section tube by direct construction, giving a
separation-independent energy density along the tube and hence a total
field energy that grows linearly with separation -- as a simplified,
illustrative 1+1D toy model, **not** a lattice-QCD or first-principles
gauge-theory calculation. See it in
:doc:`/api/gallery/fields/gauge_confinement/plot_flux_tube_confinement`.

*References:* K. G. Wilson, Phys. Rev. D 10, 2445-2459 (1974).

.. minigallery:: ../../examples/fields/gauge_confinement/plot_flux_tube_confinement.py

1994 -- Berenger's Perfectly Matched Layer
-------------------------------------------------

Jean-Pierre Berenger solved a problem that had limited FDTD simulations
since Yee's original 1966 paper: a finite computational grid needs some
boundary condition, and a simple wall reflects outgoing waves right back
into the simulation, contaminating the result. Berenger's Perfectly
Matched Layer (PML) is a graded absorbing medium, constructed so its wave
impedance matches the interior exactly at every angle of incidence --
reflectionless in principle, and reducing reflections by many orders of
magnitude in practice. It transformed FDTD from a method usable mainly for
closed cavities into one usable for open, radiating, or scattering problems.

*Implementation:* :func:`physicskit.fields.electrodynamics.pml_conductivity_profile`
implements a simplified graded-conductivity absorbing boundary in this
same spirit.

*References:* J.-P. Berenger, J. Comput. Phys. 114(2), 185-200 (1994).

.. minigallery:: ../../examples/fields/electrodynamics/plot_pml_absorbing_boundary.py

1995-2001 -- Quantized Vortex Lattices in BECs
--------------------------------------------------------

Two groups produced the first dilute-gas Bose-Einstein condensates within
months of each other in 1995: Mike Anderson, Jason Ensher, Michael
Matthews, Carl Wieman, and Eric Cornell at JILA, condensing rubidium-87,
and Kit Davis, Marc-Oliver Mewes, Michael Andrews, and Wolfgang Ketterle's
group at MIT, condensing sodium-23. It took another five to six years
before anyone coaxed a condensate into showing the vortex physics that
Gross-Pitaevskii theory -- by then nearly four decades old -- had long
predicted: in 2000, Kirk Madison, Frederic Chevy, Wendel Wohlleben, and
Jean Dalibard spun a rotating condensate fast enough to nucleate a single
quantized vortex, and in 2001 Jamil Abo-Shaeer, Chandra Raman, Jeff
Vogels, and Ketterle pushed the rotation frequency further and watched
dozens of vortices arrange themselves into a strikingly regular
triangular lattice -- a dilute-gas realization of the same triangular
flux-line lattice Alexei Abrikosov had predicted in 1957 for magnetic
flux threading a type-II superconductor -- directly visualizing single
quanta of superfluid circulation in a BEC for the first time.

*Implementation:* :func:`physicskit.fields.quantum_fields.gpe_imprint_vortex`,
:func:`~physicskit.fields.quantum_fields.gpe_relax` (with ``Omega != 0``),
and :func:`~physicskit.fields.quantum_fields.count_vortices` reproduce the
energetic vortex-nucleation criterion and detect the resulting quantized
circulation, as worked through in :doc:`/tutorials/bec_vortex_lattice_creation`;
:func:`~physicskit.fields.quantum_fields.gpe_energy` quantifies the
energy cost of nucleating a vortex, and
:func:`~physicskit.fields.visualizers.plot_bec_density` and
:func:`~physicskit.fields.visualizers.plot_bec_phase` compare the
vortex-free and vortex-carrying condensates side by side.

*References:* M. H. Anderson, J. R. Ensher, M. R. Matthews, C. E. Wieman,
and E. A. Cornell, Science 269, 198-201 (1995); K. B. Davis, M.-O. Mewes,
M. R. Andrews, N. J. van Druten, D. S. Durfee, D. M. Kurn, and W.
Ketterle, Phys. Rev. Lett. 75, 3969-3973 (1995) (first dilute-gas BECs);
A. A. Abrikosov, Zh. Eksp. Teor. Fiz. 32, 1442-1452 (1957) [Sov. Phys.
JETP 5, 1174-1182 (1957)] (predicted the triangular flux-line lattice);
K. W. Madison, F. Chevy, W. Wohlleben, and J. Dalibard, Phys. Rev. Lett.
84, 806-809 (2000); J. R. Abo-Shaeer, C. Raman, J. M. Vogels, and W.
Ketterle, Science 292, 476-479 (2001) (vortex and vortex-lattice
nucleation in a BEC).

.. minigallery:: ../../examples/fields/quantum_fields/plot_bec_vortex_lattice_nucleation.py

Everything above is the relaxed, static end state that Gross-Pitaevskii
theory predicts. The condensate's vortices are not painted-on
decorations, though: seeded off-center and propagated in genuine real
time rather than the energy-minimizing imaginary-time flow used to find
the lattice's stationary building block above, a single vortex
physically precesses around the trap center, driven by the local density
gradient it sits in -- the same orbital motion that carries a newly
nucleated vortex from the condensate's edge into its place in the
crystalline lattice.

*Implementation:* :func:`physicskit.fields.quantum_fields.gpe_evolve`
solves the identical Gross-Pitaevskii equation as
:func:`~physicskit.fields.quantum_fields.gpe_relax`, but by real-time
split-step propagation rather than imaginary-time relaxation, so an
off-center vortex's orbital precession -- genuine time dynamics, not just
its existence as a stationary solution -- is reproduced directly and
animated frame by frame with
:func:`physicskit.fields.visualizers.animate_density_2d`. See it in
:doc:`/api/gallery/fields/quantum_fields/plot_gpe_real_time_vortex_precession`.

See Also
--------

- :doc:`/tutorials/fdtd_waveguide_simulation`
- :doc:`/tutorials/bec_vortex_lattice_creation`
- :doc:`/history/fluid_breakthroughs` (the Navier-Stokes, Kelvin-Helmholtz,
  and von Karman entries that used to appear in this chronology have moved
  there, alongside the rest of :mod:`physicskit.fluids`)
- :doc:`/api/index`
