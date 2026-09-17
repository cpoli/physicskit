Breakthroughs in Statistical Mechanics
======================================

.. epigraph::

   "The true logic of this world is the calculus of probabilities."
   -- James Clerk Maxwell, letter to Lewis Campbell, 1850

Statistical mechanics is the discovery that the definite laws of
thermodynamics -- entropy, temperature, irreversibility -- are nothing
but the overwhelmingly probable behavior of vast numbers of
microscopic degrees of freedom obeying ordinary, reversible mechanics.
:mod:`physicskit.statphys` follows that idea from a single urn of
balls to a self-organizing pile of sand, spanning lattice magnets,
disordered fluids, and critical phenomena along the way. This
chronology traces the major conceptual breakthroughs behind the
package, with a pointer to the corresponding implementation at each
stop.

.. contents:: Timeline
   :local:
   :depth: 1

1860 -- Maxwell's Distribution of Molecular Velocities
-------------------------------------------------------

James Clerk Maxwell derived the first genuinely statistical law in
physics: from little more than the isotropy of velocity space and the
statistical independence of a molecule's three velocity components, he
found that a gas in equilibrium has a definite, universal distribution
of molecular speeds,

.. math::

   f(v) = 4\pi v^2 \left(\frac{m}{2\pi k_B T}\right)^{3/2}
   e^{-mv^2/2k_B T},

extracted without ever solving an individual molecule's equation of
motion. It was, at the time, a special-case result -- Maxwell assumed
the distribution's form and showed it was self-consistent under
elastic collisions, for a gas already at equilibrium. Twelve years
later Boltzmann would generalize the argument to interacting gases away
from equilibrium and, with the H-theorem below, prove that essentially
*any* initial velocity distribution relaxes toward exactly this
Maxwellian form -- turning Maxwell's static equilibrium law into the
attracting fixed point of a genuine dynamical process.

*Implementation:* :func:`physicskit.statphys.utils.thermodynamics.maxwell_boltzmann_speed_pdf`
evaluates exactly this equilibrium speed density in 1, 2, or 3
dimensions, and
:func:`physicskit.statphys.visualizers.particle_render.plot_velocity_histogram`
overlays it directly against the simulated speed histogram of a real
molecular dynamics trajectory.

*References:* J. C. Maxwell, "Illustrations of the Dynamical Theory of
Gases," Phil. Mag. Ser. 4, 19, 19-32, and 20, 21-37 (1860); L.
Boltzmann, "Weitere Studien über das Wärmegleichgewicht unter
Gasmolekülen," Wiener Berichte 66, 275-370 (1872).

.. minigallery:: ../../examples/statphys/molecular_dynamics/plot_maxwell_boltzmann_relaxation.py

1870s -- Boltzmann's Statistical Entropy
----------------------------------------

Ludwig Boltzmann proposed that a macroscopic system's entropy simply
counts the number of microscopic arrangements consistent with its
observed macrostate,

.. math::

   S = k_B \ln W,

an equation so central to statistical mechanics that Max Planck later
had it carved on Boltzmann's tombstone (Boltzmann himself never wrote
it in exactly this form; Planck introduced the constant :math:`k_B` and
this notation around 1900). Boltzmann's companion H-theorem showed that
a quantity built from the single-particle velocity distribution,
:math:`H = \int f\ln f\,d^3v`, decreases monotonically under molecular
collisions -- a microscopic derivation of the second law that
provoked serious objections over the following decades, though not
all at once: Loschmidt's reversibility paradox arrived in 1876, just
four years later, while Zermelo's recurrence paradox followed only in
1896, nearly a quarter century on, both asking how irreversibility
could possibly emerge from reversible, recurrent mechanics.

*Implementation:* :meth:`physicskit.statphys.chapters.ehrenfest_urn.EhrenfestUrn.entropy`
computes :math:`S/k_B = \ln W` exactly, via the log-gamma binomial
coefficient of the current macrostate; and
:meth:`physicskit.statphys.chapters.molecular_dynamics.LennardJonesGas.h_function`
tracks Boltzmann's H-function directly through a real molecular
dynamics trajectory, watching it decrease as the gas relaxes toward
equilibrium.

*References:* L. Boltzmann, "Weitere Studien über das
Wärmegleichgewicht unter Gasmolekülen," Wiener Berichte 66, 275-370
(1872); L. Boltzmann, "Über die Beziehung zwischen dem zweiten
Hauptsatze der mechanischen Wärmetheorie und der
Wahrscheinlichkeitsrechnung," Wiener Berichte 76, 373-435 (1877);
M. Planck, Ann. Phys. 4, 553-563 (1901).

.. minigallery::
   ../../examples/statphys/ehrenfest_urn/plot_ehrenfest_irreversibility.py
   ../../examples/statphys/molecular_dynamics/plot_maxwell_boltzmann_relaxation.py

1902 -- Gibbs's Ensemble Formalism
----------------------------------

Josiah Willard Gibbs organized statistical mechanics around the
*ensemble*: not a single system's trajectory, but a probability
distribution over all systems consistent with fixed macroscopic
constraints -- microcanonical (fixed energy), canonical (fixed
temperature), and grand canonical (fixed chemical potential). Gibbs's
canonical ensemble is the workhorse of the entire field: it is what
makes a Monte Carlo sampler at fixed temperature meaningful, and it
ties macroscopic response functions directly to microscopic
fluctuations,

.. math::

   C_v = \frac{\langle E^2\rangle - \langle E\rangle^2}{k_B T^2 N},
   \qquad
   \chi = \frac{\langle M^2\rangle - \langle M\rangle^2}{k_B T N}.

These are often loosely called "fluctuation-dissipation relations,"
but they are equilibrium canonical-ensemble identities -- static
snapshots of a system's own thermal fluctuations -- rather than the
actual (dynamical) fluctuation-dissipation theorem of Callen and
Welton, which relates a system's spontaneous equilibrium fluctuations
to its time-dependent linear response to an external perturbation.

*Implementation:* :func:`physicskit.statphys.utils.thermodynamics.specific_heat`
and :func:`~physicskit.statphys.utils.thermodynamics.susceptibility`
compute exactly these canonical-ensemble relations from a sampled
trajectory, used throughout the package -- e.g. by
:meth:`physicskit.statphys.chapters.ising_lattice.Ising2D.run_temperature_sweep`.

*References:* J. W. Gibbs, Elementary Principles in Statistical
Mechanics (Charles Scribner's Sons / Yale University Press, 1902).
Cf. H. B. Callen and T. A. Welton, "Irreversibility and Generalized
Noise," Phys. Rev. 83, 34-40 (1951), for the dynamical
fluctuation-dissipation theorem proper.

.. minigallery:: ../../examples/statphys/ising/plot_ising_phase_transition.py

1905 -- Einstein's Explanation of Brownian Motion
-------------------------------------------------

Albert Einstein showed that the erratic jiggling of a pollen grain
suspended in water, observed by Robert Brown in 1827, is the visible
signature of unseen molecular collisions: a particle undergoing a
random walk of microscopic kicks has a mean squared displacement
growing *linearly* in time,

.. math::

   \langle x^2(t)\rangle = 2Dt,

with diffusion coefficient :math:`D = k_B T / (6\pi\eta r)` (the
Stokes-Einstein relation) fixed by the fluid's viscosity and the
particle's radius. This gave Jean Perrin an experimental handle,
realized within a few years, to measure Avogadro's number from
microscope observations alone -- turning "atoms" from a convenient
hypothesis into a directly, quantitatively measured fact. (Coincidentally,
that same year Karl Pearson posed "the problem of the random walk" in
a letter to *Nature*, giving the underlying process its now-standard
name.)

*Implementation:* :class:`physicskit.statphys.chapters.random_walk.RandomWalk`
simulates exactly this process (lattice or continuous Gaussian steps);
:meth:`~physicskit.statphys.chapters.random_walk.RandomWalk.mean_squared_displacement`
and :meth:`~physicskit.statphys.chapters.random_walk.RandomWalk.diffusion_coefficient`
recover the linear-in-time law and the diffusion coefficient directly
from simulated trajectories.

*References:* A. Einstein, Ann. Phys. 322(8), 549-560 (1905); K.
Pearson, Nature 72, 294 (1905).

.. minigallery:: ../../examples/statphys/random_walk/plot_diffusion_and_clt.py

1907 -- The Ehrenfest Urn Model
-------------------------------

Paul and Tatiana Ehrenfest answered Loschmidt's and Zermelo's
objections to Boltzmann's H-theorem with the simplest model that could
possibly settle the argument: :math:`N` labeled balls split between two
boxes, one ball moved at random each step. The dynamics is exactly
reversible and does, in principle, recur exactly to its initial state
(as Zermelo's Poincare-recurrence objection demands) -- yet for any
macroscopic :math:`N` the expected recurrence time is astronomically
larger than the age of the universe, while the approach to the
near-equal-occupancy equilibrium is essentially immediate. Reversible
microphysics and an observed thermodynamic arrow of time turn out to be
perfectly compatible; only the timescales involved are utterly
different.

*Implementation:* :class:`physicskit.statphys.chapters.ehrenfest_urn.EhrenfestUrn`,
:meth:`~physicskit.statphys.chapters.ehrenfest_urn.EhrenfestUrn.step`,
and :meth:`~physicskit.statphys.chapters.ehrenfest_urn.EhrenfestUrn.run`
reproduce exactly this reversible random-exchange dynamics and its
rapid, statistical relaxation.

*References:* P. Ehrenfest and T. Ehrenfest, Phys. Z. 8, 311-314
(1907).

.. minigallery:: ../../examples/statphys/ehrenfest_urn/plot_ehrenfest_irreversibility.py

1908 -- The Langevin Equation
------------------------------

Paul Langevin gave Einstein's statistical account of Brownian motion a
dynamical companion: instead of a diffusion equation for the
probability of finding a particle somewhere, write Newton's second law
for the particle itself, split its interaction with the fluid into a
systematic drag and a rapidly fluctuating random force,

.. math::

   m \frac{dv}{dt} = -\gamma v + \xi(t), \qquad
   \langle \xi(t)\xi(t')\rangle = 2\gamma k_B T\,\delta(t-t'),

with the noise strength fixed, by the fluctuation-dissipation balance
implicit in this equation, to the same friction coefficient
:math:`\gamma` that damps the systematic motion. Overdamped and
integrated, the Langevin equation reproduces exactly Einstein's
:math:`\langle x^2\rangle = 2Dt` law and its diffusion coefficient, but
it does something Einstein's diffusion-equation picture cannot: it
generates individual, simulatable stochastic trajectories, one random
force realization at a time. That trajectory-level stochastic
differential equation is the direct ancestor of essentially all modern
stochastic simulation, including the driven, far-from-equilibrium
protocols behind the Jarzynski equality below.

*Implementation:* :meth:`physicskit.statphys.chapters.nonequilibrium_work.JarzynskiHarmonicTrap.run_protocol`
integrates exactly the overdamped Langevin equation (via
Euler-Maruyama) for a Brownian particle in a moving harmonic trap,
generating individual stochastic trajectories under a systematic drag
and a Gaussian random force of the prescribed
fluctuation-dissipation-balanced strength.

*References:* P. Langevin, "Sur la théorie du mouvement brownien,"
C. R. Acad. Sci. 146, 530-533 (1908).

.. minigallery:: ../../examples/statphys/nonequilibrium_work/plot_jarzynski_equality.py

1920-1925 -- The Ising Model
----------------------------

Wilhelm Lenz proposed, and his student Ernst Ising solved (in his 1924
doctoral thesis, published in 1925), the simplest possible model of a
magnet: spins :math:`s_i = \pm 1` on a lattice, coupled to their nearest
neighbors, with Hamiltonian :math:`H = -J\sum_{\langle i,j\rangle}
s_i s_j`. Ising's exact one-dimensional solution found no phase
transition at any nonzero temperature -- a disappointing result he
(incorrectly) suspected would hold in every dimension, very nearly
burying the model before it had a chance to prove its worth. In two and
more dimensions, as would only be settled two decades later, the model
does order ferromagnetically below a critical temperature, and it has
since become the single most-studied model in statistical mechanics: a
minimal system exhibiting a genuine continuous phase transition, spontaneous
symmetry breaking, and (as later understood) universal critical
behavior shared with wildly different physical systems.

*References:* W. Lenz, Phys. Z. 21, 613-615 (1920); E. Ising,
"Beitrag zur Theorie des Ferromagnetismus," Z. Phys. 31, 253-258
(1925).

.. minigallery::
   ../../examples/statphys/ising/plot_ising_phase_transition.py
   ../../examples/statphys/ising/plot_ising_criticality_animation.py

1924-1926 -- Bose-Einstein and Fermi-Dirac Quantum Statistics
-------------------------------------------------------------

Satyendra Nath Bose's 1924 rederivation of Planck's radiation law -- by
treating photons as indistinguishable, unlabeled particles counted purely by
how many occupy each state, rather than as classically labeled objects --
caught Einstein's attention enough that he translated it, extended it to
massive particles, and predicted something Bose had not: below a critical
temperature, a macroscopic fraction of the particles collapses into the
single-particle ground state, a new phase of matter with no classical
analogue. This Bose-Einstein condensation (BEC) was not observed directly
until 1995, in dilute alkali-atom gases cooled to nanokelvin temperatures
(Cornell and Wieman; Ketterle independently; 2001 Nobel Prize). Enrico Fermi
and Paul Dirac found, independently and almost simultaneously (1926, though
Fermi's paper was received some six months before Dirac's), the
complementary statistics obeyed by indistinguishable particles subject to
the Pauli exclusion principle: no state may ever hold more than one such
particle, which sharpens into a hard Fermi surface at :math:`T=0` and
underlies the stability of ordinary matter, the behavior of metals, and the
structure of white dwarfs and neutron stars.

.. math::

   n_{\text{BE}}(\varepsilon) = \frac{1}{e^{(\varepsilon-\mu)/k_B T} - 1},
   \qquad
   n_{\text{FD}}(\varepsilon) = \frac{1}{e^{(\varepsilon-\mu)/k_B T} + 1},

both of which reduce to the classical Maxwell-Boltzmann distribution
:math:`n \approx e^{-(\varepsilon-\mu)/k_B T}` when occupation numbers are
small (:math:`\varepsilon - \mu \gg k_B T`). For a homogeneous ideal Bose
gas, the condensate fraction below :math:`T_c` follows

.. math::

   \frac{N_0}{N} = 1 - \left(\frac{T}{T_c}\right)^{3/2}, \qquad T < T_c.

*Implementation:* :func:`physicskit.statphys.utils.thermodynamics.bose_einstein_occupation`
and :func:`~physicskit.statphys.utils.thermodynamics.fermi_dirac_occupation`
compute the two quantum occupation numbers directly;
:func:`~physicskit.statphys.utils.thermodynamics.bec_condensate_fraction`
reproduces the ideal-gas condensate curve confirmed in the 1995 experiments,
and :func:`~physicskit.statphys.utils.thermodynamics.maxwell_boltzmann_speed_pdf`
gives the classical high-temperature limit both quantum distributions
converge to.

*References:* S. N. Bose, Z. Phys. 26, 178-181 (1924); A. Einstein,
Sitzungsber. Preuss. Akad. Wiss., 261-267 (1924), and 3-14 (1925); E.
Fermi, Rend. Lincei 3, 145-149 (1926); P. A. M. Dirac, Proc. R. Soc. A
112, 661-677 (1926); M. H. Anderson et al., Science 269, 198-201
(1995); K. B. Davis et al., Phys. Rev. Lett. 75, 3969-3973 (1995).

.. minigallery:: ../../examples/statphys/quantum_statistics/plot_bose_fermi_distributions.py

1937 -- Landau Mean-Field Theory
---------------------------------

Rather than derive a phase transition from a microscopic Hamiltonian, Lev
Landau asked what the free energy itself must look like near a continuous
transition, given only the symmetry of an order parameter :math:`m` that
the disordered phase forbids from appearing at odd powers. The simplest
possible expansion consistent with that symmetry,

.. math::

   F(m, T) = a(T - T_C)\, m^2 + b\, m^4 - h\, m, \qquad a, b > 0,

reproduces, from analyticity and symmetry alone, the qualitative shape of
every continuous transition: a single minimum at :math:`m=0` above
:math:`T_C`, a spontaneously broken pair of minima below it, and a
susceptibility diverging on both sides. Landau theory's mean-field
critical exponents (:math:`\beta = 1/2`, :math:`\gamma = 1`) are
quantitatively wrong in low dimensions -- Onsager's exact solution below
gives :math:`\beta = 1/8` in two dimensions instead -- a discrepancy the
renormalization group would not fully explain for another three decades.
But the qualitative order-parameter picture Landau introduced, and the
idea of classifying transitions by how they break symmetry, underlies the
entire modern theory of phase transitions built on top of it.

*Implementation:* :func:`physicskit.statphys.utils.landau_theory.landau_free_energy`
evaluates :math:`F(m,T)` directly;
:func:`~physicskit.statphys.utils.landau_theory.landau_equilibrium_magnetization`
finds its minimizing :math:`m(T)`, reproducing the mean-field
:math:`\beta=1/2` power law, and
:func:`~physicskit.statphys.utils.landau_theory.landau_susceptibility`
gives the diverging zero-field susceptibility on both sides of :math:`T_C`.
The same minimizer, with nonzero ``h``, also traces the field-driven
:math:`m(h)` response at fixed :math:`T`, showing the globally stable
branch responding smoothly above :math:`T_C` but jumping discontinuously
through :math:`h=0` below it.

*References:* L. Landau, Zh. Eksp. Teor. Fiz. 7, 19-32, and 627-632
(1937).

.. minigallery:: ../../examples/statphys/landau_theory/plot_landau_mean_field.py

1944 -- Onsager's Exact Solution of the 2D Ising Model
------------------------------------------------------

Lars Onsager achieved what many considered impossible: an exact,
closed-form solution of the two-dimensional Ising model in zero field,
computing its free energy and locating a genuine second-order phase
transition at

.. math::

   \sinh\!\left(\frac{2J}{k_B T_c}\right) = 1
   \quad\Longleftrightarrow\quad
   T_c = \frac{2J}{k_B \ln(1+\sqrt2)} \approx 2.269\,\frac{J}{k_B}.

Onsager's solution remains one of the few exactly solved interacting
models in statistical mechanics, and the definitive proof that Ising's
pessimistic one-dimensional intuition does not generalize: dimension
matters enormously to whether a phase transition can occur at all.

*Implementation:* :class:`physicskit.statphys.chapters.ising_lattice.Ising2D`
implements the model with both single-spin Metropolis and Wolff-cluster
dynamics; its
:attr:`~physicskit.statphys.chapters.ising_lattice.Ising2D.T_C` property
is Onsager's exact critical temperature, reproduced to machine
precision from the closed-form expression above.

*References:* L. Onsager, "Crystal Statistics. I. A Two-Dimensional
Model with an Order-Disorder Transition," Phys. Rev. 65, 117-149
(1944).

.. minigallery:: ../../examples/statphys/ising/plot_ising_phase_transition.py

1952 -- The Yang-Lee Circle Theorem
------------------------------------

A finite system's partition function is a finite sum of manifestly
analytic terms -- so where can a genuine phase transition, a true
discontinuity in the free energy, possibly come from? Tsung-Dao Lee and
Chen-Ning Yang's answer was to continue the partition function into the
*complex* fugacity plane and study its zeros. Writing the Ising
Hamiltonian with an external field, :math:`H = -J\sum_{\langle i,j
\rangle} s_i s_j - h \sum_i s_i`, and the fugacity :math:`z = e^{2\beta
h}`, the partition function is (up to a nonvanishing analytic prefactor) a
degree-:math:`N` polynomial in :math:`z`. Lee and Yang proved that for a
ferromagnetic system, every one of that polynomial's zeros lies exactly on
the unit circle,

.. math::

   |z| = 1 \quad \text{for every root of } Z(z), \qquad J \ge 0,

no matter the system size -- and that a phase transition occurs, in the
:math:`N \to \infty` limit, exactly where these zeros pinch the positive
real axis at :math:`z=1` (:math:`h=0`). This gave the field its first
rigorous, model-independent picture of *why* a phase transition is
mathematically possible at all, despite every finite-system ingredient
being perfectly smooth.

*Implementation:* :func:`physicskit.statphys.utils.partition_function.ising_partition_polynomial`
enumerates a small Ising chain's :math:`2^N` configurations exactly and
returns :math:`Z` as a polynomial in the fugacity, and
:func:`~physicskit.statphys.utils.partition_function.yang_lee_zeros` finds
that polynomial's roots directly, reproducing the exact unit circle to
machine precision for any ferromagnetic coupling.

*References:* C. N. Yang and T. D. Lee, "Statistical Theory of
Equations of State and Phase Transitions. I. Theory of Condensation,"
Phys. Rev. 87, 404-409 (1952); T. D. Lee and C. N. Yang, "Statistical
Theory of Equations of State and Phase Transitions. II. Lattice Gas
and Ising Model," Phys. Rev. 87, 410-419 (1952).

.. minigallery:: ../../examples/statphys/yang_lee/plot_yang_lee_zeros.py

1952 -- The Potts Model
-----------------------

Renfrey Potts, in his 1951 Cambridge doctoral thesis (published 1952,
building on a suggestion by his advisor Cyril Domb), generalized Ising's
two-state spin to :math:`q` discrete states, rewarding neighboring sites
only for matching exactly rather than for a continuous degree of alignment.
This single generalization organizes an entire family of transitions within
one model: :math:`q=2` recovers the Ising model exactly (with a rescaled
coupling), while on the square lattice the transition sharpens from
continuous (second order) at :math:`q \le 4` to discontinuous (first order)
at :math:`q > 4` -- making the Potts model the cleanest available laboratory
for watching a phase transition's *order itself* change as a single
parameter is dialed. Its random-cluster (Fortuin-Kasteleyn) representation
later became a bridge connecting statistical mechanics to graph theory and
the Tutte polynomial.

.. math::

   H = -J \sum_{\langle i,j \rangle} \delta(s_i, s_j), \qquad
   s_i \in \{0, 1, \ldots, q-1\}.

*Implementation:* :class:`physicskit.statphys.chapters.ising_lattice.PottsModel2D`
implements exactly this q-state generalization; its
:attr:`~physicskit.statphys.chapters.ising_lattice.PottsModel2D.T_C` property
gives the exact square-lattice critical temperature
:math:`J/(k_B \ln(1+\sqrt q))`, and
:meth:`~physicskit.statphys.chapters.ising_lattice.PottsModel2D.order_parameter`
tracks the rescaled majority-state fraction whose transition sharpens from
continuous to discontinuous as :math:`q` increases past 4.

*References:* R. B. Potts, Proc. Cambridge Phil. Soc. 48, 106-109
(1952).

.. minigallery:: ../../examples/statphys/potts_model/plot_potts_order_of_transition.py

1953 -- The Metropolis Algorithm
--------------------------------

Nicholas Metropolis, Arianna and Marshall Rosenbluth, and Augusta and
Edward Teller introduced a way to sample a canonical (Gibbs) ensemble
without ever computing its normalizing partition function: propose a
random local change, and accept it with probability
:math:`\min(1, e^{-\beta\Delta E})`. Iterated, this Markov chain
converges to exactly the Boltzmann-weighted equilibrium distribution --
and turned statistical mechanics from a field of exactly solvable
special cases (like Onsager's) into one where almost any model could be
simulated on a computer. It remains, seventy years later, the algorithm
underneath essentially every Monte Carlo statistical-mechanics
calculation, including every lattice-model chapter in this package.

*Implementation:* :func:`physicskit.statphys.core.monte_carlo.metropolis_sweep_ising`
and its counterparts
:func:`~physicskit.statphys.core.monte_carlo.metropolis_sweep_potts`,
:func:`~physicskit.statphys.core.monte_carlo.metropolis_sweep_xy`, and
:func:`~physicskit.statphys.core.monte_carlo.metropolis_sweep_spin_glass`
are the Numba-JIT-compiled Metropolis-Hastings kernels driving every
lattice-model chapter (:class:`~physicskit.statphys.chapters.ising_lattice.Ising2D`,
:class:`~physicskit.statphys.chapters.ising_lattice.PottsModel2D`,
:class:`~physicskit.statphys.chapters.ising_lattice.XYModel2D`,
:class:`~physicskit.statphys.chapters.spin_glass.EdwardsAndersonSpinGlass2D`)
in the package.

*References:* N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H.
Teller, and E. Teller, "Equation of State Calculations by Fast
Computing Machines," J. Chem. Phys. 21, 1087-1092 (1953).

.. minigallery:: ../../examples/statphys/ising/plot_critical_slowing_down.py

1957 -- Molecular Dynamics Simulation
-------------------------------------

Berni Alder and Thomas Wainwright took a different route to the same
goal: rather than sampling configurations stochastically, integrate
Newton's equations for hundreds of interacting hard-sphere particles
directly, and let equilibrium statistics emerge from the trajectory
itself. Their 1957 discovery of a fluid-solid phase transition in a
system of *pure* hard spheres, with no attractive interaction at all,
was itself a landmark: order could arise from packing constraints
alone. Molecular dynamics -- generalized to continuous potentials like
the Lennard-Jones interaction, and put on a firm symplectic footing with
integrators such as Velocity Verlet -- remains the direct-simulation
complement to Metropolis-style Monte Carlo throughout physics,
chemistry, and biology.

*Implementation:* :class:`physicskit.statphys.chapters.molecular_dynamics.LennardJonesGas`
integrates exactly this kind of system via
:func:`physicskit.statphys.core.md_engine.lj_forces` and
:func:`~physicskit.statphys.core.md_engine.velocity_verlet_step`,
letting an arbitrary non-equilibrium initial velocity distribution
relax toward the Maxwell-Boltzmann form under real Newtonian dynamics.

*References:* B. J. Alder and T. E. Wainwright, J. Chem. Phys. 27,
1208-1209 (1957).

.. minigallery:: ../../examples/statphys/molecular_dynamics/plot_maxwell_boltzmann_relaxation.py

1957 -- Percolation Theory
--------------------------

Simon Broadbent and John Hammersley introduced percolation theory to
model fluid flow through a random porous medium: occupy each site (or
bond) of a lattice independently with probability :math:`p`, and ask
whether a connected path spans the system. Below a sharp threshold
:math:`p_c` only finite clusters exist; above it, an infinite spanning
cluster appears with probability one. Unlike a thermal phase
transition, percolation has no energy scale or temperature at all -- the
transition is purely geometric -- yet it displays the full apparatus of
critical phenomena: a sharp threshold, power-law cluster-size
distributions, and universal critical exponents, making it the cleanest
possible laboratory for studying criticality itself.

*Implementation:* :class:`physicskit.statphys.chapters.percolation.Percolation2D`
generates site and bond percolation configurations and labels their
clusters via :func:`~physicskit.statphys.chapters.percolation.hoshen_kopelman`;
its :attr:`~physicskit.statphys.chapters.percolation.Percolation2D.p_c`
property gives the known square-lattice thresholds (:math:`p_c \approx
0.592746` for sites, :math:`p_c = 0.5` exactly for bonds, by
self-duality).

*References:* S. R. Broadbent and J. M. Hammersley, Proc. Cambridge
Phil. Soc. 53, 629-641 (1957).

.. minigallery:: ../../examples/statphys/percolation/plot_percolation_threshold.py

1971 -- Wilson's Renormalization Group
--------------------------------------

Leo Kadanoff's 1966 block-spin picture -- group a lattice into blocks,
replace each block by a single effective spin, and ask how the
effective coupling changes -- gave universality and scaling an intuitive
geometric story. Kenneth Wilson turned that picture into a rigorous
calculational framework, the renormalization group, showing that
repeated coarse-graining defines a flow on the space of possible
Hamiltonians whose fixed points -- ordered, disordered, and critical --
organize all of critical phenomena, and explain why wildly different
microscopic systems can share identical critical exponents (the same
"universality class"). The renormalization group, for which Wilson
received the 1982 Nobel Prize, is now understood as one of the deepest
organizing ideas in all of theoretical physics, far beyond its
statistical-mechanics origin.

*Implementation:* :class:`physicskit.statphys.chapters.renormalization.BlockSpinRG`
implements exactly Kadanoff's :math:`2\times2` majority-rule
block-spin transformation on an Ising configuration;
:meth:`~physicskit.statphys.chapters.renormalization.BlockSpinRG.coarse_grain_step`
performs one coarse-graining step and
:meth:`~physicskit.statphys.chapters.renormalization.BlockSpinRG.iterate`
follows the resulting flow toward the ordered, disordered, or critical
fixed point.

*References:* L. P. Kadanoff, Physics 2, 263-272 (1966); K. G. Wilson,
Phys. Rev. B 4, 3174-3183 (1971), and 3184-3205 (1971).

.. minigallery:: ../../examples/statphys/renormalization/plot_block_spin_rg_flow.py

1971-1973 -- Vortex Unbinding in the 2D XY Model
--------------------------------------------------

Vadim Berezinskii showed in 1971 (in an English translation of a 1970
Russian original) that the two-dimensional XY model of planar spins
:math:`\theta_i \in [0, 2\pi)` -- forbidden by the Mermin-Wagner theorem
from ever developing conventional long-range order -- nonetheless
undergoes a genuine phase transition, driven entirely by the unbinding
of topological vortex-antivortex pairs; John Kosterlitz and David
Thouless, working independently and arriving at the transition two
years later, in 1973, gave the mechanism its now-standard, more
complete treatment, with Kosterlitz following up in 1974 with the
renormalization-group analysis of the unbinding itself. Below the
transition temperature, vortices remain bound in tightly circling
pairs of opposite topological charge; above it, thermal fluctuations
tear these pairs apart into a free "vortex plasma," destroying the
algebraic quasi-long-range order of the low-temperature phase.
Kosterlitz and Thouless shared the 2016 Nobel Prize in part for this
work (Berezinskii, who died in 1980, was by then long ineligible);
the transition is nonetheless routinely credited to all three names
today, and is often written "BKT" in recognition of Berezinskii's
priority.

*Implementation:* :class:`physicskit.statphys.chapters.ising_lattice.XYModel2D`
simulates exactly this lattice of planar spins; its
:attr:`~physicskit.statphys.chapters.ising_lattice.XYModel2D.T_KT`
property gives the numerically established transition temperature
(:math:`\approx 0.893\,J/k_B`), and
:meth:`~physicskit.statphys.chapters.ising_lattice.XYModel2D.vorticity`
(backed by
:func:`physicskit.statphys.core.monte_carlo.xy_plaquette_vorticity`)
measures the integer topological charge of every plaquette directly,
making the bound-pair-to-plasma unbinding visible configuration by
configuration.

*References:* V. L. Berezinskii, Sov. Phys. JETP 32, 493-500 (1971)
(English translation of the 1970 Russian original); J. M. Kosterlitz
and D. J. Thouless, J. Phys. C 6, 1181-1203 (1973); J. M. Kosterlitz,
J. Phys. C 7, 1046-1060 (1974).

.. minigallery:: ../../examples/statphys/xy_model/plot_kt_vortex_unbinding.py

1975 -- The Edwards-Anderson Spin Glass
---------------------------------------

Sam Edwards and Philip Anderson proposed that magnetic alloys with random,
competing ferromagnetic and antiferromagnetic interactions freeze, below a
sharp temperature, into a disordered but static configuration -- a "spin
glass" -- held together not by translational order but by the sheer
geometric impossibility of satisfying every bond at once. A plaquette with
an odd number of antiferromagnetic bonds is *frustrated*: no single spin
configuration minimizes all of its bonds simultaneously, and the resulting
ground state is highly degenerate, dominated by a rugged hierarchy of
nearly-equal-energy valleys rather than one clean minimum. Detecting the
transition demanded a genuinely new order parameter, since the ordinary
magnetization stays zero on both sides of it: Edwards and Anderson compared
two independently thermalized replicas subject to the *same* quenched
disorder, whose spin overlap freezes to a nonzero value below the
spin-glass transition and decorrelates to zero above it.

.. math::

   H = -\sum_{\langle i,j \rangle} J_{ij}\, s_i s_j, \qquad J_{ij} = \pm J
   \ \text{(quenched, random)},

.. math::

   q = \frac{1}{N}\sum_i s_i^{(1)} s_i^{(2)}.

*Implementation:* :class:`physicskit.statphys.chapters.spin_glass.EdwardsAndersonSpinGlass2D`
implements exactly this :math:`\pm J` bimodal-disorder Ising spin glass;
:meth:`~physicskit.statphys.chapters.spin_glass.EdwardsAndersonSpinGlass2D.frustration_density`
measures the fixed, geometric fraction of frustrated plaquettes, and
:meth:`~physicskit.statphys.chapters.spin_glass.EdwardsAndersonSpinGlass2D.edwards_anderson_order_parameter`
estimates :math:`\langle q^2 \rangle` from two independently thermalized
replicas sharing the same bonds.

*References:* S. F. Edwards and P. W. Anderson, J. Phys. F 5, 965-974
(1975).

.. minigallery:: ../../examples/statphys/spin_glass/plot_edwards_anderson_frustration.py

1975-1980 -- The Sherrington-Kirkpatrick Model and Parisi's Replica Symmetry Breaking
---------------------------------------------------------------------------------------

The same year as Edwards and Anderson's short-range model, David
Sherrington and Scott Kirkpatrick proposed the opposite extreme: every
spin coupled to *every* other spin, with independent Gaussian bonds,

.. math::

   H = -\sum_{i<j} J_{ij}\, s_i s_j, \qquad
   J_{ij} \sim \mathcal{N}\!\left(0, \frac{J^2}{N}\right).

Its infinite-range connectivity made the model look exactly solvable by
the replica trick -- until the naive replica-symmetric ansatz produced a
manifestly unphysical negative entropy at low temperature, a glaring sign
that something in the calculation was wrong. Giorgio Parisi's 1979-1980
resolution was to break replica symmetry itself: rather than one frozen
ground state, the low-temperature phase is an infinite, hierarchically
nested family of pure states, organized by a nontrivial, sample-dependent
distribution :math:`P(q)` of the replica overlap

.. math::

   q = \frac{1}{N}\sum_i s_i^{(1)} s_i^{(2)}

rather than the single frozen value the Edwards-Anderson picture predicts.
Parisi's replica-symmetry-breaking solution -- rigorously proven correct
only decades later (Talagrand, 2006; Guerra's bound, 2003) -- earned him a
share of the 2021 Nobel Prize, and its hierarchical "ultrametric" structure
went on to find applications far outside physics, from neural networks and
combinatorial optimization to protein folding.

*Implementation:* :class:`physicskit.statphys.chapters.spin_glass.SherringtonKirkpatrick`
implements exactly this fully-connected Gaussian-disorder spin glass;
:meth:`~physicskit.statphys.chapters.spin_glass.SherringtonKirkpatrick.overlap_distribution`
pools replica-overlap samples across many independent disorder
realizations, exposing -- in a finite-size Monte Carlo simulation -- the
broad, non-self-averaging :math:`P(q)` that is the numerically observable
fingerprint of Parisi's replica-symmetry-breaking solution.

*References:* D. Sherrington and S. Kirkpatrick, Phys. Rev. Lett. 35,
1792-1796 (1975); G. Parisi, Phys. Rev. Lett. 43, 1754-1756 (1979),
and J. Phys. A 13, L115-L121 (1980); F. Guerra, Commun. Math. Phys.
233, 1-12 (2003); M. Talagrand, Ann. Math. 163, 221-263 (2006).

.. minigallery:: ../../examples/statphys/spin_glass/plot_sk_overlap_distribution.py

1981 -- The Binder Cumulant and Finite-Size Scaling
---------------------------------------------------

Kurt Binder introduced a scale-independent diagnostic for locating a
continuous phase transition from necessarily finite simulations: the
fourth-order cumulant of the order parameter, dimensionless by
construction, whose curves at different lattice sizes :math:`L` all cross
at very nearly the same point -- the infinite-volume critical temperature
-- because at criticality :math:`U_4` depends on :math:`L` only through the
universal combination :math:`L^{1/\nu}`. This turned finite-size effects,
previously seen only as an inconvenient rounding of what should be a sharp
singularity, into a precision tool: simulating a handful of modest lattice
sizes and finding where their Binder-cumulant curves cross extracts both
:math:`T_C` and the critical exponents without ever having to simulate an
infinite system.

.. math::

   U_4(T, L) = 1 - \frac{\langle M^4 \rangle}{3 \langle M^2 \rangle^2},
   \qquad
   \chi_{\max}(L) \sim L^{\gamma/\nu}, \qquad
   M(T_C, L) \sim L^{-\beta/\nu}.

*Implementation:* :func:`physicskit.statphys.utils.thermodynamics.binder_cumulant`
computes :math:`U_4` from a magnetization sample; the
:mod:`physicskit.statphys.utils.finite_size_scaling` module implements the
full analysis built on it --
:func:`~physicskit.statphys.utils.finite_size_scaling.binder_cumulant_crossing`
locates :math:`T_C` as the crossing of two lattice sizes' :math:`U_4(T)`
curves, and
:func:`~physicskit.statphys.utils.finite_size_scaling.estimate_gamma_over_nu`,
:func:`~physicskit.statphys.utils.finite_size_scaling.estimate_beta_over_nu`,
and
:func:`~physicskit.statphys.utils.finite_size_scaling.estimate_nu_from_binder_slope`
extract the critical exponents from the corresponding finite-size scaling
laws.

*References:* K. Binder, Z. Phys. B 43, 119-140 (1981).

.. minigallery:: ../../examples/statphys/ising/plot_finite_size_scaling.py

1986 -- The Kardar-Parisi-Zhang Equation
------------------------------------------

Mehran Kardar, Giorgio Parisi, and Yi-Cheng Zhang proposed a stochastic
partial differential equation for a generic growing interface -- crystal
surfaces, bacterial colonies, burning paper fronts -- driven by
uncorrelated noise, smoothed by surface tension, and locally accelerated
by growth normal to the surface rather than straight up:

.. math::

   \frac{\partial h}{\partial t} = \nu \nabla^2 h
   + \frac{\lambda}{2}(\nabla h)^2 + \eta(x, t).

That nonlinear :math:`(\nabla h)^2` term is what separates the KPZ
universality class from the simpler, linear Edwards-Wilkinson equation
(:math:`\lambda=0`): it couples length scales together and forces a wholly
new set of scaling exponents on the interface width :math:`w(t) =
\sqrt{\langle(h-\bar h)^2\rangle}`, growing as :math:`w \sim t^\beta`
(:math:`\beta=1/3` in 1+1 dimensions) before saturating at
:math:`w_{\text{sat}} \sim L^\alpha` (:math:`\alpha=1/2`) once the system
size is reached. These exponents turned out to be extraordinarily
universal, appearing (both theoretically and experimentally) in systems
with no obvious surface at all, from directed polymers to burning fronts,
and to have deep, still-active connections to random matrix theory.

*Implementation:* :func:`physicskit.statphys.core.kpz_engine.run_rsos` and
:class:`physicskit.statphys.chapters.kpz_growth.KPZInterface` implement
the Restricted Solid-On-Solid growth automaton, an exact discrete model
with no adjustable surface-tension or coupling constants that is long
established to lie in the KPZ universality class;
:meth:`~physicskit.statphys.chapters.kpz_growth.KPZInterface.run_growth_curve`
recovers both the :math:`\beta=1/3` growth exponent and the
:math:`\alpha=1/2` saturation exponent directly from simulation.

*References:* M. Kardar, G. Parisi, and Y.-C. Zhang, Phys. Rev. Lett.
56, 889-892 (1986).

.. minigallery:: ../../examples/statphys/kpz/plot_kpz_roughening.py

1987 -- Self-Organized Criticality and the BTW Sandpile
-------------------------------------------------------

Per Bak, Chao Tang, and Kurt Wiesenfeld asked whether a system could
reach a critical state -- power-law-distributed fluctuations at every
scale -- *without* anyone fine-tuning a parameter to a special value,
the way temperature must be tuned to :math:`T_c` in the Ising model.
Their sandpile automaton answers yes: grains added one at a time to a
lattice topple whenever a site's height exceeds a threshold, and the
slow-driving, fast-relaxation dynamics tunes the pile *itself* to a
critical state, with avalanche sizes following a power law with no
characteristic scale. "Self-organized criticality" was proposed as a
generic mechanism for the ubiquity of power laws in nature -- earthquakes,
forest fires, neural avalanches -- and remains one of the most
influential, and most debated, ideas to come out of statistical physics
in the last fifty years.

*Implementation:* :class:`physicskit.statphys.chapters.sandpile.BTWSandpile`
implements exactly this toppling automaton;
:meth:`~physicskit.statphys.chapters.sandpile.BTWSandpile.run` drives
the pile with a stream of randomly placed grains and records each
one's avalanche size, backed by
:func:`physicskit.statphys.core.sandpile_engine.topple_to_stability`.
Repeating this at several grid sizes and fitting the growth of the largest
observed avalanche with
:func:`~physicskit.statphys.utils.finite_size_scaling.power_law_exponent`
recovers the finite-size cutoff's own scaling with :math:`L`.

*References:* P. Bak, C. Tang, and K. Wiesenfeld, Phys. Rev. Lett. 59,
381-384 (1987).

.. minigallery:: ../../examples/statphys/sandpile/plot_btw_avalanches.py

1989 -- The Wolff Cluster Algorithm
-----------------------------------

Even Metropolis dynamics suffers from *critical slowing down*: single-spin
flips decorrelate a configuration only diffusively, so the autocorrelation
time diverges as a power of the correlation length exactly where accurate
statistics matter most, right at :math:`T_C`. Robert Swendsen and Jian-Sheng
Wang's 1987 cluster algorithm broke this bottleneck by growing and flipping
whole clusters of bond-activated, same-sign spins in a single move; Ulli
Wolff's 1989 refinement grows and flips just one such cluster per update,
seeded at a random site, with each bond added independently with
probability :math:`p_{\text{add}} = 1 - e^{-2\beta J}`. Because a Wolff
cluster is, on average, exactly a correlated domain, flipping it is a
genuinely global Monte Carlo move rather than a local one; the algorithm's
dynamical critical exponent is dramatically smaller than Metropolis's,
turning simulations right at :math:`T_C` -- previously the hardest regime to
sample -- into one of the easiest.

.. math::

   p_{\text{add}} = 1 - e^{-2\beta J}.

*Implementation:* :func:`physicskit.statphys.core.monte_carlo.wolff_step_ising`
implements exactly this cluster growth and flip, and backs
:meth:`physicskit.statphys.chapters.ising_lattice.Ising2D.sweep`
(``algorithm="wolff"``); it is also what
:class:`~physicskit.statphys.chapters.renormalization.BlockSpinRG` uses to
equilibrate its initial fine-grained configuration reliably even deep below
:math:`T_C`, where single-spin Metropolis dynamics would stall on slow
domain-wall coarsening.

*References:* R. H. Swendsen and J.-S. Wang, Phys. Rev. Lett. 58,
86-88 (1987); U. Wolff, Phys. Rev. Lett. 62, 361-364 (1989).

.. minigallery:: ../../examples/statphys/ising/plot_critical_slowing_down.py

1997 -- The Jarzynski Equality
--------------------------------

The second law only bounds the *average* work needed to drive a system
between two equilibrium states, :math:`\langle W \rangle \ge \Delta F`,
with equality only in the reversible, infinitely slow limit -- any
individual fast, irreversible realization tells you nothing exact about
:math:`\Delta F` on its own. Christopher Jarzynski found an exact equality
hiding behind that inequality: averaging not :math:`W` itself but
:math:`e^{-\beta W}` over an ensemble of repeated, arbitrarily fast
realizations of the same protocol recovers the equilibrium free energy
difference exactly,

.. math::

   e^{-\beta \Delta F} = \left\langle e^{-\beta W} \right\rangle,

no matter how far from equilibrium any individual trajectory is driven.
Jensen's inequality applied to this identity immediately reproduces the
second law as a corollary rather than an independent postulate --
nonequilibrium work fluctuations turn out to carry exact equilibrium
information. The result kicked off modern nonequilibrium statistical
mechanics, including the closely related Crooks fluctuation theorem
(1999) and single-molecule "pulling" experiments that now measure protein
and RNA folding free energies this way.

*Implementation:* :class:`physicskit.statphys.chapters.nonequilibrium_work.JarzynskiHarmonicTrap`
drags a Brownian particle's harmonic trap at finite speed via overdamped
Langevin dynamics;
:meth:`~physicskit.statphys.chapters.nonequilibrium_work.JarzynskiHarmonicTrap.jarzynski_free_energy_estimate`
recovers the exact :math:`\Delta F = 0` of this translation-invariant trap
(the stiffness never changes, only its center moves) from the same
irreversible work samples whose mean, by the second law, sits strictly
above zero.

*References:* C. Jarzynski, Phys. Rev. Lett. 78, 2690-2693 (1997); G.
E. Crooks, Phys. Rev. E 60, 2721-2726 (1999).

.. minigallery:: ../../examples/statphys/nonequilibrium_work/plot_jarzynski_equality.py

See Also
--------

- :doc:`/api/statphys`
- :doc:`/api/gallery/statphys/index`
- :doc:`/tutorials/statphys_deep_dives`
- :doc:`/history/condensed_breakthroughs`
