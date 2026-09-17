Breakthroughs in Quantum Mechanics
==================================

.. epigraph::

   "I think I can safely say that nobody understands quantum mechanics."
   -- Richard Feynman, *The Character of Physical Law*, 1965

Quantum mechanics replaced a world of definite trajectories with one of
wavefunctions, probabilities, and operators -- and did so not as a single
discovery but as a three-decade argument among the physicists who built it,
from Planck's reluctant quantum in 1900 to Bell's 1964 proof that no
"common-sense" theory could ever reproduce its predictions. The chronology
behind :mod:`physicskit.quantum` follows that argument in order: the old
quantum theory of Bohr, the two rival (and ultimately equivalent)
mechanics of Heisenberg and Schrodinger, the interpretive battles over
what the wavefunction even *means*, and the exactly solvable model
problems -- the hydrogen atom, the harmonic oscillator, the square well --
that remain the field's working vocabulary today. This chronology traces
that thread, with a pointer to the corresponding implementation in this
package at each stop.

.. contents:: Timeline
   :local:
   :depth: 1

1900 -- Planck's Quantum Hypothesis
-----------------------------------

To fit the observed spectrum of blackbody radiation -- which classical
electrodynamics predicted should diverge at short wavelengths (the "ultraviolet
catastrophe") -- Max Planck proposed, in a December 1900 paper to the German
Physical Society, that the energy exchanged between radiation and matter is
not continuous but comes in discrete quanta,

.. math::

   E_n = n h \nu, \qquad n = 0, 1, 2, \dots,

for an oscillator of frequency :math:`\nu`. Planck himself considered this a
mathematical device rather than a physical claim -- he later called it "an
act of desperation" -- but the idea that a harmonic oscillator's energy
spectrum is discrete, evenly spaced by :math:`h\nu` (or, in the full quantum
treatment developed a generation later, by :math:`\hbar\omega`), turned out
to be exactly right, and became the seed of everything that follows in this
chronology.

*Connection:* :meth:`physicskit.quantum.chapters.harmonic_spin.HarmonicOscillator.energy`
returns exactly this evenly spaced spectrum, :math:`E_n = \hbar\omega(n +
\tfrac12)`, for the quantum harmonic oscillator that eventually made
Planck's quantization rule precise.

*References:* M. Planck, Verhandlungen der Deutschen Physikalischen
Gesellschaft 2, 237-245 (1900).

.. minigallery:: ../../examples/quantum/harmonic_oscillator/plot_ladder_operators.py

1913 -- Bohr's Atomic Model
---------------------------

Niels Bohr proposed that an electron orbiting a nucleus is restricted to a
discrete set of "stationary states" with quantized angular momentum
:math:`L = n\hbar`, in which -- contrary to classical electrodynamics -- it
does not radiate, jumping between orbits only by emitting or absorbing a
photon of energy :math:`\Delta E`. For hydrogen this postulate, combined
with a classical circular-orbit force balance, gives the energy levels

.. math::

   E_n = -\frac{1}{2n^2}\,\frac{m e^4}{\hbar^2} \quad\text{(Hartree atomic units)},

reproducing the empirical Rydberg formula for hydrogen's spectral lines to
remarkable precision. Bohr's model was frankly *ad hoc* -- a classical orbit
with a quantization rule bolted on -- but it was the first successful
quantum theory of an atom, and every one of its energy levels survives
unchanged in the exact 1926 treatment that replaced it.

*Connection:* :class:`physicskit.quantum.chapters.hydrogen_am.HydrogenOrbital`
reproduces Bohr's :math:`n`-dependence exactly in its ``energy`` property,
:math:`E_n = -Z^2/(2n^2)`, even though it is derived from the full
Schrodinger equation rather than Bohr's semiclassical orbits. See it in
:doc:`/api/gallery/quantum/hydrogen/plot_hydrogen_orbitals`.

*References:* N. Bohr, "On the Constitution of Atoms and Molecules," Phil.
Mag. Ser. 6, 26, 1-25, 476-502, 857-875 (1913).

.. minigallery:: ../../examples/quantum/hydrogen/plot_hydrogen_orbitals.py

1913 -- The Zeeman and Stark Effects
------------------------------------

Pieter Zeeman had already shown, in 1896, that spectral lines split into
several components in a magnetic field (Nobel Prize, 1902, shared with
Hendrik Lorentz, who supplied the classical explanation). In 1913 Johannes
Stark discovered the electrical analogue: hydrogen's Balmer lines split
linearly under a strong external electric field. Both effects became
essential tools of "old quantum theory" spectroscopy, and both later
became textbook applications of perturbation theory -- the Zeeman effect
of non-degenerate perturbation theory, and the *linear* Stark effect of
hydrogen specifically of the *degenerate* variety, made possible only by
hydrogen's accidental level degeneracy in :math:`l`.

*Implementation:* :func:`physicskit.quantum.chapters.perturbation.zeeman_splitting`
and :func:`~physicskit.quantum.chapters.perturbation.zeeman_spectrum`
compute the first-order Zeeman sublevel shifts;
:func:`~physicskit.quantum.chapters.perturbation.linear_stark_shift` and
:func:`~physicskit.quantum.chapters.perturbation.stark_n2_quartet`
reproduce the classic linear Stark quartet of hydrogen's :math:`n=2` shell.

*References:* P. Zeeman, Phil. Mag. 43, 226-239 (1897) (originally published
in Dutch in 1896); J. Stark, Sitzungsber. Preuss. Akad. Wiss. 1913, 932-946.

.. minigallery:: ../../examples/quantum/perturbation/plot_perturbation_and_floquet.py

1915-1916 -- Wilson-Sommerfeld Quantization
-------------------------------------------

William Wilson and Arnold Sommerfeld independently generalized Bohr's
quantization rule from circular orbits to any periodic classical motion,
replacing it with the phase-space integral

.. math::

   \oint p\,dq = n h, \qquad n = 1, 2, 3, \dots,

taken once around a full period. Applied to elliptical Kepler orbits this
reproduced hydrogen's Bohr energies unchanged (adding a second quantum
number for orbital shape), and Sommerfeld's relativistic refinement of the
same rule correctly predicted hydrogen's fine structure -- an early triumph
of "old quantum theory." But the rule only quantizes systems whose
classical motion is periodic and separable; it could not be extended to
the helium atom or to molecular spectra, a failure that motivated the
wholesale replacement Heisenberg and Schrodinger supplied over a decade
later.

*Connection:* :func:`physicskit.semiclassical.core.wkb.bohr_sommerfeld_energies`
implements the modern, Maslov-corrected descendant of this same
phase-space quantization integral,
:math:`\int_{x_1}^{x_2}p\,dx=(n+\tfrac12)\pi\hbar`, turning Wilson and
Sommerfeld's postulate into a working numerical bound-state solver for an
arbitrary one-dimensional potential.

*References:* W. Wilson, Phil. Mag. 29, 795-802 (1915); A. Sommerfeld, Ann.
Phys. 356(17), 1-94 (1916).

.. minigallery:: ../../examples/semiclassical/wkb/plot_wkb_bohr_sommerfeld.py

1922 -- The Stern-Gerlach Experiment
-------------------------------------

Otto Stern proposed, and with Walther Gerlach carried out at the University
of Frankfurt, an experiment to test whether the "space quantization" of old
quantum theory -- the claim that an atom's magnetic moment can point only
along a discrete set of directions relative to an external field, rather
than any classical angle -- was physically real rather than a mathematical
bookkeeping device. They sent a beam of silver atoms, each with one
unpaired valence electron, through a strongly inhomogeneous magnetic field:
a magnetic moment free to point at any angle should smear the beam into one
broadened band, while a moment restricted to a discrete set of orientations
should split it into a corresponding number of separate spots. In the early
hours of 8 February 1922 they found the beam split cleanly in two -- not
smeared, confirming that spatial quantization is real, though the *number*
of spots (two, not the odd count old quantum theory's integer orbital
angular momentum would have implied for silver's ground state) could only
be explained after Uhlenbeck and Goudsmit posited electron spin three years
later, in 1925.

.. math::

   F_y = \pm\, \mu\,\frac{\partial B_z}{\partial y},

the spin-dependent transverse force -- its sign set by which way the atom's
magnetic moment projects along the field gradient -- that pushes the two
spin branches apart into separately resolved lobes. Stern received the
1943 Nobel Prize in Physics for the molecular-beam method this experiment
inaugurated; Gerlach, who did much of the hands-on experimental work
(including the slit refinement that made the clean February 1922 result
possible), was never awarded a share, a historical omission usually
attributed to wartime politics rather than to any dispute over credit.

*Implementation:* :class:`physicskit.quantum.chapters.spin.SternGerlach`
implements the standard semiclassical two-branch treatment: each spin
projection is modeled as an independent Gaussian wavepacket subject to
exactly this constant transverse force, so its
:meth:`~physicskit.quantum.chapters.spin.SternGerlach.joint_density` (and
its time-stacked
:meth:`~physicskit.quantum.chapters.spin.SternGerlach.joint_density_stack`)
show the two lobes separating in real time as the beam propagates through
the apparatus, animated with
:func:`~physicskit.quantum.visualizers.wavefunctions.animate_density_2d`.
See it in :doc:`/api/gallery/quantum/measurement/plot_stern_gerlach`.

*References:* W. Gerlach and O. Stern, Z. Phys. 9, 349-352 (1922);
precursor O. Stern, Z. Phys. 7, 249-253 (1921).

.. minigallery:: ../../examples/quantum/measurement/plot_stern_gerlach.py

1924 -- de Broglie's Matter Waves
---------------------------------

In his doctoral thesis, Louis de Broglie proposed that the wave-particle
duality Einstein had already established for light should run in reverse:
every material particle of momentum :math:`p` has an associated wavelength

.. math::

   \lambda = \frac{h}{p}.

The hypothesis was speculative -- de Broglie had no direct evidence for it
-- until Davisson and Germer observed electron diffraction from a nickel
crystal in 1927, confirming it experimentally. De Broglie's wave is the
first appearance of the traveling-wave factor :math:`e^{ikx}` that appears
in every wavepacket and plane-wave state used throughout quantum mechanics.

*Implementation:* :func:`physicskit.quantum.chapters.wave_packets.free_gaussian_wavepacket`
builds a localized wavepacket carrying exactly this de Broglie phase,
:math:`e^{ik_0 x}`, modulating a Gaussian envelope centered on momentum
:math:`p_0 = \hbar k_0`. The same packet's subsequent free evolution --
:class:`~physicskit.quantum.chapters.wave_packets.GaussianDispersion`, the
exact analytic solution of the free-particle Schrodinger equation for a
Gaussian initial condition -- is animated frame by frame with its
:meth:`~physicskit.quantum.chapters.wave_packets.GaussianDispersion.trajectory`
method and :func:`~physicskit.quantum.visualizers.wavefunctions.animate_density`,
showing the de Broglie phase riding along a center that advances
ballistically at :math:`v=\hbar k_0/m` while the envelope spreads. The
animated dispersion is shown in
:doc:`/api/gallery/quantum/wave_packets/plot_wave_packet_dynamics`.
:class:`~physicskit.quantum.visualizers.phase_space.WignerVisualizer`
renders the same packet's Wigner phase-space distribution directly, a
single blob centered on :math:`(x_0, p_0)` that is position space and
momentum space at once -- the modern phase-space picture of exactly the
wave-particle duality de Broglie proposed.

*References:* L. de Broglie, Ann. Phys. (Paris) 10e série, 3 (1925) (thesis
defended 1924); experimental confirmation C. Davisson and L. Germer, Phys.
Rev. 30, 705-740 (1927).

.. minigallery:: ../../examples/quantum/wave_packets/plot_de_broglie_wavepacket.py

1925-1927 -- Pauli's Exclusion Principle and Spin Matrices
------------------------------------------------------------

Wolfgang Pauli proposed, in 1925, that no two electrons in an atom can
share the same complete set of quantum numbers -- the exclusion principle
that finally explained the shell structure of the periodic table old
quantum theory could only fit by hand. Two years later he made the
electron's spin degree of freedom, already inferred by Uhlenbeck and
Goudsmit from the Stern-Gerlach result, mathematically explicit,
representing a spin-1/2 particle's two-valued internal state with a set of
three :math:`2\times2` matrices,

.. math::

   \sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \qquad
   \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \qquad
   \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix},

satisfying :math:`\sigma_i\sigma_j = \delta_{ij}I + i\epsilon_{ijk}\sigma_k`.
The exclusion principle is a statement about many-electron wavefunctions
(their required antisymmetry under particle exchange, a connection Pauli
himself only made fully explicit with the 1940 spin-statistics theorem);
the Pauli matrices, by contrast, are the working algebra of any single
two-level quantum system, and appear throughout this package wherever a
spin-1/2 or qubit degree of freedom is modeled.

*Implementation:* :data:`physicskit.quantum.core.operators.sigma_x`,
:data:`~physicskit.quantum.core.operators.sigma_y`, and
:data:`~physicskit.quantum.core.operators.sigma_z` are exactly these three
matrices, and :func:`~physicskit.quantum.core.operators.spin_operator`
generalizes them to arbitrary spin quantum number :math:`s`. They are the
computational basis for two-level physics elsewhere in this chronology:
:class:`physicskit.quantum.chapters.spin.RabiProblem` builds its RWA
Hamiltonian and propagator directly from ``sigma_z`` and ``sigma_x``, and
:class:`physicskit.quantum.chapters.entanglement.BellCorrelations` uses the
same two matrices to build its spin-projection measurement operators. The
exclusion principle itself -- multi-electron antisymmetrization -- has no
direct counterpart in this package, which works throughout in the
single- and two-qubit/single-particle regime.

*References:* W. Pauli, "Uber den Zusammenhang des Abschlusses der
Elektronengruppen im Atom mit der Komplexstruktur der Spektren," Z. Phys.
31, 765-783 (1925) (exclusion principle); W. Pauli, "Zur Quantenmechanik
des magnetischen Elektrons," Z. Phys. 43, 601-623 (1927) (spin matrices).

.. minigallery:: ../../examples/quantum/entanglement/plot_bloch_sphere_spin_dynamics.py

1926 -- Schrodinger's Wave Equation and the Hydrogen Atom
---------------------------------------------------------

Erwin Schrodinger, seeking a wave equation whose stationary states would
reproduce Bohr's energy levels, published the time-independent equation

.. math::

   -\frac{\hbar^2}{2m}\nabla^2\psi + V\psi = E\psi

in a rapid sequence of four papers in 1926, and immediately solved it
exactly for the Coulomb potential -- recovering Bohr's hydrogen spectrum
:math:`E_n = -Z^2/(2n^2)` not as a postulate but as an eigenvalue, together
with the full three-dimensional orbital structure Bohr's model could never
supply. Within months Schrodinger also showed his wave mechanics to be
mathematically equivalent to Heisenberg's matrix mechanics, unifying the two
independently-discovered quantum theories.

*Implementation:* :class:`physicskit.quantum.core.eigensolvers.NumerovSolver`
solves exactly this time-independent equation numerically for an arbitrary
1D potential; :func:`physicskit.quantum.chapters.hydrogen_am.radial_wavefunction`
and :func:`~physicskit.quantum.chapters.hydrogen_am.spherical_harmonic`
give the exact analytic 3D hydrogen solution
:math:`\psi_{nlm}=R_{nl}(r)Y_l^m(\theta,\phi)`. A coherent superposition of
two such eigenstates is no longer stationary:
:func:`~physicskit.quantum.chapters.hydrogen_am.orbital_superposition_psi`
and :func:`~physicskit.quantum.chapters.hydrogen_am.orbital_superposition_density`
evolve :math:`\psi = c_a\psi_a e^{-iE_a t/\hbar}+c_b\psi_b e^{-iE_b t/\hbar}`
and show its density beating at the Bohr transition frequency
:math:`\omega_{ab}=(E_a-E_b)/\hbar`, rendered as a Plotly play-button
isosurface animation by
:func:`~physicskit.quantum.visualizers.orbitals.animate_orbital_beating`. The
exact hydrogen solution, including the beating animation, is shown in
:doc:`/api/gallery/quantum/hydrogen/plot_hydrogen_orbitals`.

*References:* four "Mitteilungen" papers, E. Schrodinger, Ann. Phys.
384(4), 361-376; 384(6), 489-527; 385(13), 437-490; 386(18), 109-139
(1926); equivalence proof, Ann. Phys. 384(8), 734-756 (1926).

.. minigallery:: ../../examples/quantum/potentials/plot_bound_states_numerov.py

1926 -- Born's Probabilistic Interpretation
-------------------------------------------

While working out scattering theory, Max Born proposed that Schrodinger's
wavefunction :math:`\psi` has no direct physical reality itself; rather,
:math:`\lvert\psi(x)\rvert^2` is a probability density for finding the
particle at :math:`x` upon measurement,

.. math::

   P(x)\,dx = \lvert\psi(x)\rvert^2\, dx.

This "Born rule" was the interpretive break that made quantum mechanics
irreducibly statistical -- not merely as a matter of incomplete knowledge,
as in classical statistical mechanics, but as a fundamental feature of
nature -- a conclusion Einstein famously never accepted ("God does not
play dice"). Born received the Nobel Prize for this insight in 1954.

*Implementation:* :func:`physicskit.quantum.utils.measure.simulate_position_measurement`
draws simulated measurement outcomes directly from the Born rule,
:math:`x_i \sim \lvert\psi(x)\rvert^2`;
:func:`~physicskit.quantum.utils.measure.position_expectation` computes the
resulting expectation value :math:`\langle x\rangle`.

*References:* M. Born, Z. Phys. 37, 863-867 (1926), corrected and extended
in Z. Phys. 38, 803-827 (1926).

.. minigallery:: ../../examples/quantum/measurement/plot_born_rule_measurement.py

1926 -- The WKB Approximation
-----------------------------

Building on de Broglie's matter wave, Gregor Wentzel, Hendrik Kramers, and
Leon Brillouin -- following Harold Jeffreys' earlier, independent
mathematical treatment of the same asymptotic expansion -- showed in 1926
how to recover Schrodinger's exact wave mechanics as a controlled
semiclassical limit. Writing the wavefunction as an amplitude times a
rapidly oscillating phase,

.. math::

   \psi(x) \sim \frac{1}{\sqrt{p(x)}}\, \exp\!\left(\pm\frac{i}{\hbar}\int p(x')\,dx'\right), \qquad p(x) = \sqrt{2m\bigl(E-V(x)\bigr)},

and expanding order by order in :math:`\hbar`, the WKB approximation
reproduces the classical limit exactly where it should: the amplitude
:math:`p(x)^{-1/2}` is precisely the classical probability of finding a
particle where it moves slowest, and matching the oscillatory solution
through each turning point (via the Airy-function connection formulas)
costs a phase of :math:`\pi/4`, refining Wilson-Sommerfeld's integer
quantization into the half-integer Einstein-Brillouin-Keller (EBK) rule.

*Implementation:* :func:`physicskit.semiclassical.core.wkb.classical_momentum`
and :func:`~physicskit.semiclassical.core.wkb.turning_points` build the
semiclassical momentum :math:`p(x)` and locate its turning points;
:func:`~physicskit.semiclassical.core.wkb.wkb_action` and
:func:`~physicskit.semiclassical.core.wkb.wkb_wavefunction` assemble
the resulting approximate wavefunction, reproducing the node-counting
theorem and the exact harmonic-oscillator spectrum to machine precision.

*References:* G. Wentzel, Z. Phys. 38, 518-529 (1926); H. Kramers, Z. Phys.
39, 828-840 (1926); L. Brillouin, C. R. Acad. Sci. 183, 24-26 (1926);
precursor H. Jeffreys, Proc. London Math. Soc. 23, 428-436 (1925).

.. minigallery:: ../../examples/semiclassical/wkb/plot_wkb_bohr_sommerfeld.py

1927 -- Heisenberg's Uncertainty Principle
------------------------------------------

Werner Heisenberg had already given quantum mechanics its first
formulation in 1925, replacing classical trajectories with infinite arrays
("matrices") of transition amplitudes -- matrix mechanics, in which
position and momentum are non-commuting operators satisfying
:math:`[\hat x,\hat p] = i\hbar`. Heisenberg's initial paper treated only
one degree of freedom and left the general algebra incomplete; its full
development into a systematic many-body matrix mechanics is jointly
credited to Heisenberg, Max Born, and Pascual Jordan, whose "Drei-Manner-
Arbeit" ("three-man paper") supplied the rigorous multi-dimensional
formulation, including the canonical commutation relation in the form
above. In 1927 Heisenberg drew out that algebra's starkest physical
consequence: position and momentum cannot both be known to arbitrary
precision,

.. math::

   \Delta x \, \Delta p \ge \frac{\hbar}{2}.

This is not a statement about measurement clumsiness but a structural
property of any state -- a direct consequence of :math:`x` and :math:`p`
failing to commute, and the first hint that quantum mechanics would demand
a wholesale revision of what "knowing a system's state" even means.

*Implementation:* :func:`physicskit.quantum.utils.measure.uncertainty`
computes :math:`\Delta x`, :math:`\Delta p`, and their product directly
from a wavefunction, checking the Heisenberg bound;
:func:`physicskit.quantum.core.operators.commutator` together with
:func:`~physicskit.quantum.core.operators.position_operator` and
:func:`~physicskit.quantum.core.operators.momentum_operator` verify the
underlying canonical commutation relation :math:`[\hat x,\hat p]=i\hbar`
in a truncated oscillator basis.

*References:* W. Heisenberg, Z. Phys. 33, 879-893 (1925); M. Born and P.
Jordan, Z. Phys. 34, 858-888 (1925); M. Born, W. Heisenberg, and P.
Jordan, Z. Phys. 35, 557-615 (1926); the uncertainty principle itself, W.
Heisenberg, Z. Phys. 43, 172-198 (1927).

.. minigallery:: ../../examples/quantum/measurement/plot_heisenberg_uncertainty.py

1927 -- Ehrenfest's Theorem
----------------------------

Paul Ehrenfest asked how classical mechanics could ever emerge from a
theory built entirely on wavefunctions and operators, and found that it
already had: the expectation values of position and momentum obey
Newton's equations of motion exactly, for *any* potential, not merely a
harmonic one,

.. math::

   \frac{d\langle x\rangle}{dt} = \frac{\langle p\rangle}{m}, \qquad
   \frac{d\langle p\rangle}{dt} = -\left\langle\frac{dV}{dx}\right\rangle.

This is a precise structural fact about the Schrodinger equation, not an
approximation valid only for slowly varying potentials -- the subtlety is
that the second equation involves :math:`\langle dV/dx\rangle`, the
average of the force over the whole spread-out wavefunction, which
coincides with the classical force evaluated at the mean position,
:math:`-dV/dx(\langle x\rangle)`, only when :math:`V` is quadratic. For any
other potential a wavepacket's centroid drifts away from the corresponding
classical trajectory as it spreads, broadens, or splits -- the precise
sense in which quantum and classical dynamics agree exactly in expectation
value, yet disagree about everything a single classical trajectory would
predict.

*Implementation:* :class:`physicskit.quantum.core.solvers.SplitOperatorSolver1D`
propagates an arbitrary wavepacket through an arbitrary potential, and
:class:`physicskit.quantum.utils.measure.ExpectationMonitor` records
:math:`\langle x\rangle(t)` and :math:`\langle p\rangle(t)` from each
snapshot; comparing their numerical time derivatives against
:math:`\langle p\rangle/m` and :math:`-\langle dV/dx\rangle`, computed
independently via :func:`~physicskit.quantum.utils.measure.expectation_value`,
verifies both Ehrenfest identities directly from a real, propagated
wavefunction in a genuinely anharmonic (quartic) well; the same propagated
frames also feed :class:`~physicskit.quantum.visualizers.phase_space.WignerVisualizer`,
showing the state's phase-space distribution shear away from a rigid
rotation as the anharmonic potential acts, with the :math:`(\langle
x\rangle,\langle p\rangle)` trajectory traced on top of it.

*References:* P. Ehrenfest, Z. Phys. 45, 455-457 (1927).

.. minigallery:: ../../examples/quantum/wave_packets/plot_ehrenfest_theorem.py

1927 -- Dirac's Time-Dependent Perturbation Theory
--------------------------------------------------

Paul Dirac, developing the quantum theory of radiation, worked out how a
weak, time-dependent perturbation drives transitions between the unperturbed
stationary states of a system -- the foundation of time-dependent
perturbation theory. Two decades later Enrico Fermi gave the leading-order
transition-rate result a name that stuck: the "golden rule," so called in
his course-derived textbook *Nuclear Physics*, compiled from his University
of Chicago lectures and published in 1950 -- not, as sometimes stated,
unpublished lecture notes. The same machinery describes a system driven
periodically in time, where transitions become resonant multiphoton
processes whenever an integer number of drive quanta :math:`\hbar\omega`
bridges an energy gap.

*Implementation:* :class:`physicskit.quantum.chapters.perturbation.FloquetDrivenBox`
propagates a periodically driven infinite well with the FFT split-operator
method, and its
:meth:`~physicskit.quantum.chapters.perturbation.FloquetDrivenBox.transition_probability`
method exhibits exactly these resonant multiphoton transitions between box
eigenstates.

*References:* P. A. M. Dirac, Proc. R. Soc. A 114(767), 243-265 (1927);
"golden rule" naming, E. Fermi, Nuclear Physics (University of Chicago
Press, 1950), a textbook compiled from his lecture course.

.. minigallery:: ../../examples/quantum/perturbation/plot_floquet_driven_box.py

1928 -- Gamow, Gurney and Condon: Quantum Tunneling
---------------------------------------------------

George Gamow, and independently Ronald Gurney and Edward Condon, explained
why radioactive nuclei undergo alpha decay at all: classically, the alpha
particle sits in a potential well surrounded by a Coulomb barrier far
higher than its kinetic energy, and should never escape. Quantum
mechanically, the wavefunction does not vanish inside a classically
forbidden region -- it decays exponentially but remains nonzero -- giving a
small but nonzero probability of finding the particle on the far side of
the barrier. This single idea, quantum tunneling, explained the enormous
range of observed alpha-decay half-lives (many orders of magnitude, for
barriers differing only modestly in height) and remains the archetypal
example of a genuinely quantum phenomenon with no classical counterpart.

*Implementation:* :meth:`physicskit.quantum.chapters.potentials.FiniteSquareWell.scattering`
gives the exact transmission probability through a barrier, continuing
correctly into the sub-barrier tunneling regime via complex arithmetic;
:meth:`physicskit.quantum.chapters.potentials.DoubleWellSimulator.tunneling_oscillation`
and :meth:`~physicskit.quantum.chapters.potentials.DoubleWellSimulator.left_well_probability`
reproduce the related phenomenon of coherent tunneling oscillation between
the two wells of a symmetric double well. Beyond the stationary
transmission coefficient,
:meth:`~physicskit.quantum.chapters.potentials.FiniteSquareWell.wavepacket_scattering`
propagates an actual moving Gaussian wavepacket through the same barrier
(or, with a positive well depth instead, the related resonant-scattering
case) with the FFT split-operator method, splitting it into a reflected
and a transmitted piece in real time; and
:meth:`~physicskit.quantum.chapters.potentials.DoubleWellSimulator.tunneling_wavefunction`
gives the complex two-state wavefunction underlying the density-only
``tunneling_oscillation`` above. Both are rendered frame by frame with
:func:`~physicskit.quantum.visualizers.wavefunctions.animate_density`. See
the barrier-tunneling animation in
:doc:`/api/gallery/quantum/potentials/plot_barrier_tunneling`, the animated
well-scattering case in
:doc:`/api/gallery/quantum/potentials/plot_ramsauer_townsend_resonance`, and
the double-well oscillation, including its animated complex-valued version,
in :doc:`/api/gallery/quantum/potentials/plot_double_well_tunneling`.

*References:* G. Gamow, Z. Phys. 51, 204-212 (1928); R. Gurney and E.
Condon, Nature 122, 439 (1928), and Phys. Rev. 33, 127-140 (1929).

.. minigallery::
   ../../examples/quantum/potentials/plot_barrier_tunneling.py
   ../../examples/quantum/potentials/plot_ramsauer_townsend_resonance.py
   ../../examples/quantum/potentials/plot_double_well_tunneling.py

1928 -- The Van Vleck-Morette Semiclassical Propagator
--------------------------------------------------------

The same semiclassical logic behind WKB applies equally to time evolution
rather than stationary states: in 1928 John Van Vleck (extended into a full
path-integral formulation by Cecile DeWitt-Morette in 1951) showed that the
quantum propagator :math:`\langle x\rvert e^{-i\hat Ht/\hbar}\lvert
x_0\rangle` is dominated, to leading order in :math:`\hbar`, by a single
classical trajectory connecting :math:`x_0` to :math:`x` in time :math:`t`,

.. math::

   K(x,t;x_0,0) \approx \sqrt{\frac{1}{2\pi i\hbar}\left\lvert\frac{\partial p_0}{\partial x}\right\rvert}\; e^{iS(x,t;x_0,0)/\hbar - i\sigma\pi/2},

where :math:`S` is the classical action along that trajectory and
:math:`\sigma` (the Maslov index) counts the caustics -- focal points
where nearby trajectories cross -- it passes through. This single-trajectory
picture of propagation, two decades before Richard Feynman's 1948
path-integral formulation summed over *every* trajectory rather than just
the classical one, is the direct ancestor of every semiclassical
propagation method used in modern quantum chaos and chemical physics.

*Implementation:* :func:`physicskit.semiclassical.core.propagators.propagate_trajectory_monodromy_action`
integrates a classical trajectory together with its monodromy matrix and
action; :func:`~physicskit.semiclassical.core.propagators.van_vleck_prefactor`,
:func:`~physicskit.semiclassical.core.propagators.count_caustics`, and
:func:`~physicskit.semiclassical.core.propagators.van_vleck_propagator_1d`
assemble these into exactly the propagator amplitude above, Maslov phase
included.

*References:* J. Van Vleck, PNAS 14(2), 178-188 (1928); C. Morette, Phys.
Rev. 81, 848-852 (1951); R. P. Feynman, "Space-Time Approach to
Non-Relativistic Quantum Mechanics," Rev. Mod. Phys. 20, 367-387 (1948).

.. minigallery:: ../../examples/semiclassical/propagators/plot_semiclassical_propagators.py

1930 -- Dirac's Operator Method and the Harmonic Oscillator
-----------------------------------------------------------

In *The Principles of Quantum Mechanics* (1930), Dirac popularized an
algebraic shortcut for the quantum harmonic oscillator that sidesteps
solving the Schrodinger equation as a differential equation entirely:
factor the Hamiltonian into raising and lowering ("ladder") operators
:math:`\hat a^\dagger, \hat a` satisfying :math:`[\hat a,\hat a^\dagger]=1`,
build the entire spectrum by repeated raising from a ground state
annihilated by :math:`\hat a`, and read off :math:`E_n=\hbar\omega(n+\tfrac12)`
directly from the operator algebra. The technique, generalized to
Glauber's 1963 coherent states :math:`\lvert\alpha\rangle` -- eigenstates
of :math:`\hat a` that behave as classically as a quantum state can --
became the standard language of quantum optics.

*Implementation:* :func:`physicskit.quantum.core.operators.annihilation_operator`,
:func:`~physicskit.quantum.core.operators.creation_operator`, and
:func:`~physicskit.quantum.core.operators.number_operator` build exactly
this ladder-operator algebra in a truncated Fock basis;
:meth:`physicskit.quantum.chapters.harmonic_spin.HarmonicOscillator.coherent_wavefunction`
and :meth:`~physicskit.quantum.chapters.harmonic_spin.HarmonicOscillator.squeezed_vacuum_wavefunction`
build Glauber coherent states and their squeezed generalizations from this
same Fock-state expansion. The coherent/squeezed states are shown in
:doc:`/api/gallery/quantum/harmonic_oscillator/plot_harmonic_oscillator_suite`.

*References:* P. A. M. Dirac, The Principles of Quantum Mechanics
(Clarendon Press, 1930); coherent states, R. Glauber, Phys. Rev. 131,
2766-2788 (1963).

.. minigallery:: ../../examples/quantum/harmonic_oscillator/plot_ladder_operators.py

1935 -- The EPR Paradox
-----------------------

Einstein, Podolsky, and Rosen argued, in a paper meant to demonstrate
quantum mechanics' incompleteness, that if two particles are prepared in a
correlated ("entangled") state and then separated by an arbitrary distance,
measuring one instantaneously determines the corresponding outcome for the
other. Since no real influence should propagate faster than light, EPR
concluded that the particles must have possessed definite values all along,
carried by some "hidden variable" quantum mechanics simply fails to
describe -- entanglement, to them, was evidence the theory was unfinished
rather than a genuinely new feature of nature. It would take Bell
twenty-nine years to show that this intuition, however reasonable, is
testable -- and wrong.

*Implementation:* :class:`physicskit.quantum.chapters.entanglement.IsingEntangler`
builds exactly the kind of correlated two-particle state EPR had in mind,
but shows how it actually arises rather than simply positing it: two
qubits, each individually unbiased and initially unentangled
(:math:`\lvert+\rangle\otimes\lvert+\rangle`), accumulate genuine
entanglement purely through an Ising coupling
:math:`\hat H=\hbar J\,\sigma_z^{(1)}\otimes\sigma_z^{(2)}`, with its
:meth:`~physicskit.quantum.chapters.entanglement.IsingEntangler.concurrence`
and :meth:`~physicskit.quantum.chapters.entanglement.IsingEntangler.purity`
methods tracking the buildup from a product state (concurrence 0) to a
maximally entangled, Bell-equivalent state (concurrence 1) at
:math:`Jt=\pi/4`, animated with
:func:`~physicskit.quantum.visualizers.entanglement.animate_entanglement_growth`.
See it in
:doc:`/api/gallery/quantum/entanglement/plot_ising_entanglement_growth`.

*References:* A. Einstein, B. Podolsky, and N. Rosen, Phys. Rev. 47,
777-780 (1935).

.. minigallery:: ../../examples/quantum/entanglement/plot_ising_entanglement_growth.py

1935 -- Schrodinger Names "Entanglement"
-----------------------------------------

Responding directly to Einstein, Podolsky, and Rosen within months of their
paper, Erwin Schrodinger wrote a long analysis of what their correlated
state actually implies, and in it coined the word that has described the
phenomenon ever since: *Verschrankung*, "entanglement." Schrodinger argued
that entanglement -- not any single particle's uncertainty -- is the
truly characteristic trait of quantum mechanics, the feature that most
sharply separates it from any classical theory, and in the same paper
introduced his now-famous cat, whose life hangs on the entangled fate of a
radioactive atom, to dramatize how strange it is that this feature seems to
survive all the way up to macroscopic scale. The package's own
entanglement measures and Bell-state machinery are built on precisely the
concept Schrodinger named here, three decades before Bell showed it was
testable and half a century before it acquired experimental and
technological weight of its own.

*Connection:* :class:`physicskit.quantum.chapters.entanglement.IsingEntangler`
and :func:`physicskit.quantum.chapters.entanglement.bell_state`, together
with the concurrence and purity measures used throughout the entanglement
entries in this chronology, quantify exactly the non-classical
correlation -- entanglement -- that Schrodinger's paper first named and
argued was quantum mechanics' defining feature.

*References:* E. Schrodinger, "Die gegenwartige Situation in der
Quantenmechanik," Naturwissenschaften 23, 807-812, 823-828, 844-849
(1935); English translation, "The Present Situation in Quantum Mechanics,"
Proc. Am. Phil. Soc. 124, 323-338 (1980).

.. minigallery:: ../../examples/quantum/entanglement/plot_ising_entanglement_growth.py

1937-1946 -- Rabi's Magnetic Resonance Method and the Bloch Sphere
--------------------------------------------------------------------

In 1937, at Columbia University, Isidor Rabi devised the molecular-beam
magnetic-resonance method: a beam of molecules passes through a static
magnetic field and a region of oscillating field tuned near the natural
(Larmor) precession frequency of the nuclear magnetic moment, and that
moment's projection flips with near-unit probability exactly when the
drive is resonant -- the phenomenon now called a "Rabi oscillation." The
theoretical paper was completed in February and published in April 1937;
the first experimental resonance curve, obtained with a lithium chloride
beam, was submitted to *Physical Review* in January 1938. The method gave,
for the first time, a direct and extraordinarily precise measurement of
nuclear magnetic moments, and Rabi received the 1944 Nobel Prize in
Physics for it.

Nine years later, at Stanford, Felix Bloch (and independently, at Harvard,
Edward Purcell) extended the same resonance idea from a molecular beam in
vacuum to nuclear spins in bulk matter, detecting the induced radiofrequency
signal directly rather than a deflected beam -- "nuclear induction,"
published in 1946. In that same paper Bloch introduced the geometric device
that now carries his name: representing bulk nuclear magnetization as a
classical vector precessing (under a static field) or nutating (under a
resonant drive) on the surface of a unit sphere, exactly as a classical
magnetic moment would. Bloch's own sphere was a picture of a real classical
vector -- an ensemble average over many nuclear spins -- rather than the
quantum state of one two-level system; the now-standard generalization of
the same geometric picture to the pure quantum state of *any* single
two-level system (a single spin, or, in modern usage, a single qubit) is
usually credited to a 1957 paper by Richard Feynman, Frank Vernon Jr., and
Robert Hellwarth, which showed that any two-state Schrodinger evolution can
be mapped onto exactly Bloch's precession/nutation geometry. Bloch and
Purcell shared the 1952 Nobel Prize in Physics for the original discovery,
the direct ancestor of nuclear magnetic resonance (NMR) spectroscopy, MRI,
and -- via the Feynman-Vernon-Hellwarth generalization -- the standard way
of visualizing a single qubit's state in modern quantum computing.

.. math::

   U(t) = \cos\!\left(\frac{\Omega_R t}{2}\right) I
        - i\sin\!\left(\frac{\Omega_R t}{2}\right)
          \frac{\Delta\,\sigma_z + \Omega\,\sigma_x}{\Omega_R},
   \qquad \Omega_R = \sqrt{\Delta^2+\Omega^2},

the closed-form rotating-wave-approximation propagator for a two-level
system driven near resonance -- the "Rabi formula" -- with :math:`\Delta`
the drive detuning and :math:`\Omega` the drive strength.

*Implementation:* :class:`physicskit.quantum.chapters.spin.RabiProblem`
implements exactly this closed-form propagator, and its
:meth:`~physicskit.quantum.chapters.spin.RabiProblem.excited_state_population`
method reproduces the Rabi formula's population-flopping curve;
:func:`~physicskit.quantum.visualizers.bloch_sphere.state_to_bloch_trajectory`
and :func:`~physicskit.quantum.visualizers.bloch_sphere.animate_bloch_sphere`
trace the resulting state as a vector precessing/nutating on Bloch's own
sphere, animated frame by frame with a matplotlib ``FuncAnimation``. See the
Rabi-driven Bloch-sphere animation in
:doc:`/api/gallery/quantum/entanglement/plot_bloch_sphere_spin_dynamics`.

*References:* I. Rabi, Phys. Rev. 51, 652-654 (1937); I. Rabi et al., Phys.
Rev. 53, 318 (1938); F. Bloch, Phys. Rev. 70, 460-474 (1946); E. M.
Purcell, H. C. Torrey, and R. V. Pound, Phys. Rev. 69, 37-38 (1946);
generalization of the Bloch sphere to a single two-state quantum system, R.
P. Feynman, F. L. Vernon Jr., and R. W. Hellwarth, J. Appl. Phys. 28, 49-52
(1957).

.. minigallery:: ../../examples/quantum/entanglement/plot_bloch_sphere_spin_dynamics.py

1959 -- The Aharonov-Bohm Effect
--------------------------------

Yakir Aharonov and David Bohm predicted, in 1959, a startling consequence
of quantum mechanics with no classical analogue at all: a charged particle
can be measurably affected by an electromagnetic potential even in a
region where every field it could classically feel -- :math:`\mathbf B`
and :math:`\mathbf E` -- vanishes identically. Threading a magnetic flux
:math:`\Phi` through the hole of a conducting ring, without ever letting
the electron enter the field region, shifts every energy level of the
ring,

.. math::

   E_n(\Phi) = \frac{\hbar^2}{2mR^2}\left(n - \frac{\Phi}{\Phi_0}\right)^2, \qquad \Phi_0 = \frac{2\pi\hbar}{q},

purely through the vector potential's Aharonov-Bohm phase
:math:`\Delta\phi = 2\pi\Phi/\Phi_0` picked up by the electron's
wavefunction as it circles the ring. Confirmed experimentally by Akira
Tonomura's group using electron holography in 1986, the effect showed
that the electromagnetic potentials -- not just the fields derived from
them -- are physically real, and it seeded the entire field of mesoscopic
persistent-current physics. The same phase effect had in fact been derived
a decade earlier, in 1949, by Werner Ehrenberg and Raymond Siday, who
noted it explicitly in a paper on electron optics that went largely
unnoticed at the time; in recognition of that priority the effect is
sometimes called the "Ehrenberg-Siday-Aharonov-Bohm effect."

*Implementation:* :class:`physicskit.quantum.chapters.entanglement.AharonovBohmRing`
reproduces exactly this flux-dependent spectrum in its ``energy`` and
``spectrum`` methods, its ``aharonov_bohm_phase`` method gives
:math:`\Delta\phi`, and its ``persistent_current`` method computes the
equilibrium ring current the flux dependence implies.

*References:* Y. Aharonov and D. Bohm, Phys. Rev. 115, 485-491 (1959);
experimental confirmation, A. Tonomura et al., Phys. Rev. Lett. 56,
792-795 (1986); priority, W. Ehrenberg and R. E. Siday, Proc. Phys. Soc. B
62, 8-21 (1949).

.. minigallery:: ../../examples/quantum/entanglement/plot_aharonov_bohm_ring.py

1961 -- Jonsson's Electron Double-Slit Experiment
-------------------------------------------------

Richard Feynman would later call the electron double-slit experiment "a
phenomenon which is impossible ... to explain in any classical way, and
which has in it the heart of quantum mechanics." Claus Jonsson had already
performed it: using electron-biprism interferometry through up to five
parallel micro-slits machined in copper foil, he observed exactly the
interference fringes de Broglie's matter waves and Born's probabilistic
interpretation predicted, extending Young's 1801 two-slit demonstration
for light to individual massive particles for the first time. In 2002,
readers of *Physics World* voted the electron double-slit experiment "the
most beautiful experiment in physics."

.. math::

   I(y) = \left\lvert \psi_1(y) + \psi_2(y) \right\rvert^2,

the same two-amplitude interference law as classical wave optics, now
describing the probability density of finding a single electron -- fired
one at a time -- at position :math:`y` on the downstream screen.

*Implementation:* :class:`physicskit.quantum.chapters.wave_packets.TwinSlit`
builds this two-source interference pattern from coherent Gaussian slit
sources propagated to a screen; its ``amplitude`` and ``intensity``
methods reproduce the fringe pattern :math:`\lvert\psi_1+\psi_2\rvert^2`
directly. See it in :doc:`/api/gallery/quantum/wave_packets/plot_wave_packet_dynamics`.
Rather than that far-field (Fraunhofer) shortcut,
:func:`~physicskit.quantum.chapters.wave_packets.double_slit_potential` and
:func:`~physicskit.quantum.chapters.wave_packets.propagate_double_slit`
build the same two-gap wall as a genuine 2D potential and propagate a
Gaussian wavepacket through it with
:class:`~physicskit.quantum.core.solvers.SplitOperatorSolver2D`, so the
fringe pattern builds up frame by frame from the full time-dependent
Schrodinger equation rather than a closed-form formula, animated with
:func:`~physicskit.quantum.visualizers.wavefunctions.animate_density_2d`.
See it in
:doc:`/api/gallery/quantum/measurement/plot_double_slit_propagation`.

*References:* C. Jonsson, Z. Phys. 161, 454-474 (1961); English
translation, Am. J. Phys. 42, 4-11 (1974).

.. minigallery::
   ../../examples/quantum/wave_packets/plot_wave_packet_dynamics.py
   ../../examples/quantum/measurement/plot_double_slit_propagation.py

1964 -- Bell's Theorem and the CHSH Inequality
----------------------------------------------

John Bell showed that *any* theory of the EPR type -- one where measurement
outcomes are fixed in advance by local "hidden variables," unknown to
quantum mechanics but real nonetheless -- must satisfy an inequality that
quantum mechanics itself predicts will be violated. Clauser, Horne,
Shimony, and Holt reformulated Bell's inequality in 1969 into the
experimentally practical CHSH form,

.. math::

   S = E(a,b) - E(a,b') + E(a',b) + E(a',b'), \qquad \lvert S\rvert \le 2
   \ \text{(local hidden variables)},

which the singlet state violates up to the Tsirelson bound
:math:`\lvert S\rvert = 2\sqrt2 \approx 2.828`, established by Boris
Tsirelson in 1980 as quantum mechanics' own ceiling. Experiments -- from
Aspect's in the early 1980s to the loophole-free tests of 2015 -- have
consistently confirmed the quantum prediction, closing the question EPR
opened: entanglement is real, and no local hidden-variable theory can
reproduce it.

*Implementation:* :func:`physicskit.quantum.chapters.entanglement.bell_state`
builds the singlet used throughout;
:class:`~physicskit.quantum.chapters.entanglement.BellCorrelations` computes
the spin :meth:`~physicskit.quantum.chapters.entanglement.BellCorrelations.correlation`,
and its
:meth:`~physicskit.quantum.chapters.entanglement.BellCorrelations.chsh_S`
and :meth:`~physicskit.quantum.chapters.entanglement.BellCorrelations.chsh_optimal`
methods reproduce exactly this CHSH violation up to the Tsirelson bound.

*References:* J. Bell, Physics 1(3), 195-200 (1964); J. Clauser, M. Horne,
A. Shimony, and R. Holt, Phys. Rev. Lett. 23, 880-884 (1969); B. Tsirelson,
Lett. Math. Phys. 4, 93-100 (1980); A. Aspect et al., Phys. Rev. Lett. 49,
1804-1807 (1982); loophole-free tests, B. Hensen et al., Nature 526,
682-686 (2015).

.. minigallery:: ../../examples/quantum/entanglement/plot_entanglement_and_topology.py

1971 -- Gutzwiller's Trace Formula
----------------------------------

Martin Gutzwiller asked, in a series of papers culminating in 1971, how a
quantum energy spectrum could be reconstructed purely from the
*classical* mechanics of a system -- including chaotic systems, with no
exact quantum numbers at all. His answer, the trace formula, expresses the
quantum density of states as a sum over every classical periodic orbit,
weighted by that orbit's action, period, and (in higher dimensions)
stability:

.. math::

   g(E) \approx \bar g(E) + \frac{1}{\pi\hbar}\sum_{\text{p.o.}} \sum_{r=1}^{\infty} \frac{T_p}{\sqrt{\lvert\det(M_p^r - I)\rvert}} \cos\!\left(\frac{r S_p(E)}{\hbar} - r\,\sigma_p\frac{\pi}{2}\right),

where the outer sum runs over each primitive periodic orbit :math:`p`
(action :math:`S_p`, period :math:`T_p`, monodromy matrix :math:`M_p`,
Maslov index :math:`\sigma_p`) and the inner sum over its repetitions
:math:`r`. For an integrable one-dimensional system this identity is
exact rather than approximate, and it founded the field of quantum chaos:
the first rigorous bridge between a system's classical dynamics --
regular or chaotic -- and the fine structure of its quantum spectrum.

*Implementation:* :func:`physicskit.semiclassical.core.gutzwiller.classical_period`
supplies :math:`T(E)=dS/dE` for a 1D bound orbit, and
:func:`~physicskit.semiclassical.core.gutzwiller.gutzwiller_density_of_states`
sums the resulting exact 1D trace formula, reconstructing the
Bohr-Sommerfeld spectrum's delta-function peaks purely from repetitions
of a single classical orbit;
:func:`~physicskit.semiclassical.core.gutzwiller.gutzwiller_amplitude_from_monodromy`
gives the general stability amplitude :math:`1/\sqrt{\lvert
2-\operatorname{tr}M\rvert}` needed beyond the exactly-solvable 1D case.

*References:* M. Gutzwiller, J. Math. Phys. 12, 343-358 (1971).

.. minigallery:: ../../examples/semiclassical/gutzwiller/plot_gutzwiller_trace_formula.py

1982 -- The No-Cloning Theorem
-------------------------------

William Wootters and Wojciech Zurek, and independently Dennis Dieks,
proved a negative result as consequential as any of Bell's: no unitary
process can take an arbitrary, unknown quantum state and produce two
independent copies of it, :math:`\lvert\psi\rangle\lvert e\rangle \mapsto
\lvert\psi\rangle\lvert\psi\rangle`, for *every* :math:`\lvert\psi\rangle`
simultaneously. The proof is a direct consequence of the linearity that
makes quantum mechanics quantum mechanics: a fixed unitary that faithfully
clones two particular non-orthogonal states :math:`\lvert\psi\rangle` and
:math:`\lvert\phi\rangle` would, by linearity, act on their superposition
in a way that is *not* a faithful copy of that superposition -- a
contradiction unless the two states were orthogonal (or identical) to
begin with. The no-cloning theorem is the reason quantum cryptography can
detect eavesdropping (an eavesdropper cannot covertly copy a qubit to
inspect it later), and the reason quantum error correction has to encode
information redundantly across *entangled* qubits rather than by simply
duplicating a single qubit's state.

*Connection:* the package has no dedicated cloning-machine example, but
the two-qubit unitary machinery the theorem constrains is exactly the
machinery built elsewhere in this chronology:
:func:`physicskit.quantum.chapters.entanglement.bell_state` and
:class:`~physicskit.quantum.chapters.entanglement.IsingEntangler` both
construct genuinely entangled two-qubit states from unitary, linear time
evolution, the same linearity whose consequences the Wootters-Zurek-Dieks
proof exploits; no combination of a fixed unitary and an ancilla built
from these same primitives could instead be made to duplicate an arbitrary
input qubit.

*References:* W. K. Wootters and W. H. Zurek, Nature 299, 802-803 (1982);
D. Dieks, Phys. Lett. A 92, 271-272 (1982).

.. minigallery:: ../../examples/quantum/entanglement/plot_no_cloning_theorem.py

1984 -- Heller's Quantum Scars and the Herman-Kluk Propagator
---------------------------------------------------------------

Eric Heller discovered, in 1984, a phenomenon that contradicted the
prevailing expectation for quantum systems whose classical dynamics is
chaotic: rather than spreading uniformly over all of accessible phase
space (as random-matrix "quantum ergodicity" predicted they should), many
eigenstates of chaotic billiards show statistically significant density
enhancement concentrated along a handful of short, unstable periodic
orbits -- "scars." That same year, Michael Herman and Edward Kluk jointly
published, as co-authors of a single paper, a multi-trajectory
semiclassical propagator that made computing such wavepacket dynamics in
chaotic and anharmonic systems tractable, replacing the single (and often
singular, at a caustic) Van Vleck trajectory with a smooth sum over a
whole family of frozen Gaussian wavepackets,

.. math::

   \psi(x,t) \approx \int\!\frac{dq_0\,dp_0}{2\pi\hbar}\, C_t(q_0,p_0)\, e^{iS(t)/\hbar}\, \langle g_{q_0,p_0}\vert\psi_0\rangle\, g_{q_t,p_t}(x),

an integral over classical initial conditions that, unlike the single
Van Vleck trajectory, never itself passes through a caustic singularity.

*Implementation:* :func:`physicskit.semiclassical.systems.scarring.bouncing_ball_energies`
and :func:`~physicskit.semiclassical.systems.scarring.bouncing_ball_orbit_points`
locate the stadium billiard's shortest unstable periodic orbit family, and
:func:`~physicskit.semiclassical.systems.scarring.scar_enhancement`
quantifies exactly Heller's density-enhancement signature, :math:`\eta =
\langle\lvert\psi\rvert^2\rangle_{\text{tube}}/\langle\lvert\psi\rvert^2\rangle_{\text{billiard}}`;
:func:`~physicskit.semiclassical.core.propagators.herman_kluk_propagate_wavepacket`,
built on :func:`~physicskit.semiclassical.core.propagators.frozen_gaussian_1d`
and :func:`~physicskit.semiclassical.core.propagators.herman_kluk_prefactor`,
implements the Herman-Kluk multi-trajectory propagator above.

*References:* E. Heller, Phys. Rev. Lett. 53, 1515-1518 (1984); M. Herman
and E. Kluk, Chem. Phys. 91, 27-34 (1984) (a single joint paper, not two
independent discoveries).

.. minigallery::
   ../../examples/semiclassical/scarring/plot_quantum_scars.py
   ../../examples/semiclassical/propagators/plot_herman_kluk_wavepacket.py

1986-1990 -- Wave-Packet Revivals
----------------------------------

A wavepacket built from many eigenstates of an *anharmonic* potential --
where, unlike the harmonic oscillator, the energy levels are not evenly
spaced -- spreads and appears to lose all resemblance to its initial shape
within a few classical periods. J. Parker and C. R. Stroud predicted
theoretically in 1986, and J. A. Yeazell, M. Mallalieu, and C. R. Stroud
confirmed experimentally in Rydberg wave packets in 1990, what the
quadratic term in the level spacing actually implies: the packet is not
gone, only dephased, and it exactly reconstructs itself at a calculable
revival time, with smaller-scale "fractional revival" clones (named and
analyzed by Ilya Averbukh and Nathan Perelman in 1989) appearing at
rational fractions of that time along the way. For the textbook case of a
particle in an infinite square well, where :math:`E_n \propto n^2`
exactly,

.. math::

   t_\text{rev} = \frac{4mL^2}{\pi\hbar},

and a mirror-image replica of the initial packet appears already at
:math:`t_\text{rev}/2`.

*Implementation:* :class:`physicskit.quantum.chapters.wave_packets.QuantumRevival`
expands an arbitrary localized initial state (via ``gaussian_initial_state``
and ``eigenbasis_coefficients``) in the infinite-well eigenbasis and
evolves it with ``wavefunction``; its ``revival_time`` property gives
:math:`t_\text{rev}` above, and ``fidelity_to_initial`` tracks the
wavepacket's overlap with :math:`\psi(x,0)` collapsing and then sharply
recovering at that time. See it in
:doc:`/api/gallery/quantum/wave_packets/plot_wave_packet_dynamics`.

*References:* J. Parker and C. Stroud, Phys. Rev. Lett. 56, 716-719
(1986); J. A. Yeazell, M. Mallalieu, and C. R. Stroud Jr., Phys. Rev.
Lett. 64, 2007-2010 (1990); fractional revivals, I. Sh. Averbukh and N. F.
Perelman, Phys. Lett. A 139, 449-453 (1989).

.. minigallery:: ../../examples/quantum/wave_packets/plot_wave_packet_dynamics.py

See Also
--------

- :doc:`/api/quantum`
- :doc:`/api/gallery/quantum/index`
