:orphan:

Statistical Mechanics Deep Dives
====================================

Twenty-one worked examples, one per conceptual breakthrough in
:doc:`/history/statphys_breakthroughs`, each pairing a schematic diagram of
the underlying lattice, chain, or ensemble with a short, runnable
``physicskit.statphys`` script reproducing the phenomenon's textbook
observable signature. Every reference here has a verified counterpart in
the :doc:`/api/gallery/statphys/index` example gallery; this page exists to
put the *physics* of each breakthrough next to its diagram and its code, in
one place.

.. contents:: Contents
   :local:
   :depth: 1

1870s -- Boltzmann's Statistical Entropy and the H-Theorem
--------------------------------------------------------------

Boltzmann's :math:`S = k_B \ln W` and his H-theorem are two sides of one
claim: macroscopic irreversibility is nothing but the overwhelmingly
probable behavior of reversible microscopic collisions. The clearest way to
*see* the H-theorem is to watch a real N-body gas relax: start every
particle with the same speed (about as far from equilibrium as a fixed
total energy allows) and watch the speed distribution spread into the
Maxwell-Boltzmann bell curve while :math:`H = \int f \ln f\, dv` falls.

.. code-block:: text

   Periodic box, N Lennard-Jones particles, all |v| equal at t=0:

     +-------------------------------+
     |  o->    o->   o->     o->     |   <- every particle: same speed,
     |     o->    o->    o->         |      random direction (delta-fn
     |  o->   o->    o->    o->      |      speed distribution)
     |     o->   o->     o->         |
     +-------------------------------+
                    |  collisions randomize speeds
                    v
     +-------------------------------+
     |  o^  o<-  o>  ov   o\  o/     |   <- speeds now spread out;
     |    o>  o\   o^  o<-  o/       |      H(t) has decreased toward
     |  o<-  o/  ov   o>   o\        |      its Maxwell-Boltzmann minimum
     +-------------------------------+

:class:`physicskit.statphys.chapters.molecular_dynamics.LennardJonesGas`
integrates exactly this system with Velocity-Verlet dynamics, and
:meth:`~physicskit.statphys.chapters.molecular_dynamics.LennardJonesGas.h_function`
estimates :math:`H(t)` from the instantaneous speed histogram at every
recorded step.

.. code-block:: python

   import matplotlib.pyplot as plt
   from physicskit.statphys.chapters.molecular_dynamics import LennardJonesGas
   from physicskit.statphys.visualizers.particle_render import plot_velocity_histogram

   # Start every particle with the same speed -- far from Maxwell-Boltzmann.
   gas = LennardJonesGas(
       n_particles=225, box_size=25.0, initial_velocity_distribution="delta", seed=0
   )
   history = gas.run(n_steps=4000, steps_per_record=40)

   fig, axes = plt.subplots(1, 2, figsize=(10, 4))
   axes[0].plot(history["time"], history["H"])
   axes[0].set_xlabel("time")
   axes[0].set_ylabel(r"Boltzmann $H(t) = \int f \ln f\, dv$")
   axes[0].set_title("H-theorem: monotonic relaxation")

   plot_velocity_histogram(gas, ax=axes[1], show_theory=True)
   axes[1].set_title("Final speed distribution vs. Maxwell-Boltzmann")
   plt.tight_layout()
   plt.show()

1902 -- Gibbs's Ensemble Formalism
---------------------------------------

The canonical ensemble ties macroscopic response functions directly to
microscopic fluctuations. Sweeping a lattice model's temperature and
measuring the variance of its energy and magnetization at each point
reproduces the specific heat and susceptibility peaks that every later
lattice-model chapter in this package relies on.

.. code-block:: text

              ................
             :   Heat bath     :  T fixed, energy exchanged freely
             :   (reservoir)   :
              ''''''''|''''''''
                       | <- fluctuating energy E
                +------v------+
                |  L x L      |   P(configuration) proportional to
                |  spin       |        exp(-beta * E)
                |  lattice    |
                +-------------+
        Var(E) -> C_v ,  Var(M) -> chi   (fluctuation-dissipation)

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from physicskit.statphys.chapters.ising_lattice import Ising2D
   from physicskit.statphys.visualizers.lattice_render import plot_thermodynamics

   model = Ising2D(L=32, seed=0)
   temperatures = np.linspace(model.T_C + 1.2, max(model.T_C - 1.2, 0.5), 20)
   result = model.run_temperature_sweep(
       temperatures, n_equil=200, n_measure=300, algorithm="wolff"
   )

   plot_thermodynamics(result, T_c=model.T_C)
   plt.suptitle(r"Canonical-ensemble $C_v$ and $\chi$ from energy/magnetization variance")
   plt.show()

1905 -- Einstein's Explanation of Brownian Motion
-------------------------------------------------------

An ensemble of independent random walkers, each taking many small
independent kicks, has a mean squared displacement growing linearly in
time -- the diffusion law Einstein derived to explain Brownian motion and
that Perrin used to measure Avogadro's number.

.. code-block:: text

   Single walker trajectory (2D):        Ensemble MSD(t):
                                              MSD
        *  <- start                           |            .  '
         \                                     |         .'
          \___                                 |       .'
              \                                |     .'   slope = 2 d D
           ____/                               |   .'
          /                                     | .'
         *  <- t=1000                           +------------------ t

.. code-block:: python

   import matplotlib.pyplot as plt
   from physicskit.statphys.chapters.random_walk import RandomWalk
   from physicskit.statphys.visualizers.random_walk_render import plot_msd, plot_trajectories_2d

   walk = RandomWalk(n_walkers=500, n_steps=2000, dim=2, kind="gaussian", seed=0)
   D = walk.diffusion_coefficient()
   print(f"Estimated diffusion coefficient D = {D:.4f}")

   fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
   plot_trajectories_2d(walk, ax=axes[0], n_show=25)
   plot_msd(walk, ax=axes[1], show_theory=True)
   plt.tight_layout()
   plt.show()

1907 -- The Ehrenfest Urn Model
-------------------------------------

The Ehrenfest urn is exactly reversible and does recur, yet a system
started far from equilibrium (all :math:`N` balls in one box) relaxes to
near-equal occupancy almost immediately, while the recurrence time is
astronomically longer than the simulation could ever run.

.. code-block:: text

   t = 0 (maximally non-equilibrium)      t large (near equilibrium)
   +-----------+   +-----------+          +-----------+   +-----------+
   |o o o o o o|   |           |          |o o o  o o |   |o  o o  o o|
   |o o o o o o|   |           |   -->    |  o o  o   |   |o  o   o o |
   |o o o o o o|   |           |          |o    o o o |   | o o o    o|
   +-----------+   +-----------+          +-----------+   +-----------+
    n_left = N       n_right = 0           n_left ~ N/2      n_right ~ N/2
   one ball, chosen uniformly at random, crosses over at every time step

.. code-block:: python

   import matplotlib.pyplot as plt
   from physicskit.statphys.chapters.ehrenfest_urn import EhrenfestUrn
   from physicskit.statphys.visualizers.urn_render import plot_ehrenfest_history

   urn = EhrenfestUrn(n_balls=200, seed=0)  # all 200 balls start in the left box
   history = urn.run(n_steps=6000)

   plot_ehrenfest_history(history, n_balls=200)
   plt.suptitle("Reversible dynamics, irreversible-looking relaxation to n_left ~ N/2")
   plt.show()

1920-1944 -- The Ising Model and Onsager's Exact Solution
----------------------------------------------------------------

Onsager's exact solution locates the 2D Ising model's continuous phase
transition at :math:`T_C = 2J/(k_B \ln(1+\sqrt2)) \approx 2.269\, J/k_B`.
Sweeping temperature across :math:`T_C` and snapshotting the lattice below,
at, and above it makes the transition -- from an ordered, mostly-aligned
phase to a disordered one -- directly visible.

.. code-block:: text

   T << T_C (ordered)      T ~ T_C (critical)       T >> T_C (disordered)
   ^ ^ ^ ^ ^ ^ ^ ^          ^ v ^ ^ v v ^ ^           ^ v v ^ v ^ v v
   ^ ^ ^ ^ ^ ^ ^ ^          v ^ ^ v v ^ v ^           v ^ v v ^ v ^ ^
   ^ ^ ^ ^ ^ ^ ^ ^          ^ ^ v v v ^ ^ ^           v v ^ ^ v ^ v v
   ^ ^ ^ ^ ^ ^ ^ ^          v v ^ ^ ^ v v ^           ^ v ^ v v ^ ^ v
     bulk magnetized          scale-invariant                zero net
     (spontaneous              domains, all sizes             magnetization
      symmetry breaking)

.. code-block:: python

   import matplotlib.pyplot as plt
   from physicskit.statphys.chapters.ising_lattice import Ising2D
   from physicskit.statphys.visualizers.lattice_render import plot_spin_grid

   print(f"Onsager T_C = {Ising2D().T_C:.4f} J/kB")

   fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
   for ax, dT in zip(axes, [-1.0, 0.0, 1.0]):
       model = Ising2D(L=64, seed=1)
       T = model.T_C + dT
       model.sweep(beta=1.0 / T, algorithm="wolff", n_sweeps=200)
       plot_spin_grid(model.spins, ax=ax, title=f"T = {T:.2f}")
   plt.tight_layout()
   plt.show()

1937 -- Landau Mean-Field Theory
-----------------------------------

Expanding the free energy in powers of an order parameter :math:`m`, with
no microscopic input beyond symmetry, already reproduces spontaneous
symmetry breaking: one minimum above :math:`T_C`, a split double well
below it.

.. code-block:: text

   F(m)                              F(m)
    |    \      /                     |  \        /
    |     \    /      T > T_C         |   \  /\  /      T < T_C
    |      \  /        (m=0)          |    \/  \/      (m = +-m_eq)
    |       \/                        |    /\  /\
    +------------- m                  +------------- m
             0                            -m_eq  0  +m_eq

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from physicskit.statphys.utils.landau_theory import (
       landau_free_energy, landau_equilibrium_magnetization,
   )

   Tc = 2.0
   m = np.linspace(-1.5, 1.5, 300)
   fig, ax = plt.subplots(figsize=(6, 4.5))
   for T in [Tc + 0.8, Tc, Tc - 0.8]:
       ax.plot(m, landau_free_energy(m, T=T, Tc=Tc), label=f"T={T:.1f}")
   ax.set_xlabel("order parameter m"); ax.set_ylabel("F(m)"); ax.legend()
   plt.tight_layout()
   plt.show()

   print(f"m(T_C - 0.8) = {landau_equilibrium_magnetization(Tc - 0.8, Tc):.4f}  "
         f"(mean-field beta=1/2 power law)")

1952 -- The Yang-Lee Circle Theorem
--------------------------------------

Continuing a finite Ising system's partition function into the complex
fugacity plane, every zero of a ferromagnetic system lands exactly on the
unit circle -- and a phase transition, in the infinite-size limit, is
where these zeros pinch the positive real axis.

.. code-block:: text

          Im(z)
            |     x  x
            |   x        x       every zero of Z(z) lies
        ----x------+------x---- Re(z)   exactly on |z|=1
            |   x        x       (ferromagnetic J > 0)
            |     x  x

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from physicskit.statphys.utils.partition_function import (
       ising_partition_polynomial, yang_lee_zeros,
   )

   g = ising_partition_polynomial(N=18, beta=0.4, J=1.0, periodic=True)
   zeros = yang_lee_zeros(g)
   print(f"max deviation from |z|=1: {np.max(np.abs(np.abs(zeros) - 1)):.2e}")

   theta = np.linspace(0, 2 * np.pi, 200)
   plt.figure(figsize=(5, 5))
   plt.plot(np.cos(theta), np.sin(theta), "--", color="gray", linewidth=0.8)
   plt.scatter(zeros.real, zeros.imag, color="tab:red")
   plt.gca().set_aspect("equal")
   plt.xlabel("Re(z)"); plt.ylabel("Im(z)")
   plt.tight_layout()
   plt.show()

1924-1926 -- Bose-Einstein and Fermi-Dirac Quantum Statistics
---------------------------------------------------------------

Bosons pile without limit into a shared ground state below a critical
temperature; fermions, forbidden by the Pauli exclusion principle from
ever doubling up, fill states one at a time up to a sharp Fermi level as
:math:`T \to 0`.

.. code-block:: text

   Bosons, T < T_c:                  Fermions, T -> 0:
   energy                            energy
     |                                 |  (empty states)
     |          o                      |  ----------------
     |          o                      |  ---------------- <- Fermi level mu
     |          o                      |  ================ (filled, 1 per
     |     o    o   o                  |  ================  state, Pauli-
     |__ooooooooooooo__ ground state   |__================_ excluded above)
        macroscopic condensate            hard step at T=0

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from physicskit.statphys.utils.thermodynamics import (
       bose_einstein_occupation, fermi_dirac_occupation, bec_condensate_fraction,
   )

   energy = np.linspace(0.01, 5.0, 400)
   fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

   for T, style in zip([0.5, 1.0, 2.0], ["-", "--", ":"]):
       axes[0].plot(energy, fermi_dirac_occupation(energy, mu=0.0, temperature=T),
                    style, color="steelblue", label=f"Fermi-Dirac, T={T}")
       axes[0].plot(energy, bose_einstein_occupation(energy, mu=-0.01, temperature=T),
                    style, color="crimson", label=f"Bose-Einstein, T={T}")
   axes[0].set_ylim(0, 5)
   axes[0].set_xlabel(r"energy $\varepsilon$"); axes[0].legend(fontsize=8)

   T = np.linspace(0.0, 2.0, 200)
   axes[1].plot(T, bec_condensate_fraction(T, critical_temperature=1.0))
   axes[1].axvline(1.0, color="k", linestyle="--", linewidth=1, label=r"$T_c$")
   axes[1].set_xlabel(r"$T/T_c$"); axes[1].set_ylabel(r"condensate fraction $N_0/N$")
   axes[1].legend()
   plt.tight_layout()
   plt.show()

1952 -- The Potts Model
-----------------------------

The q-state Potts model rewards neighboring sites only for matching
*exactly*. On the square lattice, watching the order parameter as
temperature sweeps through :math:`T_C` shows the transition sharpen from a
smooth, continuous drop at small :math:`q` into an abrupt jump once
:math:`q > 4`.

.. code-block:: text

   q = 3 Potts configuration (letters = discrete states, not spin signs):

   A B B C A A          H = -J * sum_<i,j> delta(s_i, s_j)
   A A C C B B          (energy lowered only by *exact* neighbor matches,
   C A A B B C           not by a continuous alignment as in Ising/XY)
   B B A A C A
   C C A B B A

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from physicskit.statphys.chapters.ising_lattice import PottsModel2D

   fig, ax = plt.subplots(figsize=(6, 4.5))
   for q in [2, 4, 8]:
       model = PottsModel2D(L=24, q=q, seed=0)
       temperatures = np.linspace(model.T_C + 0.8, max(model.T_C - 0.8, 0.1), 15)
       result = model.run_temperature_sweep(temperatures, n_equil=150, n_measure=150)
       ax.plot(result["T"] / model.T_C, result["m"], marker="o", label=f"q={q}")
   ax.set_xlabel(r"$T / T_C(q)$"); ax.set_ylabel("Potts order parameter $m$")
   ax.set_title("Transition sharpens from continuous (q <= 4) to first-order (q > 4)")
   ax.legend()
   plt.tight_layout()
   plt.show()

1953 -- The Metropolis Algorithm
--------------------------------------

Every lattice chapter in this package is, underneath, the same
accept/reject rule: propose a local change, accept downhill moves always
and uphill moves with probability :math:`e^{-\beta \Delta E}`.

.. code-block:: text

   current configuration s
             |
             v
   propose s' (flip one randomly chosen spin)
             |
             v
   dE = E(s') - E(s)
        |                     |
     dE <= 0                dE > 0
        |                     |
        v                     v
   always accept       accept with prob. exp(-beta * dE)
        |                     |
        +----------+----------+
                   v
          repeat for every site (one sweep)

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from physicskit.statphys.chapters.ising_lattice import Ising2D

   model = Ising2D(L=32, seed=0)
   beta = 1.0 / model.T_C
   energies = [model.energy() / model.n_sites]
   for _ in range(200):
       model.sweep(beta=beta, algorithm="metropolis", n_sweeps=1)
       energies.append(model.energy() / model.n_sites)

   plt.figure(figsize=(6, 4))
   plt.plot(energies)
   plt.xlabel("Metropolis sweep"); plt.ylabel("energy per site")
   plt.title(r"Metropolis relaxation toward equilibrium at $T_C$")
   plt.tight_layout()
   plt.show()

1957 -- Molecular Dynamics Simulation
-------------------------------------------

Alder and Wainwright integrated Newton's equations directly for hundreds of
interacting particles rather than sampling configurations stochastically --
the direct-simulation complement to Metropolis Monte Carlo.

.. code-block:: text

   +-------------------------------+  <- periodic boundary: a particle
   |  o     o        o    o       |     leaving the right edge re-enters
   |     o      o  o    o    o    |     on the left (and similarly top/
   |  o     o    o     o     o    |     bottom)
   |     o    o     o     o     o |
   |  o     o    o    o      o    |     Velocity Verlet: x(t+dt), v(t+dt)
   +-------------------------------+     from Lennard-Jones pairwise forces

.. code-block:: python

   import matplotlib.pyplot as plt
   from physicskit.statphys.chapters.molecular_dynamics import LennardJonesGas
   from physicskit.statphys.visualizers.particle_render import plot_particle_snapshot

   gas = LennardJonesGas(n_particles=144, box_size=20.0, seed=0)
   E0 = gas.total_energy()
   gas.step(n_steps=2000)
   E1 = gas.total_energy()
   print(f"Energy drift over 2000 Verlet steps: {abs(E1 - E0) / abs(E0):.2e} (relative)")

   fig, ax = plt.subplots(figsize=(5, 5))
   plot_particle_snapshot(gas, ax=ax, color_by_speed=True)
   plt.tight_layout()
   plt.show()

1957 -- Percolation Theory
---------------------------------

Below the percolation threshold :math:`p_c`, only finite clusters exist;
above it, an infinite spanning cluster appears with probability one -- a
purely geometric phase transition with a sharp threshold and a power-law
cluster-size distribution right at :math:`p_c`.

.. code-block:: text

   p < p_c (subcritical)        p = p_c (critical)         p > p_c (spanning)
   . X X . X . X .              X X . X X . X X             X X X X X X X X
   X . X . . X . X              X . X X . X . X             X X X X X X X X
   . X . X X . X .              . X X . X X X .             X X X X X X X X
   X . X . X . X .              X . X X . X X X             X X X X X X X X
     (small clusters only)        (spans, fractal)          (fully occupied)

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from physicskit.statphys.chapters.percolation import Percolation2D
   from physicskit.statphys.visualizers.lattice_render import (
       plot_percolation_clusters, plot_cluster_size_distribution,
   )

   perc = Percolation2D(L=80, seed=0)
   p_values = np.linspace(0.3, 0.85, 25)
   p, P_span = perc.spanning_probability(p_values, n_trials=60)

   fig, axes = plt.subplots(1, 3, figsize=(13, 4))
   axes[0].plot(p, P_span, marker="o")
   axes[0].axvline(perc.p_c, color="k", linestyle="--", label=r"$p_c$")
   axes[0].set_xlabel("p"); axes[0].set_ylabel(r"$P_{\rm span}(p)$"); axes[0].legend()

   perc.generate(perc.p_c)
   plot_percolation_clusters(perc, ax=axes[1])
   plot_cluster_size_distribution(perc.cluster_size_distribution(), ax=axes[2])
   plt.tight_layout()
   plt.show()

1966-1971 -- Kadanoff / Wilson Renormalization Group
------------------------------------------------------------

Coarse-graining a critical (:math:`T = T_C`) configuration by repeatedly
replacing :math:`2 \times 2` blocks with a single majority-rule spin
produces a self-similar sequence of lattices; away from :math:`T_C`, the
same procedure flows visibly toward a fully ordered or fully disordered
fixed point.

.. code-block:: text

   Fine lattice L                 2x2 majority rule            Coarse L/2
   + + - -                                                       +   -
   + + - -        each 2x2 block -> sign(sum of block)   ->
   - - + +                                                       -   +
   - - + +

.. code-block:: python

   import matplotlib.pyplot as plt
   from physicskit.statphys.chapters.renormalization import BlockSpinRG
   from physicskit.statphys.visualizers.rg_render import plot_rg_flow

   fig, axes = plt.subplots(1, 3, figsize=(12, 4))
   for ax, (label, T) in zip(axes, [("T << T_C", 1.0), ("T = T_C", None), ("T >> T_C", 6.0)]):
       rg = BlockSpinRG(L=64, T=T, seed=0)
       grids = rg.iterate()
       order_params = [BlockSpinRG.order_parameter(g) for g in grids]
       ax.plot(order_params, marker="o")
       ax.set_title(label); ax.set_xlabel("coarse-graining step")
       ax.set_ylabel("|<s>|"); ax.set_ylim(-0.05, 1.05)
   plt.tight_layout()
   plt.show()

1973 -- Vortex Unbinding in the 2D XY Model
---------------------------------------------------

Below the Kosterlitz-Thouless temperature, vortices remain bound in tight
pairs of opposite topological charge; above it, thermal fluctuations tear
these pairs into a free "vortex plasma" -- a phase transition organized by
topology rather than a local order parameter.

.. code-block:: text

   Bound vortex-antivortex pair (T < T_KT):

        ^  ->                    charge +1 (vortex): spins wind
       \    \                    counterclockwise once around the core
        (+1)  ->
       /    \                    charge -1 (antivortex): spins wind
      v   <-  (-1)  <-           clockwise once around the core
             /   \
           <-     v              tightly bound below T_KT; unbind into a
                                  free plasma above T_KT

.. code-block:: python

   import matplotlib.pyplot as plt
   from physicskit.statphys.chapters.ising_lattice import XYModel2D
   from physicskit.statphys.visualizers.vortex_render import plot_vortices

   fig, axes = plt.subplots(1, 2, figsize=(10, 5))
   for ax, dT in zip(axes, [-0.3, 0.5]):
       model = XYModel2D(L=48, seed=0)
       T = model.T_KT + dT
       model.sweep(beta=1.0 / T, n_sweeps=300)
       nv, na = model.vortex_count()
       plot_vortices(model.theta, ax=ax)
       ax.set_title(f"T = {T:.2f} ({'below' if dT < 0 else 'above'} $T_{{KT}}$): "
                    f"{nv} vortices, {na} antivortices")
   plt.tight_layout()
   plt.show()

1975 -- The Edwards-Anderson Spin Glass
-----------------------------------------------

A generic plaquette with an odd number of antiferromagnetic bonds cannot
satisfy all four of its bonds at once -- it is frustrated. The
Edwards-Anderson order parameter, the overlap between two replicas sharing
the same disorder, stays nonzero as long as both replicas freeze into the
same disorder-selected (if disordered) pattern.

.. code-block:: text

   o --+J-- o --+J-- o        Plaquette bonds (clockwise): +J, -J, +J, -J
   |        |        |        product of four bonds < 0  ==>  FRUSTRATED:
  -J       +J       -J        no single spin assignment on its four sites
   |        |        |        satisfies all four bonds simultaneously
   o --+J-- o --+J-- o

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from physicskit.statphys.chapters.spin_glass import EdwardsAndersonSpinGlass2D
   from physicskit.statphys.visualizers.lattice_render import plot_spin_grid

   model = EdwardsAndersonSpinGlass2D(L=32, seed=0)
   print(f"Frustrated plaquette fraction: {model.frustration_density():.3f} (expect ~0.5)")

   temperatures = np.linspace(0.3, 3.0, 10)
   q2_values = [
       model.edwards_anderson_order_parameter(beta=1.0 / T, n_equil=150, n_measure=150)
       for T in temperatures
   ]

   fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
   model.sweep(beta=2.0, n_sweeps=500)
   plot_spin_grid(model.spins, ax=axes[0], title="Frozen configuration, T=0.5 J/kB")
   axes[1].plot(temperatures, q2_values, marker="o")
   axes[1].set_xlabel("Temperature"); axes[1].set_ylabel(r"$\langle q^2 \rangle$")
   plt.tight_layout()
   plt.show()

1975-1980 -- The Sherrington-Kirkpatrick Model and Parisi's Replica Symmetry Breaking
------------------------------------------------------------------------------------------

Every spin coupled to every other spin by an independent Gaussian bond.
Parisi's replica-symmetry-breaking solution predicts the pooled
replica-overlap distribution P(q) is a single narrow peak above the
transition, but broad and structured below it.

.. code-block:: text

   P(q)                              P(q)
    |     ___                         |    __
    |    /   \      T > T_SG          |  _/  \_       T < T_SG
    |   /     \      (paramagnetic)   | /      \      (many pure states,
    +--+---+---+-- q                  +---------+-- q  broad, non-Gaussian)
      -1   0   1                       -1   0   1

.. code-block:: python

   import matplotlib.pyplot as plt
   from physicskit.statphys.chapters.spin_glass import SherringtonKirkpatrick
   from physicskit.statphys.visualizers.spin_glass_render import plot_overlap_distribution

   fig, ax = plt.subplots(figsize=(6, 4))
   for beta, label in [(0.3, "T=3.3 J/kB (paramagnetic)"), (2.0, "T=0.5 J/kB (spin-glass)")]:
       model = SherringtonKirkpatrick(N=64, seed=0)
       samples = model.overlap_distribution(beta=beta, n_disorder=25, n_equil=300, n_measure=60)
       plot_overlap_distribution(samples, ax=ax, label=label)
   plt.tight_layout()
   plt.show()

1981 -- The Binder Cumulant and Finite-Size Scaling
--------------------------------------------------------

The fourth-order cumulant :math:`U_4 = 1 - \langle M^4\rangle / (3\langle
M^2\rangle^2)` is dimensionless, so at :math:`T_C` its curves for different
lattice sizes :math:`L` all cross at (very nearly) the same temperature --
letting :math:`T_C` be pinned down without ever simulating an infinite
system.

.. code-block:: text

   U4
    |     L=8  \             / L=8
    |            \    X    /              X marks the common crossing:
    |              \  |  /                the finite-size-independent
    |               \ | /                 estimate of T_C
    |     L=16 /      |      \ L=16
    |         /       |       \
    +------------------------------------- T
                       T_C

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from physicskit.statphys.chapters.ising_lattice import Ising2D
   from physicskit.statphys.utils.thermodynamics import binder_cumulant
   from physicskit.statphys.utils.finite_size_scaling import binder_cumulant_crossing

   temperatures = np.linspace(1.8, 2.8, 12)
   U4_by_L = {}
   for L in [8, 16, 24]:
       model = Ising2D(L=L, seed=0)
       U4 = []
       for T in temperatures:
           beta = 1.0 / T
           model.sweep(beta=beta, algorithm="wolff", n_sweeps=100)  # equilibrate
           mags = []
           for _ in range(200):
               model.sweep(beta=beta, algorithm="wolff", n_sweeps=1)
               mags.append(model.magnetization())
           U4.append(binder_cumulant(np.array(mags)))
       U4_by_L[L] = np.array(U4)

   T_c_estimate = binder_cumulant_crossing(temperatures, U4_by_L)
   print(f"Binder-cumulant crossing estimate of T_C: {T_c_estimate:.3f} "
         f"(Onsager: {Ising2D().T_C:.3f})")

   plt.figure(figsize=(6, 4.5))
   for L, U4 in U4_by_L.items():
       plt.plot(temperatures, U4, marker="o", label=f"L={L}")
   plt.axvline(T_c_estimate, color="k", linestyle="--", label=r"$T_C$ estimate")
   plt.xlabel("T"); plt.ylabel(r"$U_4$"); plt.legend()
   plt.tight_layout()
   plt.show()

1986 -- The Kardar-Parisi-Zhang Equation
--------------------------------------------

A local growth rule with no adjustable surface-tension or coupling
constant at all -- the Restricted Solid-On-Solid model -- is enough to
place a growing interface in the KPZ universality class: early-time
width growth :math:`w \sim t^{1/3}` crossing over to a size-limited
saturation :math:`w_{\text{sat}} \sim L^{1/2}`.

.. code-block:: text

   height                                 log w
     |        .  .                          |            _________ w_sat ~ L^0.5
     |     . .  ..  .   rough,               |         _/
     |   ..        . .  uncorrelated         |       _/    w ~ t^(1/3)
     |  .            .. noise + local        |     _/
     +------------------- x                  +---------------------- log t
        growing interface                       growth then saturation

.. code-block:: python

   import matplotlib.pyplot as plt
   from physicskit.statphys.chapters.kpz_growth import KPZInterface
   from physicskit.statphys.utils.finite_size_scaling import power_law_exponent
   from physicskit.statphys.visualizers.kpz_render import plot_width_growth

   interface = KPZInterface(L=400, seed=0)
   times, widths = interface.run_growth_curve(t_max=1500, n_points=30)
   beta_fit, _ = power_law_exponent(times, widths)
   print(f"fitted growth exponent beta = {beta_fit:.3f} (KPZ theory: 1/3)")

   plot_width_growth(times, widths, beta=1.0 / 3.0)
   plt.tight_layout()
   plt.show()

1987 -- Self-Organized Criticality and the BTW Sandpile
---------------------------------------------------------------

Grains added one at a time tune the pile *itself* to a critical state, with
no parameter fine-tuning: avalanche sizes follow a power law with no
characteristic scale, the geometric signature of self-organized
criticality.

.. code-block:: text

   1 2 1 3 2
   2 4*2 1 3    site (1,1) reaches height 4 = threshold -> topples,
   1 3 1 2 1    sending one grain to each of its four neighbors,
   3 2 4 1 2    which may push a neighbor over threshold too --
   2 1 2 3 1    an avalanche, size = total topplings triggered

.. code-block:: python

   import matplotlib.pyplot as plt
   from physicskit.statphys.chapters.sandpile import BTWSandpile
   from physicskit.statphys.visualizers.sandpile_render import (
       plot_sandpile_heights, plot_avalanche_size_distribution,
   )

   pile = BTWSandpile(L=64, seed=0)
   sizes = pile.run(n_grains=20000, warmup=20000)  # warmup reaches the SOC state

   fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
   plot_sandpile_heights(pile, ax=axes[0])
   plot_avalanche_size_distribution(sizes, ax=axes[1])
   plt.tight_layout()
   plt.show()

1989 -- The Wolff Cluster Algorithm
-------------------------------------------

A Wolff update grows one cluster of same-sign spins from a random seed --
each bond added independently with probability :math:`1 - e^{-2\beta J}` --
and flips the whole cluster in a single global Monte Carlo move, sidestepping
the critical slowing down that limits single-spin Metropolis dynamics right
at :math:`T_C`.

.. code-block:: text

   + + + + +          + [+ + +]+       cluster grown from a random seed
   + + + + +    -->   +[+ + + +]       (bracketed): each same-sign neighbor
   + - + + +          +[+]- + +        bond joins with prob. 1-exp(-2*beta*J)
   + + + - +          + + + - +        then the entire cluster is flipped
                                        as ONE Monte Carlo move

.. code-block:: python

   import matplotlib.pyplot as plt
   from physicskit.statphys.chapters.ising_lattice import Ising2D

   model = Ising2D(L=64, seed=0)
   beta = 1.0 / model.T_C  # right at T_C: hardest regime for single-spin dynamics

   cluster_sizes = [model.sweep(beta=beta, algorithm="wolff", n_sweeps=1) for _ in range(300)]

   plt.figure(figsize=(6, 4))
   plt.plot(cluster_sizes)
   plt.axhline(model.n_sites, color="k", linestyle="--", linewidth=1, label="full lattice")
   plt.xlabel("Wolff update"); plt.ylabel("cluster size")
   plt.title(r"Wolff cluster sizes at $T_C$: a global, not local, Monte Carlo move")
   plt.legend()
   plt.tight_layout()
   plt.show()

1997 -- The Jarzynski Equality
----------------------------------

Averaging :math:`e^{-\beta W}`, not :math:`W`, over repeated realizations
of a nonequilibrium protocol recovers the exact equilibrium free energy
difference -- however fast or dissipative each individual realization is.

.. code-block:: text

   density
     |        <W>                        <W> sits above the true Delta F
     |         v                         (second law), but the exponential
     |    _   _|_                        average of the SAME work samples
     |   / \_/   \___   Delta F          lands exactly back on Delta F
     +------|----------- W                (Jensen's inequality, saturated)
          Delta F

.. code-block:: python

   import matplotlib.pyplot as plt
   from physicskit.statphys.chapters.nonequilibrium_work import JarzynskiHarmonicTrap
   from physicskit.statphys.visualizers.jarzynski_render import plot_work_distribution

   model = JarzynskiHarmonicTrap(seed=0)
   work = model.run_protocol(lambda_0=0.0, lambda_1=2.0, tau=0.3, n_steps=400, n_trajectories=4000)
   dF_estimate = model.jarzynski_free_energy_estimate(work)
   print(f"<W> = {work.mean():.3f}   Jarzynski dF_estimate = {dF_estimate:.3f}   (true dF = 0)")

   plot_work_distribution(work, dF_true=0.0, dF_estimate=dF_estimate)
   plt.tight_layout()
   plt.show()

See Also
--------

- :doc:`/history/statphys_breakthroughs` for the full chronology and the
  historical context behind each entry above.
- :doc:`/api/gallery/statphys/index` for the complete, gallery-rendered
  example suite these snippets are drawn from.
- :doc:`/api/statphys`
