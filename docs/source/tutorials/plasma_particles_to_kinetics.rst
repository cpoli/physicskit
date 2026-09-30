Plasma Physics from One Particle to Kinetic Theory
==================================================

A plasma can be described at three levels: as individual charged
particles gyrating in fields, as a conducting fluid (magnetohydrodynamics),
or as a distribution function in phase space (kinetic theory). This
tutorial works through all three with :mod:`physicskit.plasma`. It
checks each numerical result against the formula it should reproduce,
and ends with Landau damping, a collisionless effect that only the
kinetic description captures.

.. note:: **Units.** The single-particle and fluid functions take SI
   inputs and return SI outputs. :data:`~physicskit.plasma.single_particle.QE`,
   :data:`~physicskit.plasma.single_particle.MP` and :data:`~physicskit.plasma.single_particle.ME` are
   the elementary charge and the proton and electron masses in SI. The
   particle-in-cell (PIC) code instead uses the standard normalized
   units of plasma simulation: time in :math:`1/\omega_{pe}`, length in
   Debye lengths :math:`\lambda_D`, and velocity in thermal speeds
   :math:`v_{th} = \omega_{pe}\lambda_D`. Keep the two conventions apart.

Gyration: the cyclotron frequency and Larmor radius
---------------------------------------------------

A charge in a uniform magnetic field circles it at the cyclotron
frequency :math:`\omega_c = qB/m`, on a circle of Larmor radius
:math:`r_L = m v_\perp/(|q|B)`. The Boris pusher
(:func:`~physicskit.plasma.boris_integrate`; Boris, 1970), the standard
integrator in PIC codes, splits each step into half electric kicks and a
pure rotation. The magnetic force does no work, and a pure rotation
can't change the particle's speed either, so the kinetic energy is
conserved to rounding error:

.. code-block:: python

   import numpy as np
   from physicskit import plasma

   B0, v_perp = 1.0, 1.0e5  # 1 T, 100 km/s proton
   omega_c = plasma.cyclotron_frequency(plasma.QE, plasma.MP, B0)
   r_L = plasma.larmor_radius(v_perp, plasma.QE, plasma.MP, B0)
   print(f"omega_c = {omega_c:.4e} rad/s, period = {2 * np.pi / omega_c * 1e9:.2f} ns, r_L = {r_L * 1e3:.4f} mm")
   # omega_c = 9.5788e+07 rad/s, period = 65.59 ns, r_L = 1.0440 mm

   pos, vel = plasma.boris_integrate(
       np.zeros(3), np.array([v_perp, 0.0, 0.0]), plasma.QE, plasma.MP,
       E=np.zeros(3), B=np.array([0.0, 0.0, B0]), dt=1e-10, steps=2000,
   )
   print(f"orbit radius = {np.ptp(pos[:, 1]) / 2 * 1e3:.4f} mm, "
         f"max |speed error| = {np.abs(np.linalg.norm(vel, axis=1) - v_perp).max():.1e} m/s")
   # orbit radius = 1.0440 mm, max |speed error| = 1.5e-10 m/s

The simulated orbit has exactly the Larmor radius. Over three gyrations
the speed changes by :math:`10^{-10}` m/s out of :math:`10^5`, which is
floating-point rounding.

Drifts: E×B is the same for every species
-----------------------------------------

Add a perpendicular electric field and the gyration centre drifts with

.. math::

   \mathbf v_E = \frac{\mathbf E\times\mathbf B}{B^2},

which contains neither the charge nor the mass. Electrons and ions drift
together, so an :math:`\mathbf E\times\mathbf B` drift carries the whole
plasma without driving a current. To measure the drift cleanly, sample
the position after a whole number of gyro-periods, where the gyration
has returned to its starting phase:

.. code-block:: python

   E, B = np.array([0.0, 1.0e3, 0.0]), np.array([0.0, 0.0, B0])
   print(plasma.exb_drift(E, B))
   # [1000.    0.    0.]

   for name, q, m in [("proton", plasma.QE, plasma.MP), ("electron", -plasma.QE, plasma.ME)]:
       period = 2 * np.pi / abs(plasma.cyclotron_frequency(q, m, B0))
       dt = period / 1000
       pos, _ = plasma.boris_integrate(np.zeros(3), np.array([v_perp, 0.0, 0.0]), q, m, E, B, dt=dt, steps=20 * 1000)
       print(f"{name:<8} drift over 20 gyro-periods: {pos[-1, 0] / (20 * period):.1f} m/s")

   # proton   drift over 20 gyro-periods: 999.7 m/s
   # electron drift over 20 gyro-periods: 999.7 m/s

The proton gyrates 1836 times more slowly than the electron and in the
opposite direction, yet both guiding centres move at :math:`E/B = 1000`
m/s in :math:`+x`. A drift that *does* depend on charge,
:func:`~physicskit.plasma.grad_b_drift`, separates the species and
drives the ring current in Earth's magnetosphere.

Collective scales: plasma frequency and Alfvén speed
----------------------------------------------------

Collective behaviour starts once there are many particles. Displace the
electrons and they oscillate about the ions at the plasma frequency
:math:`\omega_{pe} = \sqrt{n e^2/(\varepsilon_0 m_e)}`, the natural
frequency of a plasma. In the fluid (MHD) picture, magnetic field lines
behave like strings under tension :math:`B^2/\mu_0` loaded with the
plasma's mass density, and waves travel along them at the Alfvén speed
:math:`v_A = B/\sqrt{\mu_0\rho}` (Alfvén, *Nature* 150, 405, 1942):

.. code-block:: python

   omega_pe = plasma.plasma_frequency(n=1.0e12)       # ionospheric F-layer, m^-3
   print(f"f_pe = {omega_pe / (2 * np.pi) / 1e6:.3f} MHz")
   # f_pe = 8.979 MHz

   v_A = plasma.alfven_speed(B=5.0e-9, rho=5.0e6 * plasma.MP)   # solar wind at 1 AU
   print(f"v_A = {v_A / 1e3:.1f} km/s")
   # v_A = 48.8 km/s

The first number is the familiar rule of thumb
:math:`f_{pe} \approx 8.98\sqrt{n}` Hz. It explains why the ionosphere
reflects shortwave radio below about 9 MHz and is transparent above.
The second is the typical Alfvén speed in the solar wind, around ten
times slower than the wind itself, so the solar wind is super-Alfvénic.
:func:`~physicskit.plasma.cold_plasma_dispersion` and
:func:`~physicskit.plasma.stix_parameters` extend these two scales to
the full set of cold-plasma waves (Stix, *Waves in Plasmas*, 1992).

Landau damping: collisionless decay in a particle simulation
------------------------------------------------------------

The fluid picture predicts that a Langmuir wave (an electron density
oscillation) with wavenumber :math:`k` oscillates forever at the
Bohm-Gross frequency :math:`\omega^2 = \omega_{pe}^2 + 3k^2v_{th}^2`.
Landau (1946) showed that in a collisionless plasma the wave
nevertheless *decays*. Electrons moving at nearly the wave's phase
velocity exchange energy with it, and a Maxwellian has slightly more
slower electrons, which gain energy, than faster ones, which lose it. In
the weak-damping limit the rate is

.. math::

   \gamma = -\omega_{pe}\sqrt{\frac{\pi}{8}}\frac{1}{(k\lambda_D)^3}
   \exp\left(-\frac{1}{2(k\lambda_D)^2} - \frac32\right).

A 1D electrostatic PIC simulation (:func:`~physicskit.plasma.pic_simulate`)
contains no damping term at all. It only pushes particles through the
self-consistent field, yet it reproduces the decay. Seed a small density
perturbation with :func:`~physicskit.plasma.landau_damping_ic` (a
"quiet start" that avoids random sampling noise in the seed) and follow
the field energy, which oscillates at :math:`2\omega` and decays at
:math:`2\gamma`:

.. code-block:: python

   from scipy.signal import find_peaks

   k, v_th = 0.5, 1.0                      # k * lambda_D = 0.5, normalized units
   L = 2 * np.pi / k                       # one wavelength, periodic
   print(f"gamma (Landau)  = {plasma.landau_damping_rate(k, v_th):.4f}")
   # gamma (Landau)  = -0.1514

   x0, v0 = plasma.landau_damping_ic(100000, L=L, k_mode=k, alpha=0.05, v_th=v_th, seed=1)
   run = plasma.pic_simulate(x0, v0, L=L, ng=64, dt=0.05, steps=500)
   t, W = run["t"], run["field_energy"]

   peaks, _ = find_peaks(W)
   t_peaks, W_peaks = t[peaks][:6], W[peaks][:6]
   print("peak times", t_peaks.round(2))
   # peak times [ 2.4   4.65  6.9   9.   11.3  12.6 ]

   gamma_fit = 0.5 * np.polyfit(t_peaks, np.log(W_peaks), 1)[0]
   omega_fit = np.pi / np.mean(np.diff(t_peaks[:5]))
   print(f"gamma (PIC fit) = {gamma_fit:.4f}, omega (PIC fit) = {omega_fit:.3f}, "
         f"Bohm-Gross = {np.sqrt(1 + 3 * k**2):.3f}")
   # gamma (PIC fit) = -0.1377, omega (PIC fit) = 1.412, Bohm-Gross = 1.323

Two things come out of the simulation:

- **The damping rate** is within about 10% of Landau's formula, from
  100 000 particles and nothing but Newton's and Gauss's laws. The
  formula is itself the weak-damping approximation; the exact root of
  the kinetic dispersion relation at :math:`k\lambda_D = 0.5` is
  :math:`\gamma = -0.1533`. The remaining spread comes from particle
  noise, which scales as :math:`1/\sqrt{N}`. Different ``seed`` values
  give fits between about :math:`-0.14` and :math:`-0.17`, and after
  roughly six peaks the field energy reaches the noise floor, which is
  why the fit stops there.
- **The frequency** is 1.41, not the Bohm-Gross fluid value 1.32. It
  agrees with the real part of the exact kinetic root,
  :math:`\omega = 1.4156\,\omega_{pe}`. At :math:`k\lambda_D = 0.5`
  the fluid approximation is no longer accurate, and the particle
  simulation, which makes no fluid assumption, shows the kinetic answer.

Where to go next
----------------

- :func:`~physicskit.plasma.two_stream_ic` with the same
  :func:`~physicskit.plasma.pic_simulate` gives the opposite case: an
  *unstable*, growing wave, fed by two counter-streaming beams.
- :func:`~physicskit.plasma.simulate_reconnection` and
  :func:`~physicskit.plasma.sweet_parker_rate` cover magnetic
  reconnection, and :func:`~physicskit.plasma.solve_grad_shafranov`
  covers tokamak equilibria.
- The plasma :doc:`history </history/plasma_breakthroughs>` places each of
  these results in context, from Langmuir's 1928 naming of "plasma" to
  modern wakefield accelerators.
