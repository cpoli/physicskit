A Tour of Fluid Flow Regimes, Checked Against Exact Solutions
=============================================================

Fluid dynamics has few exact solutions, and each one describes a single
flow regime: slow viscous flow in a channel, a thin boundary layer,
inviscid flow around a body, vortex motion, shock waves, and decaying
2D flow. This tutorial visits each regime with :mod:`physicskit.fluids`
and compares the numerical output with its exact solution. Those
comparisons tell you which solver fits which regime, and how far each
one can be trusted.

.. note:: **Units.** The functions in :mod:`physicskit.fluids` take and
   return values in whatever *consistent* unit system you pass in. SI
   is used for the dimensional examples below, and the idealized
   problems (the cylinder, vortices, Sod, Taylor-Green) are
   non-dimensional, as they are in the literature.

Which regime? The Reynolds number
---------------------------------

The Reynolds number :math:`Re = UL/\nu` compares inertia with
viscosity, and it largely decides which of the following sections
applies. Water (:math:`\nu = 10^{-6}` m²/s) flowing at 1 m/s through a
1 cm pipe:

.. code-block:: python

   import numpy as np
   import physicskit.fluids as fl

   print(round(fl.reynolds_number(velocity=1.0, length=0.01, nu=1.0e-6)))
   # 10000

That is well above the pipe-flow transition near :math:`Re \approx
2300`, so this flow would be turbulent. Laminar, viscous-dominated
solutions need much smaller :math:`Re`.

Viscous flow: Poiseuille and Blasius
------------------------------------

Between two plates a distance :math:`h` apart, a constant pressure
gradient drives the parabolic Poiseuille profile
:math:`u(y) = -\frac{1}{2\mu}\frac{dp}{dx}\,y(h - y)`, an exact solution
of the full Navier-Stokes equations. Integrating it gives the flow rate
per unit depth, :math:`Q = -\frac{dp}{dx}\frac{h^3}{12\mu}`:

.. code-block:: python

   h, mu, dpdx = 1.0e-3, 1.0e-3, -100.0          # 1 mm gap, water, 100 Pa/m
   y = np.linspace(0.0, h, 201)
   u = fl.poiseuille_flow_velocity(y, dpdx=dpdx, mu=mu, h=h)
   Q = fl.poiseuille_flow_rate(dpdx=dpdx, mu=mu, h=h)
   print(f"u_max = {u.max():.4e} m/s, Q = {Q:.4e} m^2/s, "
         f"integral of u = {np.trapezoid(u, y):.4e}, -dpdx h^3/(12 mu) = {-dpdx * h**3 / (12 * mu):.4e}")
   # u_max = 1.2500e-02 m/s, Q = 8.3333e-06 m^2/s, integral of u = 8.3331e-06, -dpdx h^3/(12 mu) = 8.3333e-06

At high :math:`Re`, viscosity only matters in a thin *boundary layer*
next to a wall (Prandtl, 1904). Over a flat plate, Blasius (1908)
reduced the layer to the ODE :math:`2f''' + f f'' = 0` with
:math:`f(0) = f'(0) = 0`, :math:`f'(\infty) = 1`.
:func:`~physicskit.fluids.blasius_solve` solves it by shooting. Its
wall-shear constant :math:`f''(0) = 0.33206` gives the skin-friction law
:math:`C_f = 0.664/\sqrt{Re_x}` and the thickness
:math:`\delta_{99} \approx 4.91\,x/\sqrt{Re_x}`:

.. code-block:: python

   blasius = fl.blasius_solve()
   print(round(float(blasius["fpp"][0]), 5), round(float(blasius["fp"][-1]), 6))
   # 0.33206 1.0

   x = np.array([0.1, 1.0])                        # air at 1 m/s, 10 cm and 1 m from the edge
   print(fl.blasius_boundary_layer_thickness(x, U_inf=1.0, nu=1.5e-5).round(5))
   # [0.00601 0.01902]

   print(fl.blasius_skin_friction_coefficient(np.array([1.0e5])).round(6))
   # [0.0021]

The layer is 6 mm thick after 10 cm and grows like :math:`\sqrt{x}`, to
19 mm after one metre.

Inviscid flow: the cylinder, d'Alembert and Kutta-Joukowski
-----------------------------------------------------------

Outside the boundary layer the flow is nearly inviscid and irrotational,
so it can be built by superposing elementary potential flows
(:class:`~physicskit.fluids.PotentialFlow`). A uniform stream plus a
doublet gives flow around a cylinder of radius :math:`a`. On its surface
the speed is :math:`2U\sin\theta`, so Bernoulli gives
:math:`C_p = 1 - 4\sin^2\theta`:

.. code-block:: python

   flow = fl.flow_past_cylinder(U_inf=1.0, radius=1.0)
   theta = np.array([0.0, np.pi / 6, np.pi / 2, np.pi])
   R = 1.0 + 1e-9                                  # just outside the surface
   print(flow.pressure_coefficient(R * np.cos(theta), R * np.sin(theta)).round(4))
   print((1 - 4 * np.sin(theta) ** 2).round(4))
   # [ 1.  0. -3.  1.]
   # [ 1.  0. -3.  1.]

The stagnation points (:math:`C_p = 1`) and the shoulder
(:math:`C_p = -3`) match exactly. Integrating the surface pressure gives
the force on the cylinder, and adding a bound vortex of circulation
:math:`\Gamma` breaks the top-bottom symmetry:

.. code-block:: python

   theta = np.linspace(0.0, 2 * np.pi, 2001)[:-1]
   dtheta = theta[1] - theta[0]
   for Gamma in [0.0, -4.0]:
       flow = fl.flow_past_cylinder(U_inf=1.0, radius=1.0, circulation=Gamma)
       cp = flow.pressure_coefficient(R * np.cos(theta), R * np.sin(theta))
       drag = -0.5 * np.sum(cp * np.cos(theta)) * dtheta     # per unit span, rho = U = a = 1
       lift = -0.5 * np.sum(cp * np.sin(theta)) * dtheta
       print(f"Gamma={Gamma:+.1f}: drag = {drag:+.6f}, lift = {lift:+.6f}, "
             f"Kutta-Joukowski = {fl.kutta_joukowski_lift(rho=1.0, U_inf=1.0, circulation=Gamma):+.1f}")

   # Gamma=+0.0: drag = +0.000000, lift = -0.000000, Kutta-Joukowski = -0.0
   # Gamma=-4.0: drag = +0.000000, lift = +4.000000, Kutta-Joukowski = +4.0

The drag is zero in both cases. This is d'Alembert's paradox (1752):
ideal flow exerts no drag on a body, and real drag comes from the
boundary layer and its separation. The lift is :math:`L = -\rho U\Gamma`
exactly (Kutta, 1902; Joukowski, 1906), which is how circulation
produces the lift on a wing.

Vortex dynamics: point vortices
-------------------------------

In 2D, a flow's vorticity can be concentrated in point vortices, which
move only through the velocities they induce on each other (Helmholtz,
1858; Kirchhoff, 1876). :class:`~physicskit.fluids.PointVortexSystem`
integrates their motion. Two equal vortices orbit their midpoint at
:math:`\Omega = \Gamma/(\pi d^2)`, and two opposite vortices travel
together in a straight line at :math:`U = \Gamma/(2\pi d)`:

.. code-block:: python

   pair = fl.PointVortexSystem(positions=[[-0.5, 0.0], [0.5, 0.0]], circulations=[1.0, 1.0])
   t, traj = pair.trajectory(dt=0.01, n_steps=1000)
   sep = traj[:, 1] - traj[:, 0]
   angle = np.unwrap(np.arctan2(sep[:, 1], sep[:, 0]))
   print(f"Omega = {angle[-1] / t[-1]:.5f}, Gamma/(pi d^2) = {1 / np.pi:.5f}")
   # Omega = 0.31831, Gamma/(pi d^2) = 0.31831

   dipole = fl.PointVortexSystem(positions=[[0.0, -0.5], [0.0, 0.5]], circulations=[1.0, -1.0])
   t, traj = dipole.trajectory(dt=0.01, n_steps=1000)
   print(f"U = {(traj[-1, 0, 0] - traj[0, 0, 0]) / t[-1]:+.5f}, Gamma/(2 pi d) = {1 / (2 * np.pi):.5f}")
   # U = -0.15915, Gamma/(2 pi d) = 0.15915

Both match to all five digits. The dipole is the 2D analogue of a smoke
ring, and a staggered double row of such vortices is the von Kármán
street (:func:`~physicskit.fluids.von_karman_vortex_street`).

Compressible flow: shocks
-------------------------

Once the Mach number exceeds 1, the flow can steepen into shocks,
discontinuities across which mass, momentum and energy are conserved
(the Rankine-Hugoniot conditions). For a normal shock at :math:`M_1 = 2`
in air (:math:`\gamma = 1.4`) the jumps are standard textbook
table entries:

.. code-block:: python

   print({k: round(v, 4) for k, v in fl.normal_shock_relations(2.0).items()})
   # {'p2_p1': 4.5, 'rho2_rho1': 2.6667, 'T2_T1': 1.6875, 'M2': 0.5774}

The Sod shock tube (Sod, *J. Comput. Phys.* 27, 1, 1978) tests a
time-dependent solver. At :math:`t = 0` a diaphragm separates gas at
:math:`(\rho, u, p) = (1, 0, 1)` from gas at :math:`(0.125, 0, 0.1)`,
and removing it produces a rarefaction, a contact discontinuity and a
shock. The exact Riemann solution has, between the contact and the
shock, :math:`p^* = 0.30313`, :math:`u^* = 0.92745` and
:math:`\rho = 0.26557`, with the shock at :math:`x = 0.8504` at
:math:`t = 0.2`:

.. code-block:: python

   sod = fl.sod_shock_tube(nx=800)
   x, rho, u, p = sod["x"], sod["rho"], sod["u"], sod["p"]
   plateau = (x > 0.72) & (x < 0.78)
   print(f"p* = {p[plateau].mean():.4f}, u* = {u[plateau].mean():.4f}, rho = {rho[plateau].mean():.4f}")
   # p* = 0.3031, u* = 0.9276, rho = 0.2701

   shock = x[np.argmax(np.abs(np.gradient(p)) * (x > 0.75))]
   print(f"shock at x = {shock:.3f}")
   # shock at x = 0.852

Pressure and velocity are right to four digits and the shock is in the
right place. The density is 2% high because the first-order
Lax-Friedrichs scheme smears the contact discontinuity over many cells,
and the smeared contact reaches into this window. That is the known
weakness of the scheme, and the reason higher-order Godunov-type schemes
exist.

Viscous decay: Taylor-Green vortices
------------------------------------

Finally, the full 2D incompressible Navier-Stokes equations.
:class:`~physicskit.fluids.NavierStokes2D` is a pseudo-spectral solver
in vorticity-streamfunction form on a periodic box. The Taylor-Green
vortex :math:`\omega = 2\sin x\sin y` (Taylor and Green, 1937) is an
exact solution: its nonlinear term vanishes identically, so it keeps its
shape and decays as :math:`e^{-2\nu t}`:

.. code-block:: python

   nu = 0.05
   solver = fl.NavierStokes2D(n=64, length=2 * np.pi, nu=nu)
   omega0 = 2 * np.sin(solver.X) * np.sin(solver.Y)
   out = solver.simulate(omega0, dt=0.01, steps=200)
   print(f"{np.abs(out['omega']).max():.6f} {2 * np.exp(-2 * nu * 2.0):.6f}")
   # 1.637462 1.637462

The peak vorticity after :math:`t = 2` matches to six digits, the
spectral accuracy expected for a smooth solution. Start the same
solver from a random vorticity field instead and it evolves into
decaying 2D turbulence; :func:`~physicskit.fluids.energy_spectrum`
measures the resulting kinetic-energy spectrum.

Where to go next
----------------

- :func:`~physicskit.fluids.kelvin_helmholtz_growth_rate` and
  :func:`~physicskit.fluids.simulate_rayleigh_taylor` cover shear and
  buoyancy instabilities.
- The fluids :doc:`history </history/fluid_breakthroughs>` traces these
  solutions from Bernoulli and d'Alembert to Kolmogorov.
