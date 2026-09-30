Lattice Boltzmann
-----------------

A kinetic route to the Navier-Stokes equations. Instead of discretizing
the equations of fluid motion, the D2Q9 lattice Boltzmann method moves
populations of fictitious particles along nine lattice directions and
relaxes them toward a local equilibrium. The viscosity is set entirely by
the relaxation time, :math:`\nu = c_s^2(\tau - 1/2)`. Here it reproduces
Poiseuille's parabola and sheds a vortex street behind a cylinder.
