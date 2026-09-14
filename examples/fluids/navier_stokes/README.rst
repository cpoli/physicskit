Navier-Stokes and Turbulence
==============================

The general-purpose engine behind this package's instability simulations:
a doubly periodic, pseudo-spectral solver for the full 2D incompressible
vorticity-transport equation, exact up to machine precision and aliasing in
space and 4th-order accurate in time. The first example shows its most basic
signature -- a vortex patch's peak vorticity bleeding away under viscous
diffusion, at a rate no inviscid simulation could reproduce. The second
pushes the same solver into a genuinely turbulent regime: many vortices of
mixed sign interacting nonlinearly, cascading kinetic energy from the large
scales they were seeded at down to small scales where viscosity finally
dissipates it. Watch, in the energy spectrum plot, how a real inertial range
lines up with Kolmogorov's predicted -5/3 power law over more than a decade
of wavenumber -- one of the most-tested predictions in classical physics,
reproduced here from nothing but the bare incompressible equations.
