Navier-Stokes and Turbulence
----------------------------

The general-purpose engine behind this package's instability simulations:
a doubly periodic, pseudo-spectral solver for the full 2D incompressible
vorticity-transport equation, exact up to machine precision and aliasing in
space and 4th-order accurate in time. The first example shows its most basic
signature -- a vortex patch's peak vorticity bleeding away under viscous
diffusion, at a rate no inviscid simulation could reproduce. Another follows
Richardson's qualitative cascade -- a few large eddies folding into ever
finer filaments until viscosity takes over. The last adds steady random forcing and a large-scale drag, so that the
flow reaches a statistically steady state with a constant energy flux
through the scales. Its time-averaged energy spectrum follows Kolmogorov's
predicted -5/3 power law over the range between the forcing scale and the
drag scale.
