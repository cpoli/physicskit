Compressible Flow
-----------------

Once a flow moves fast enough that the fluid can no longer get out of its
own way, incompressible potential and viscous flow stop applying entirely:
the 1D Euler equations support genuinely discontinuous solutions that no
smooth velocity field can produce. The normal-shock example sweeps the
upstream Mach number through the exact Rankine-Hugoniot jump relations,
showing a supersonic flow is always driven back to subsonic on the other
side of a shock, at the cost of a sharp pressure and density jump. The Sod
shock tube then resolves that same jump dynamically, from a simple burst
initial condition, alongside the two other waves -- a rarefaction fan and a
contact discontinuity -- that a general compressible flow problem produces
alongside a shock. Watch, in the shock-tube profiles, how a first-order
finite-volume scheme smears the shock over several cells even as it gets
its speed and downstream state right.
