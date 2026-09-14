Continuous Systems
==================

Examples illustrating the continuous-time chaotic flows in
:mod:`physicskit.chaos.systems.continuous`: the Lorenz and Rossler attractors, the double pendulum, the
forced/damped Duffing oscillator, Chua's circuit (the double-scroll attractor), the restricted
three-body problem (with its five Lagrange points marked), the magnetic pendulum (fractal basins of
attraction), and the sinusoidally driven, damped pendulum (the period-doubling route to chaos). Each is
integrated with the Numba-accelerated RK4 integrator from :mod:`physicskit.chaos.core.integrators` via the
system's ``.trajectory()`` method.
